/**
 * DriftClient wrapper for QuantVault.
 * Handles perp position management, order placement, and market data.
 */

import {
  DriftClient,
  BN,
  calculateBidAskPrice,
  convertToNumber,
  PRICE_PRECISION,
  BASE_PRECISION,
  QUOTE_PRECISION,
  PositionDirection,
  OrderType,
  OrderParams,
  User,
  PerpMarketAccount,
  OraclePriceData,
  MarketType,
  getMarketOrderParams,
  getLimitOrderParams,
} from "@drift-labs/sdk";
import { PublicKey, Connection } from "@solana/web3.js";
import logger from "./logger";
import { PriorityFeeManager, FeeUrgency } from "./priority_fees";

export interface PerpPosition {
  marketIndex: number;
  symbol: string;
  baseAssetAmount: number;  // in base units (positive = long, negative = short)
  quoteEntryAmount: number;
  unrealizedPnl: number;
  fundingPnl: number;
  leverage: number;
}

export interface MarketSnapshot {
  marketIndex: number;
  symbol: string;
  oraclePrice: number;
  markPrice: number;
  bidPrice: number;
  askPrice: number;
  fundingRate: number;     // 8h rate
  fundingApr: number;      // annualized
  basisPct: number;
  openInterest: number;
}

export class DriftManager {
  constructor(
    private readonly driftClient: any,  // DriftClient
    private readonly user: any,          // User
    private readonly marketIndexes: { SOL: number; BTC: number; ETH: number } = {
      SOL: 0,
      BTC: 1,
      ETH: 2,
    },
    private readonly feeMgr?: PriorityFeeManager,
  ) {}

  async getMarketSnapshot(marketIndex: number, symbol: string): Promise<MarketSnapshot> {
    const market = this.driftClient.getPerpMarketAccount(marketIndex);
    if (!market) throw new Error(`Market ${marketIndex} not found`);

    const oracleData = this.driftClient.getOracleDataForPerpMarket(marketIndex);
    const oraclePrice = convertToNumber(oracleData.price, PRICE_PRECISION);

    const [bid, ask] = calculateBidAskPrice(market.amm, oracleData);
    const bidNum = convertToNumber(bid, PRICE_PRECISION);
    const askNum = convertToNumber(ask, PRICE_PRECISION);
    const markPrice = (bidNum + askNum) / 2;
    const basisPct = oraclePrice > 0 ? (markPrice - oraclePrice) / oraclePrice : 0;

    // Funding rate: last_24h_avg_premium_numerator / last_24h_avg_premium_denominator
    const fundingRate8h =
      convertToNumber(market.amm.lastFundingRate, new BN(10).pow(new BN(14))) / 8;
    const fundingApr = fundingRate8h * 24 * 365.25;

    const openInterest = convertToNumber(
      market.amm.baseAssetAmountWithAmm.abs(),
      BASE_PRECISION
    );

    return {
      marketIndex,
      symbol,
      oraclePrice,
      markPrice,
      bidPrice: bidNum,
      askPrice: askNum,
      fundingRate: fundingRate8h,
      fundingApr,
      basisPct,
      openInterest,
    };
  }

  async getAllMarketSnapshots(): Promise<Record<string, MarketSnapshot>> {
    const snapshots: Record<string, MarketSnapshot> = {};
    for (const [symbol, idx] of Object.entries(this.marketIndexes)) {
      try {
        snapshots[`${symbol}-PERP`] = await this.getMarketSnapshot(idx, `${symbol}-PERP`);
      } catch (err) {
        logger.warn(`Failed to get snapshot for ${symbol}: ${err}`);
      }
    }
    return snapshots;
  }

  async getPositions(): Promise<PerpPosition[]> {
    const positions: PerpPosition[] = [];
    for (const [symbol, idx] of Object.entries(this.marketIndexes)) {
      const pos = this.user.getPerpPosition(idx);
      if (!pos || pos.baseAssetAmount.isZero()) continue;

      const baseAmt = convertToNumber(pos.baseAssetAmount, BASE_PRECISION);
      const quoteEntry = convertToNumber(pos.quoteEntryAmount, QUOTE_PRECISION);
      const unrealizedPnl = convertToNumber(
        this.user.getUnrealizedPNL(true, idx, MarketType.PERP),
        QUOTE_PRECISION
      );
      const fundingPnl = convertToNumber(
        this.user.getUnrealizedFundingPNL(idx),
        QUOTE_PRECISION
      );
      const leverage = convertToNumber(this.user.getLeverage(), new BN(10000));

      positions.push({
        marketIndex: idx,
        symbol: `${symbol}-PERP`,
        baseAssetAmount: baseAmt,
        quoteEntryAmount: quoteEntry,
        unrealizedPnl,
        fundingPnl,
        leverage,
      });
    }
    return positions;
  }

  async getVaultNAV(): Promise<number> {
    const totalCollateral = this.user.getTotalCollateral();
    return convertToNumber(totalCollateral, QUOTE_PRECISION);
  }

  async openShortPerp(
    marketIndex: number,
    sizeUsd: number,
    oraclePrice: number,
    slippageBps: number = 50,
    urgency: FeeUrgency = "normal",
  ): Promise<string> {
    const baseSize = sizeUsd / oraclePrice;
    const baseAmountBn = new BN(Math.floor(baseSize * BASE_PRECISION.toNumber()));

    const slippageFactor = 1 - slippageBps / 10000;
    const limitPrice = oraclePrice * slippageFactor;
    const limitPriceBn = new BN(Math.floor(limitPrice * PRICE_PRECISION.toNumber()));

    logger.info(
      `Opening short perp market=${marketIndex}: $${sizeUsd.toFixed(2)} notional @ oracle ${oraclePrice.toFixed(4)}`
    );

    const orderParams = getLimitOrderParams({
      marketIndex,
      direction: PositionDirection.SHORT,
      baseAssetAmount: baseAmountBn,
      price: limitPriceBn,
      reduceOnly: false,
      postOnly: false,
    });

    const txParams = await this._buildTxParams(urgency, "openPerp");
    const tx = await this.driftClient.placePerpOrder(orderParams, txParams);
    logger.info(`Short perp order placed: ${tx}`);
    return tx;
  }

  async closePosition(
    marketIndex: number,
    oraclePrice: number,
    urgency: FeeUrgency = "normal",
  ): Promise<string> {
    const pos = this.user.getPerpPosition(marketIndex);
    if (!pos || pos.baseAssetAmount.isZero()) {
      logger.info(`No position to close for market ${marketIndex}`);
      return "";
    }

    const isShort = pos.baseAssetAmount.isNeg();
    const direction = isShort ? PositionDirection.LONG : PositionDirection.SHORT;
    const absAmount = pos.baseAssetAmount.abs();

    // Use oracle offset order for minimal slippage
    const slippageBps = isShort ? 50 : -50;
    const limitPrice = oraclePrice * (1 + slippageBps / 10000);
    const limitPriceBn = new BN(Math.floor(limitPrice * PRICE_PRECISION.toNumber()));

    const orderParams = getLimitOrderParams({
      marketIndex,
      direction,
      baseAssetAmount: absAmount,
      price: limitPriceBn,
      reduceOnly: true,
    });

    const txParams = await this._buildTxParams(urgency, "closePerp");
    const tx = await this.driftClient.placePerpOrder(orderParams, txParams);
    logger.info(`Close position order placed for market ${marketIndex}: ${tx}`);
    return tx;
  }

  // ── Private helpers ─────────────────────────────────────────────────────────

  /**
   * Build Drift SDK txParams with dynamic compute budget.
   * Falls back to hardcoded values if no PriorityFeeManager is configured.
   */
  private async _buildTxParams(
    urgency: FeeUrgency,
    opType: "openPerp" | "closePerp" | "rebalance" | "emergencyExit" | "default" = "default",
  ): Promise<{ computeUnits: number; computeUnitsPrice: number }> {
    if (this.feeMgr) {
      const fee = await this.feeMgr.estimate(urgency, opType);
      return { computeUnits: fee.cuLimit, computeUnitsPrice: fee.microLamportsPerCU };
    }
    // Static fallback: conservative values to ensure inclusion during moderate congestion
    const fallbackPrice: Record<FeeUrgency, number> = {
      routine:   1_000,
      normal:   10_000,
      urgent:  200_000,
      emergency: 1_000_000,
    };
    return { computeUnits: 400_000, computeUnitsPrice: fallbackPrice[urgency] };
  }

  async closeAllPositions(): Promise<void> {
    const positions = await this.getPositions();
    logger.warn(`Emergency close: closing ${positions.length} perp positions`);

    const snapshots = await this.getAllMarketSnapshots();
    for (const pos of positions) {
      const snap = snapshots[pos.symbol];
      if (!snap) continue;
      try {
        await this.closePosition(pos.marketIndex, snap.oraclePrice, "emergency");
      } catch (err) {
        logger.error(`Failed to close position for ${pos.symbol}: ${err}`);
      }
    }
  }

  async getTotalFundingPnl(): Promise<number> {
    const positions = await this.getPositions();
    return positions.reduce((sum, p) => sum + p.fundingPnl, 0);
  }

  async getFreeCollateral(): Promise<number> {
    const free = this.user.getFreeCollateral();
    return convertToNumber(free, QUOTE_PRECISION);
  }

  /**
   * Returns Drift health as a percentage (0–100).
   * Formula: Health% = 100 × (1 − maintenance_margin_req / total_collateral)
   * Health = 0% → liquidation threshold. Health = 100% → no positions.
   */
  async getHealthRate(_marketIndex: number): Promise<number> {
    const totalCollateral = this.user.getTotalCollateral("Maintenance");
    const maintenanceReq = this.user.getMaintenanceMarginRequirement();
    if (maintenanceReq.isZero()) return 100.0;
    const collateralNum = convertToNumber(totalCollateral, QUOTE_PRECISION);
    const reqNum = convertToNumber(maintenanceReq, QUOTE_PRECISION);
    if (collateralNum <= 0) return 0.0;
    return Math.max(0.0, (1.0 - reqNum / collateralNum) * 100.0);
  }
}

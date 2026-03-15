/**
 * Core rebalancer: computes delta between current and target allocations
 * and executes the necessary vault and perp adjustments.
 */

import logger from "./logger";
import { DriftManager } from "./drift";
import { VaultManager } from "./vault";
import { OrderExecutor } from "./executor";
import { StrategyEngineClient, AllocationResponse } from "./strategyClient";
import {
  perpAllocationGauge,
  hedgeRatioGauge,
  vaultApyGauge,
  positionScaleGauge,
  regimeGauge,
  regimeConfidenceGauge,
  rebalanceCounter,
  rebalanceErrorCounter,
  rebalanceDurationHistogram,
} from "./metrics";

const REGIME_NUMERIC: Record<string, number> = {
  BULL_CARRY: 0,
  SIDEWAYS: 1,
  HIGH_VOL_CRISIS: 2,
};

const MARKET_INDEXES: Record<string, number> = {
  "SOL-PERP": 0,
  "BTC-PERP": 1,
  "ETH-PERP": 2,
};

export class Rebalancer {
  private lastAllocation: AllocationResponse | null = null;
  private readonly executor: OrderExecutor;

  constructor(
    private readonly drift: DriftManager,
    private readonly vault: VaultManager,
    private readonly strategyClient: StrategyEngineClient,
    private readonly rebalanceThresholdPct: number = 0.05
  ) {
    this.executor = new OrderExecutor(drift);
  }

  async rebalance(emergencyExit: boolean = false): Promise<void> {
    const end = rebalanceDurationHistogram.startTimer();

    try {
      if (emergencyExit) {
        await this.executeEmergencyExit();
        rebalanceCounter.inc();
        return;
      }

      await this.executeNormalRebalance();
      rebalanceCounter.inc();
    } catch (err) {
      rebalanceErrorCounter.inc();
      logger.error(`Rebalance failed: ${err}`);
      throw err;
    } finally {
      end();
    }
  }

  private async executeNormalRebalance(): Promise<void> {
    // 1. Get current state
    const [allocations, regime, hedgeRatios, positions, nav] = await Promise.all([
      this.strategyClient.getAllocations(),
      this.strategyClient.getRegime(),
      this.strategyClient.getHedgeRatios(),
      this.drift.getPositions(),
      this.drift.getVaultNAV(),
    ]);

    // Update metrics
    regimeGauge.set(REGIME_NUMERIC[regime.regime] ?? 1);
    regimeConfidenceGauge.set(regime.confidence);
    positionScaleGauge.set(allocations.position_scale);
    vaultApyGauge.set(allocations.expected_blended_apr);

    for (const [sym, data] of Object.entries(hedgeRatios.hedge_ratios)) {
      hedgeRatioGauge.labels(sym).set(data.beta);
    }

    logger.info(
      `Rebalance: regime=${regime.regime}(${(regime.confidence * 100).toFixed(0)}%) ` +
      `scale=${allocations.position_scale.toFixed(2)} NAV=$${nav.toFixed(0)} ` +
      `E[APR]=${allocations.expected_blended_apr.toFixed(1)}%`
    );

    // 2. Check if rebalance is needed
    if (!this.needsRebalance(allocations)) {
      logger.info("Allocations within threshold, skipping rebalance");
      return;
    }

    // 3. Rebalance vault lending allocations via Voltr SDK
    try {
      const lendingTargets: Record<string, number> = {
        kamino: allocations.kamino_lending_pct,
        drift_spot: allocations.drift_spot_lending_pct,
      };
      await this.vault.rebalanceToTargets(lendingTargets);
    } catch (err) {
      logger.error(`Vault lending rebalance failed (continuing with perp rebalance): ${err}`);
    }

    // 4. Execute perp adjustments via idempotent executor
    const snapshots = await this.drift.getAllMarketSnapshots();

    for (const [sym, targetPct] of Object.entries(allocations.perp_allocations)) {
      const snap = snapshots[sym];
      if (!snap) continue;

      const targetNotional = nav * targetPct;
      const currentPos = positions.find((p) => p.symbol === sym);
      const currentNotional = currentPos
        ? Math.abs(currentPos.baseAssetAmount) * snap.oraclePrice
        : 0;

      const delta = targetNotional - currentNotional;

      perpAllocationGauge.labels(sym).set(targetPct);

      if (Math.abs(delta) < nav * this.rebalanceThresholdPct) {
        logger.debug(`${sym}: delta $${delta.toFixed(0)} below threshold, skip`);
        continue;
      }

      try {
        const orderId = OrderExecutor.newOrderId();
        if (delta > 0) {
          logger.info(`${sym}: increase short by $${delta.toFixed(0)}`);
          await this.executor.placeOrder({
            id: orderId,
            marketIndex: MARKET_INDEXES[sym] ?? 0,
            direction: "short",
            sizeUsd: delta,
            status: "pending",
            attempts: 0,
          }, snap.oraclePrice);
        } else {
          logger.info(`${sym}: reduce short by $${Math.abs(delta).toFixed(0)}`);
          await this.executor.placeOrder({
            id: orderId,
            marketIndex: MARKET_INDEXES[sym] ?? 0,
            direction: "long",
            sizeUsd: Math.abs(delta),
            status: "pending",
            attempts: 0,
          }, snap.oraclePrice);
        }
      } catch (err) {
        logger.error(`Failed to adjust ${sym} position: ${err}`);
      }
    }

    // Close positions for markets not in target allocation
    for (const pos of positions) {
      if (!(pos.symbol in allocations.perp_allocations) || allocations.perp_allocations[pos.symbol] === 0) {
        const snap = snapshots[pos.symbol];
        if (!snap) continue;
        logger.info(`Closing ${pos.symbol}: not in target allocation`);
        try {
          await this.executor.placeOrder({
            id: OrderExecutor.newOrderId(),
            marketIndex: pos.marketIndex,
            direction: "long",
            sizeUsd: 0,   // closePosition ignores sizeUsd — closes entire position
            status: "pending",
            attempts: 0,
          }, snap.oraclePrice);
        } catch (err) {
          logger.error(`Failed to close ${pos.symbol}: ${err}`);
        }
      }
    }

    this.executor.pruneTerminalOrders();
    this.lastAllocation = allocations;
  }

  private async executeEmergencyExit(): Promise<void> {
    logger.error("EMERGENCY EXIT: closing all perp positions");
    await this.drift.closeAllPositions();
    this.lastAllocation = null;
  }

  private needsRebalance(target: AllocationResponse): boolean {
    if (!this.lastAllocation) return true;

    // Compare total perp allocation delta
    const delta = Math.abs(
      target.total_perp_pct - this.lastAllocation.total_perp_pct
    );
    return delta > this.rebalanceThresholdPct;
  }
}

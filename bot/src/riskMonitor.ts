/**
 * On-chain risk monitoring: watches Drift health rates, vault balances,
 * and emits alerts when risk thresholds are approaching.
 */

import logger from "./logger";
import { DriftManager } from "./drift";
import { StrategyEngineClient } from "./strategyClient";
import {
  circuitBreakerGauge,
  drawdownHaltedGauge,
  drawdownPctGauge,
  fundingAprGauge,
  cascadeRiskGauge,
} from "./metrics";

const PERP_SYMBOLS = ["SOL-PERP", "BTC-PERP", "ETH-PERP"];
const MARKET_INDEXES: Record<string, number> = { "SOL-PERP": 0, "BTC-PERP": 1, "ETH-PERP": 2 };

export interface RiskStatus {
  isEmergency: boolean;
  cbTriggered: boolean;
  drawdownHalted: boolean;
  healthRates: Record<string, number>;
  positionScale: number;
  message: string;
}

export class RiskMonitor {
  constructor(
    private readonly drift: DriftManager,
    private readonly strategyClient: StrategyEngineClient,
    private readonly minHealthRate: number = 1.30
  ) {}

  async runCheck(): Promise<RiskStatus> {
    const [risk, snapshots, nav] = await Promise.all([
      this.strategyClient.getRisk(),
      this.drift.getAllMarketSnapshots(),
      this.drift.getVaultNAV(),
    ]);

    // Update strategy engine with latest market data
    for (const [symbol, snap] of Object.entries(snapshots)) {
      try {
        const update = await this.strategyClient.updateMarket({
          symbol,
          funding_apr: snap.fundingApr,
          ob_imbalance: 0,    // populated from DLOB if available
          basis_pct: snap.basisPct,
          oracle_price: snap.oraclePrice,
          open_interest: snap.openInterest,
          book_depth: 1.0,    // populated from DLOB if available
        });

        fundingAprGauge.labels(symbol).set(snap.fundingApr);
        cascadeRiskGauge.labels(symbol).set(update.cascade_risk);
      } catch (err) {
        logger.warn(`Failed to update market ${symbol}: ${err}`);
      }
    }

    // Record NAV for drawdown tracking
    if (nav > 0) {
      await this.strategyClient.recordNav(nav);
    }

    // Check on-chain health rates
    const healthRates: Record<string, number> = {};
    for (const [symbol, idx] of Object.entries(MARKET_INDEXES)) {
      try {
        const rate = await this.drift.getHealthRate(idx);
        healthRates[symbol] = rate;
        if (rate < this.minHealthRate) {
          logger.warn(`LOW HEALTH RATE for ${symbol}: ${rate.toFixed(2)} < ${this.minHealthRate}`);
        }
      } catch {
        healthRates[symbol] = Infinity;
      }
    }

    const cbTriggered = risk.circuit_breaker_state === "TRIGGERED";
    const drawdownHalted = risk.drawdown_halted;
    const isEmergency = cbTriggered || drawdownHalted;

    // Update Prometheus metrics
    circuitBreakerGauge.set(cbTriggered ? 1 : 0);
    drawdownHaltedGauge.set(drawdownHalted ? 1 : 0);

    const minHealth = Math.min(...Object.values(healthRates).filter(isFinite));
    const message = isEmergency
      ? `EMERGENCY: CB=${risk.circuit_breaker_state}, halted=${drawdownHalted}`
      : `OK: CB=${risk.circuit_breaker_state}, health=${minHealth.toFixed(2)}, scale=${risk.circuit_breaker_scale.toFixed(2)}`;

    if (isEmergency) {
      logger.error(message);
    }

    return {
      isEmergency,
      cbTriggered,
      drawdownHalted,
      healthRates,
      positionScale: risk.circuit_breaker_scale,
      message,
    };
  }
}

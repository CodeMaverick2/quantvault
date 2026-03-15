/**
 * Prometheus metrics for the QuantVault keeper bot.
 * Exported at http://localhost:9090/metrics
 */

import * as promClient from "prom-client";
import http from "http";
import logger from "./logger";

const register = new promClient.Registry();
promClient.collectDefaultMetrics({ register });

// ── Vault metrics ────────────────────────────────────────────────────────────

export const vaultNavGauge = new promClient.Gauge({
  name: "quantvault_nav_usd",
  help: "Current vault NAV in USD",
  registers: [register],
});

export const vaultApyGauge = new promClient.Gauge({
  name: "quantvault_expected_apy",
  help: "Expected blended APR from allocations",
  registers: [register],
});

// ── Strategy metrics ─────────────────────────────────────────────────────────

export const regimeGauge = new promClient.Gauge({
  name: "quantvault_regime",
  help: "Current HMM regime (0=BULL_CARRY, 1=SIDEWAYS, 2=HIGH_VOL_CRISIS)",
  registers: [register],
});

export const regimeConfidenceGauge = new promClient.Gauge({
  name: "quantvault_regime_confidence",
  help: "HMM regime prediction confidence [0,1]",
  registers: [register],
});

export const positionScaleGauge = new promClient.Gauge({
  name: "quantvault_position_scale",
  help: "Overall position scale factor after all risk adjustments [0,1]",
  registers: [register],
});

// ── Market metrics ────────────────────────────────────────────────────────────

export const fundingAprGauge = new promClient.Gauge({
  name: "quantvault_funding_apr",
  help: "Current funding APR per market",
  labelNames: ["symbol"],
  registers: [register],
});

export const cascadeRiskGauge = new promClient.Gauge({
  name: "quantvault_cascade_risk_score",
  help: "Cascade risk score per market [0,1]",
  labelNames: ["symbol"],
  registers: [register],
});

export const perpAllocationGauge = new promClient.Gauge({
  name: "quantvault_perp_allocation_pct",
  help: "Target perp allocation percentage per market",
  labelNames: ["symbol"],
  registers: [register],
});

export const hedgeRatioGauge = new promClient.Gauge({
  name: "quantvault_hedge_ratio",
  help: "Kalman filter hedge ratio (beta) per market",
  labelNames: ["symbol"],
  registers: [register],
});

// ── Risk metrics ──────────────────────────────────────────────────────────────

export const circuitBreakerGauge = new promClient.Gauge({
  name: "quantvault_circuit_breaker_active",
  help: "1 if circuit breaker is triggered, 0 otherwise",
  registers: [register],
});

export const drawdownHaltedGauge = new promClient.Gauge({
  name: "quantvault_drawdown_halted",
  help: "1 if strategy is halted due to drawdown, 0 otherwise",
  registers: [register],
});

export const drawdownPctGauge = new promClient.Gauge({
  name: "quantvault_drawdown_pct",
  help: "Current drawdown from HWM (negative value)",
  labelNames: ["period"],
  registers: [register],
});

// ── Operational metrics ───────────────────────────────────────────────────────

export const rebalanceCounter = new promClient.Counter({
  name: "quantvault_rebalances_total",
  help: "Total number of rebalance cycles executed",
  registers: [register],
});

export const rebalanceErrorCounter = new promClient.Counter({
  name: "quantvault_rebalance_errors_total",
  help: "Total number of failed rebalance cycles",
  registers: [register],
});

export const rebalanceDurationHistogram = new promClient.Histogram({
  name: "quantvault_rebalance_duration_seconds",
  help: "Duration of each rebalance cycle in seconds",
  buckets: [0.5, 1, 2, 5, 10, 30, 60],
  registers: [register],
});

export const txCounter = new promClient.Counter({
  name: "quantvault_transactions_total",
  help: "Total transactions submitted",
  labelNames: ["type"],
  registers: [register],
});

// ── HTTP server ───────────────────────────────────────────────────────────────

export function startMetricsServer(port: number = 9090): void {
  const server = http.createServer(async (req, res) => {
    if (req.url === "/metrics") {
      res.setHeader("Content-Type", register.contentType);
      res.end(await register.metrics());
    } else if (req.url === "/health") {
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ status: "ok" }));
    } else {
      res.statusCode = 404;
      res.end("Not found");
    }
  });

  server.listen(port, () => {
    logger.info(`Prometheus metrics server listening on :${port}/metrics`);
  });
}

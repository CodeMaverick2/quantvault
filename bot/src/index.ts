/**
 * QuantVault Keeper Bot — Entry Point
 *
 * Loop orchestration:
 *   - Risk check:      every 1 minute (fast loop)
 *   - Rebalance:       every 10 minutes (normal loop)
 *   - Strategy warmup: on startup
 */

import dotenv from "dotenv";
dotenv.config();

import { Connection, Keypair } from "@solana/web3.js";
import { DriftClient, Wallet, initialize, DriftEnv, BulkAccountLoader } from "@drift-labs/sdk";
import { AnchorProvider } from "@coral-xyz/anchor";

import logger from "./logger";
import { StrategyEngineClient } from "./strategyClient";
import { DriftManager } from "./drift";
import { PriorityFeeManager } from "./priority_fees";
import { VaultManager } from "./vault";
import { Rebalancer } from "./rebalancer";
import { RiskMonitor } from "./riskMonitor";
import { startMetricsServer, vaultNavGauge } from "./metrics";

const RISK_CHECK_INTERVAL_MS = 60_000;       // 1 minute
const REBALANCE_INTERVAL_MS = 600_000;       // 10 minutes

function loadKeypair(): Keypair {
  const key = process.env.KEEPER_PRIVATE_KEY;
  if (!key) throw new Error("KEEPER_PRIVATE_KEY not set in environment");
  const secretKey = JSON.parse(key) as number[];
  return Keypair.fromSecretKey(Uint8Array.from(secretKey));
}

async function main() {
  logger.info("=== QuantVault Keeper Bot Starting ===");

  // ── Config ──────────────────────────────────────────────────────────────
  const clusterEnv = (process.env.CLUSTER ?? "devnet") as any;  // DriftEnv
  const rpcUrl =
    process.env.RPC_URL ??
    (clusterEnv === "mainnet-beta"
      ? "https://api.mainnet-beta.solana.com"
      : "https://api.devnet.solana.com");
  const strategyUrl = process.env.STRATEGY_ENGINE_URL ?? "http://localhost:8000";
  const metricsPort = parseInt(process.env.METRICS_PORT ?? "9090");

  logger.info(`Cluster: ${clusterEnv}`);
  logger.info(`RPC: ${rpcUrl}`);
  logger.info(`Strategy Engine: ${strategyUrl}`);

  // ── Solana connection ────────────────────────────────────────────────────
  const connection = new Connection(rpcUrl, {
    commitment: "confirmed",
    wsEndpoint: process.env.WS_URL,
  });

  const keypair = loadKeypair();
  const wallet = new Wallet(keypair);
  logger.info(`Keeper wallet: ${keypair.publicKey.toBase58()}`);

  // ── Drift client ─────────────────────────────────────────────────────────
  const sdkConfig = initialize({ env: clusterEnv });
  const driftClient = new DriftClient({
    connection,
    wallet,
    env: clusterEnv,
    accountSubscription: {
      type: "polling",
      accountLoader: new BulkAccountLoader(connection, "confirmed", 1000),
    },
    perpMarketIndexes: [0, 1, 2],
    spotMarketIndexes: [0, 1],
    oracleInfos: sdkConfig.PERP_MARKETS.slice(0, 3).map((m: any) => ({
      publicKey: m.oracle,
      source: m.oracleSource,
    })),
  });

  await driftClient.subscribe();
  const user = driftClient.getUser();
  logger.info("Drift client subscribed");

  // ── Service wiring ───────────────────────────────────────────────────────
  const strategyClient = new StrategyEngineClient(strategyUrl);
  // Watch the Drift program account for accurate per-account priority fee estimation
  const feeMgr = new PriorityFeeManager(connection, [driftClient.program.programId]);
  const driftMgr = new DriftManager(driftClient, user, undefined, feeMgr);
  const vaultMgr = new VaultManager(
    connection,
    keypair,
    process.env.VAULT_ADDRESS ?? "",
    process.env.VOLTR_API_URL ?? "https://api.voltr.xyz"
  );
  const riskMonitor = new RiskMonitor(driftMgr, strategyClient);
  const rebalancer = new Rebalancer(driftMgr, vaultMgr, strategyClient);

  // ── Metrics server ───────────────────────────────────────────────────────
  startMetricsServer(metricsPort);

  // ── Wait for strategy engine ─────────────────────────────────────────────
  logger.info("Waiting for strategy engine to be ready...");
  let engineReady = false;
  for (let i = 0; i < 30; i++) {
    if (await strategyClient.isHealthy()) {
      engineReady = true;
      break;
    }
    await sleep(2000);
  }
  if (!engineReady) {
    logger.warn("Strategy engine not reachable — running in degraded mode (lending only)");
  } else {
    logger.info("Strategy engine connected");
  }

  // ── Main loops ────────────────────────────────────────────────────────────
  let lastRebalanceTs = 0;
  let isRunning = true;

  process.on("SIGINT", () => {
    logger.info("Received SIGINT — shutting down gracefully");
    isRunning = false;
  });

  process.on("SIGTERM", () => {
    logger.info("Received SIGTERM — shutting down gracefully");
    isRunning = false;
  });

  logger.info("=== Main loop started ===");

  while (isRunning) {
    const now = Date.now();

    // ── Risk check (every minute) ──────────────────────────────────────────
    let riskStatus;
    try {
      riskStatus = await riskMonitor.runCheck();
      const nav = await driftMgr.getVaultNAV();
      vaultNavGauge.set(nav);
    } catch (err) {
      logger.error(`Risk check failed: ${err}`);
      await sleep(RISK_CHECK_INTERVAL_MS);
      continue;
    }

    // ── Rebalance (every 10 minutes OR emergency trigger) ──────────────────
    const needsRebalance = now - lastRebalanceTs >= REBALANCE_INTERVAL_MS;

    if (riskStatus.isEmergency) {
      logger.error(`Emergency detected: ${riskStatus.message}. Executing emergency exit.`);
      try {
        await rebalancer.rebalance(true);
        lastRebalanceTs = now;
      } catch (err) {
        logger.error(`Emergency exit failed: ${err}`);
      }
    } else if (needsRebalance && engineReady) {
      logger.info("Starting scheduled rebalance cycle");
      try {
        await rebalancer.rebalance(false);
        lastRebalanceTs = now;
      } catch (err) {
        logger.error(`Scheduled rebalance failed: ${err}`);
      }
    }

    await sleep(RISK_CHECK_INTERVAL_MS);
  }

  // ── Cleanup ────────────────────────────────────────────────────────────────
  logger.info("Unsubscribing from Drift...");
  await driftClient.unsubscribe();
  logger.info("Keeper bot stopped.");
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

main().catch((err) => {
  logger.error(`Fatal error: ${err}`);
  process.exit(1);
});

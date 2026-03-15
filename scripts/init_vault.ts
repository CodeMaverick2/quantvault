#!/usr/bin/env ts-node
/**
 * init_vault.ts — Initialize the Voltr vault on devnet / mainnet.
 *
 * Uses @voltr/vault-sdk to create the vault on-chain.
 *
 * Usage:
 *   KEEPER_PRIVATE_KEY='[...]' RPC_URL='https://...' ts-node scripts/init_vault.ts
 */

import dotenv from "dotenv";
dotenv.config();

import * as fs from "fs";
import * as path from "path";
import { BN } from "@coral-xyz/anchor";
import { VaultConfig, VaultParams, VoltrClient } from "@voltr/vault-sdk";
import {
  Connection,
  Keypair,
  PublicKey,
  sendAndConfirmTransaction,
  Transaction,
} from "@solana/web3.js";

// ── Config ────────────────────────────────────────────────────────────────────

// Devnet USDC mint (mainnet: EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v)
const DEVNET_USDC_MINT  = "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU";
const MAINNET_USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v";

const ADDRESSES_OUT_PATH = path.resolve(__dirname, "../config/vault_addresses.json");

// ── Helpers ───────────────────────────────────────────────────────────────────

function loadKeypair(): Keypair {
  const raw = process.env.KEEPER_PRIVATE_KEY;
  if (!raw) throw new Error("KEEPER_PRIVATE_KEY env var is not set");
  return Keypair.fromSecretKey(Uint8Array.from(JSON.parse(raw) as number[]));
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  console.log("=== QuantVault Vault Initializer ===\n");

  const rpcUrl  = process.env.RPC_URL  ?? "https://api.devnet.solana.com";
  const cluster = process.env.CLUSTER  ?? "devnet";
  const usdcMint = cluster === "mainnet-beta" ? MAINNET_USDC_MINT : DEVNET_USDC_MINT;

  console.log(`Cluster   : ${cluster}`);
  console.log(`RPC URL   : ${rpcUrl}`);
  console.log(`USDC Mint : ${usdcMint}\n`);

  const connection = new Connection(rpcUrl, { commitment: "confirmed" });
  const managerKp  = loadKeypair();
  const vaultKp    = Keypair.generate();           // new on-chain vault account

  console.log(`Manager   : ${managerKp.publicKey.toBase58()}`);
  console.log(`Vault     : ${vaultKp.publicKey.toBase58()}\n`);

  // ── Vault parameters ──────────────────────────────────────────────────────
  const vaultConfig: VaultConfig = {
    maxCap:                           new BN("18446744073709551615"), // uncapped
    startAtTs:                        new BN(0),                      // immediate
    lockedProfitDegradationDuration:  new BN(86400),                  // 24h
    managerPerformanceFee:            2000,   // 20%
    adminPerformanceFee:              0,
    managerManagementFee:             200,    // 2%
    adminManagementFee:               0,
    redemptionFee:                    0,
    issuanceFee:                      0,
    withdrawalWaitingPeriod:          new BN(86400), // 24h redemption period
  };

  const vaultParams: VaultParams = {
    config:      vaultConfig,
    name:        "QuantVault AMDN",
    description: "Adaptive Multi-Signal Delta-Neutral",
  };

  // ── Build instruction ─────────────────────────────────────────────────────
  console.log("Building vault init instruction via @voltr/vault-sdk...");
  const client = new VoltrClient(connection);

  const ix = await client.createInitializeVaultIx(
    vaultParams,
    {
      vault:          vaultKp.publicKey,
      vaultAssetMint: new PublicKey(usdcMint),
      admin:          managerKp.publicKey,  // manager is also admin for devnet
      manager:        managerKp.publicKey,
      payer:          managerKp.publicKey,
    }
  );

  // ── Sign & send ───────────────────────────────────────────────────────────
  console.log("Signing and submitting transaction...");
  const tx = new Transaction().add(ix);
  const sig = await sendAndConfirmTransaction(connection, tx, [managerKp, vaultKp], {
    commitment: "confirmed",
  });
  console.log(`\nVault init confirmed: ${sig}`);

  // ── Persist addresses ─────────────────────────────────────────────────────
  const addresses = {
    vaultAddress:   vaultKp.publicKey.toBase58(),
    cluster,
    initializedAt:  new Date().toISOString(),
    initTx:         sig,
  };

  const outDir = path.dirname(ADDRESSES_OUT_PATH);
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(ADDRESSES_OUT_PATH, JSON.stringify(addresses, null, 2));

  console.log("\n=== Vault Initialized ===");
  console.log(`Vault address : ${vaultKp.publicKey.toBase58()}`);
  console.log(`Cluster       : ${cluster}`);
  console.log(`Init tx       : ${sig}`);
  console.log(`Addresses saved to: ${ADDRESSES_OUT_PATH}`);
  console.log("\nNext step: run scripts/add_strategies.ts");
}

main().catch((err) => {
  console.error(`\n[ERROR] ${err}`);
  process.exit(1);
});

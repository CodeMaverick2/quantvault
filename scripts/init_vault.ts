#!/usr/bin/env ts-node
/**
 * init_vault.ts — Initialize the Voltr vault on devnet / mainnet.
 *
 * Steps:
 *   1. Load keeper keypair from KEEPER_PRIVATE_KEY env var
 *   2. Connect to Solana RPC from RPC_URL env var
 *   3. POST /vault/init to the Voltr API with vault params
 *   4. Deserialize, sign, and broadcast the returned VersionedTransaction
 *   5. Print vault address + lookup table address
 *   6. Persist addresses to config/vault_addresses.json
 *
 * Usage:
 *   KEEPER_PRIVATE_KEY='[...]' RPC_URL='https://...' ts-node scripts/init_vault.ts
 */

import dotenv from "dotenv";
dotenv.config();

import * as fs from "fs";
import * as path from "path";
import axios from "axios";
import bs58 from "bs58";
import {
  Connection,
  Keypair,
  VersionedTransaction,
  Commitment,
} from "@solana/web3.js";

// ── Config ────────────────────────────────────────────────────────────────────

const VOLTR_API_BASE = process.env.VOLTR_API_URL ?? "https://api.voltr.xyz";

// Devnet USDC mint (mainnet: EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v)
const DEVNET_USDC_MINT = "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU";
const MAINNET_USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v";

const VAULT_PARAMS = {
  name: "QuantVault AMDN",
  description: "Adaptive Multi-Signal Delta-Neutral",
  config: {
    managementFeeBps: 200,
    performanceFeeBps: 2000,
    redemptionPeriodSeconds: 86400,
  },
};

const ADDRESSES_OUT_PATH = path.resolve(__dirname, "../config/vault_addresses.json");
const TX_COMMITMENT: Commitment = "confirmed";
const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 2_000;

// ── Types ─────────────────────────────────────────────────────────────────────

interface VaultInitRequest {
  name: string;
  description: string;
  asset_mint: string;
  manager: string;
  config: {
    management_fee_bps: number;
    performance_fee_bps: number;
    redemption_period_seconds: number;
  };
}

interface VaultInitResponse {
  transaction: string;        // base58 unsigned VersionedTransaction
  vault_address: string;
  lookup_table_address: string;
}

interface VaultAddresses {
  vaultAddress: string;
  lookupTableAddress: string;
  cluster: string;
  initializedAt: string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function loadKeypair(): Keypair {
  const raw = process.env.KEEPER_PRIVATE_KEY;
  if (!raw) {
    throw new Error("KEEPER_PRIVATE_KEY environment variable is not set");
  }
  const secretKey = JSON.parse(raw) as number[];
  return Keypair.fromSecretKey(Uint8Array.from(secretKey));
}

async function signAndConfirm(
  connection: Connection,
  keypair: Keypair,
  txBase58: string
): Promise<string> {
  const txBytes = bs58.decode(txBase58);
  const tx = VersionedTransaction.deserialize(txBytes);
  tx.sign([keypair]);

  const rawTx = tx.serialize();
  const sig = await connection.sendRawTransaction(rawTx, {
    skipPreflight: false,
    preflightCommitment: TX_COMMITMENT,
    maxRetries: 3,
  });

  console.log(`  Tx submitted: ${sig}`);
  console.log("  Awaiting confirmation...");

  const latestBlockhash = await connection.getLatestBlockhash(TX_COMMITMENT);
  const result = await connection.confirmTransaction(
    {
      signature: sig,
      blockhash: latestBlockhash.blockhash,
      lastValidBlockHeight: latestBlockhash.lastValidBlockHeight,
    },
    TX_COMMITMENT
  );

  if (result.value.err) {
    throw new Error(`Transaction failed on-chain: ${JSON.stringify(result.value.err)}`);
  }

  return sig;
}

async function withRetry<T>(label: string, fn: () => Promise<T>): Promise<T> {
  let lastErr: unknown;
  for (let i = 0; i < MAX_RETRIES; i++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      const delay = RETRY_DELAY_MS * Math.pow(2, i);
      console.warn(`  [WARN] ${label} attempt ${i + 1}/${MAX_RETRIES} failed: ${err}`);
      if (i < MAX_RETRIES - 1) {
        console.log(`  Retrying in ${delay}ms...`);
        await sleep(delay);
      }
    }
  }
  throw lastErr;
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  console.log("=== QuantVault Vault Initializer ===\n");

  // ── Load env ──────────────────────────────────────────────────────────────
  const rpcUrl = process.env.RPC_URL ?? "https://api.devnet.solana.com";
  const cluster = process.env.CLUSTER ?? "devnet";

  const usdcMint =
    cluster === "mainnet-beta" ? MAINNET_USDC_MINT : DEVNET_USDC_MINT;

  console.log(`Cluster   : ${cluster}`);
  console.log(`RPC URL   : ${rpcUrl}`);
  console.log(`USDC Mint : ${usdcMint}`);
  console.log(`Voltr API : ${VOLTR_API_BASE}\n`);

  // ── Connect ───────────────────────────────────────────────────────────────
  const connection = new Connection(rpcUrl, { commitment: TX_COMMITMENT });
  const keypair = loadKeypair();
  console.log(`Keeper wallet : ${keypair.publicKey.toBase58()}\n`);

  // ── Call Voltr API ────────────────────────────────────────────────────────
  const body: VaultInitRequest = {
    name: VAULT_PARAMS.name,
    description: VAULT_PARAMS.description,
    asset_mint: usdcMint,
    manager: keypair.publicKey.toBase58(),
    config: {
      management_fee_bps: VAULT_PARAMS.config.managementFeeBps,
      performance_fee_bps: VAULT_PARAMS.config.performanceFeeBps,
      redemption_period_seconds: VAULT_PARAMS.config.redemptionPeriodSeconds,
    },
  };

  console.log("Calling Voltr API POST /vault/init...");
  const initResponse = await withRetry<VaultInitResponse>("POST /vault/init", async () => {
    const { data } = await axios.post<VaultInitResponse>(
      `${VOLTR_API_BASE}/vault/init`,
      body,
      { timeout: 30_000, headers: { "Content-Type": "application/json" } }
    );
    return data;
  });

  console.log(`Vault address        : ${initResponse.vault_address}`);
  console.log(`Lookup table address : ${initResponse.lookup_table_address}`);

  // ── Sign and submit ───────────────────────────────────────────────────────
  console.log("\nSigning and submitting vault init transaction...");
  const txSig = await signAndConfirm(connection, keypair, initResponse.transaction);
  console.log(`\n  Vault init confirmed: ${txSig}`);

  // ── Persist addresses ─────────────────────────────────────────────────────
  const addresses: VaultAddresses = {
    vaultAddress: initResponse.vault_address,
    lookupTableAddress: initResponse.lookup_table_address,
    cluster,
    initializedAt: new Date().toISOString(),
  };

  const outDir = path.dirname(ADDRESSES_OUT_PATH);
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }
  fs.writeFileSync(ADDRESSES_OUT_PATH, JSON.stringify(addresses, null, 2));
  console.log(`\nAddresses saved to: ${ADDRESSES_OUT_PATH}`);

  // ── Summary ───────────────────────────────────────────────────────────────
  console.log("\n=== Vault Initialized Successfully ===");
  console.log(`Vault address        : ${initResponse.vault_address}`);
  console.log(`Lookup table address : ${initResponse.lookup_table_address}`);
  console.log(`Cluster              : ${cluster}`);
  console.log(`Init tx              : ${txSig}`);
  console.log(`\nNext step: run scripts/add_strategies.ts to register adaptors`);
}

main().catch((err) => {
  console.error(`\n[ERROR] ${err}`);
  process.exit(1);
});

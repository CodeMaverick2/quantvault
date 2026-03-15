#!/usr/bin/env ts-node
/**
 * add_strategies.ts — Register Kamino + Drift Spot adaptors on the vault.
 *
 * Steps:
 *   1. Load vault address from config/vault_addresses.json (or VAULT_ADDRESS env)
 *   2. Load keeper keypair from KEEPER_PRIVATE_KEY env var
 *   3. Connect to Solana RPC from RPC_URL env var
 *   4. For each adaptor (Kamino, Drift Spot):
 *        a. POST /vault/strategy/init with adaptor address + market params
 *        b. Deserialize, sign, and broadcast the returned VersionedTransaction
 *   5. Print strategy addresses
 *
 * Usage:
 *   KEEPER_PRIVATE_KEY='[...]' RPC_URL='https://...' ts-node scripts/add_strategies.ts
 *   VAULT_ADDRESS=<addr> ... ts-node scripts/add_strategies.ts   (override vault addr)
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

// ── Constants ─────────────────────────────────────────────────────────────────

const VOLTR_API_BASE = process.env.VOLTR_API_URL ?? "https://api.voltr.xyz";
const ADDRESSES_PATH = path.resolve(__dirname, "../config/vault_addresses.json");
const TX_COMMITMENT: Commitment = "confirmed";
const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 2_000;

// Adaptor program addresses (from config/devnet.ts)
const ADAPTORS = [
  {
    name: "Kamino Lending",
    adaptorAddress: "to6Eti9CsC5FGkAtqiPphvKD2hiQiLsS8zWiDBqBPKR",
    marketParams: {
      type: "kamino_lending",
    },
  },
  {
    name: "Drift Spot",
    adaptorAddress: "EBN93eXs5fHGBABuajQqdsKRkCgaqtJa8vEFD6vKXiP",
    marketParams: {
      type: "drift_spot",
      market_index: 0,   // USDC spot market index
    },
  },
] as const;

// ── Types ─────────────────────────────────────────────────────────────────────

interface VaultAddressesFile {
  vaultAddress: string;
  lookupTableAddress: string;
  cluster: string;
  initializedAt: string;
}

interface StrategyInitRequest {
  vault_address: string;
  manager: string;
  adaptor_address: string;
  market_params: Record<string, unknown>;
}

interface StrategyInitResponse {
  transaction: string;          // base58 unsigned VersionedTransaction
  strategy_address: string;
}

interface StrategyRecord {
  name: string;
  adaptorAddress: string;
  strategyAddress: string;
  txSignature: string;
  registeredAt: string;
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

function loadVaultAddress(): string {
  // Env override takes priority
  if (process.env.VAULT_ADDRESS) {
    console.log(`Using VAULT_ADDRESS from environment: ${process.env.VAULT_ADDRESS}`);
    return process.env.VAULT_ADDRESS;
  }

  if (!fs.existsSync(ADDRESSES_PATH)) {
    throw new Error(
      `config/vault_addresses.json not found and VAULT_ADDRESS env is not set. ` +
      `Run scripts/init_vault.ts first.`
    );
  }

  const raw = fs.readFileSync(ADDRESSES_PATH, "utf-8");
  const data = JSON.parse(raw) as VaultAddressesFile;

  if (!data.vaultAddress) {
    throw new Error(`vault_addresses.json exists but has no vaultAddress field`);
  }

  return data.vaultAddress;
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

  console.log(`    Tx submitted: ${sig}`);
  console.log("    Awaiting confirmation...");

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
  console.log("=== QuantVault Strategy Registrar ===\n");

  // ── Load env ──────────────────────────────────────────────────────────────
  const rpcUrl = process.env.RPC_URL ?? "https://api.devnet.solana.com";
  const cluster = process.env.CLUSTER ?? "devnet";

  console.log(`Cluster   : ${cluster}`);
  console.log(`RPC URL   : ${rpcUrl}`);
  console.log(`Voltr API : ${VOLTR_API_BASE}\n`);

  const connection = new Connection(rpcUrl, { commitment: TX_COMMITMENT });
  const keypair = loadKeypair();
  const vaultAddress = loadVaultAddress();

  console.log(`Keeper wallet : ${keypair.publicKey.toBase58()}`);
  console.log(`Vault address : ${vaultAddress}\n`);

  // ── Register each adaptor ─────────────────────────────────────────────────
  const results: StrategyRecord[] = [];

  for (const adaptor of ADAPTORS) {
    console.log(`Registering adaptor: ${adaptor.name}`);
    console.log(`  Adaptor address : ${adaptor.adaptorAddress}`);

    const body: StrategyInitRequest = {
      vault_address: vaultAddress,
      manager: keypair.publicKey.toBase58(),
      adaptor_address: adaptor.adaptorAddress,
      market_params: adaptor.marketParams,
    };

    let strategyAddress = "";
    let txSig = "";

    try {
      const initResponse = await withRetry<StrategyInitResponse>(
        `POST /vault/strategy/init (${adaptor.name})`,
        async () => {
          const { data } = await axios.post<StrategyInitResponse>(
            `${VOLTR_API_BASE}/vault/strategy/init`,
            body,
            { timeout: 30_000, headers: { "Content-Type": "application/json" } }
          );
          return data;
        }
      );

      strategyAddress = initResponse.strategy_address;
      console.log(`  Strategy address : ${strategyAddress}`);

      console.log("  Signing and submitting strategy init transaction...");
      txSig = await signAndConfirm(connection, keypair, initResponse.transaction);
      console.log(`  Strategy init confirmed: ${txSig}`);

      results.push({
        name: adaptor.name,
        adaptorAddress: adaptor.adaptorAddress,
        strategyAddress,
        txSignature: txSig,
        registeredAt: new Date().toISOString(),
      });

      console.log(`  [OK] ${adaptor.name} registered successfully\n`);
    } catch (err) {
      console.error(`  [ERROR] Failed to register ${adaptor.name}: ${err}\n`);
      // Continue with remaining adaptors — partial registration is usable
    }
  }

  // ── Persist / update addresses file ──────────────────────────────────────
  let existing: Record<string, unknown> = {};
  if (fs.existsSync(ADDRESSES_PATH)) {
    existing = JSON.parse(fs.readFileSync(ADDRESSES_PATH, "utf-8"));
  }

  const updated = {
    ...existing,
    strategies: results,
    strategiesUpdatedAt: new Date().toISOString(),
  };

  fs.writeFileSync(ADDRESSES_PATH, JSON.stringify(updated, null, 2));
  console.log(`Strategy addresses saved to: ${ADDRESSES_PATH}`);

  // ── Summary ───────────────────────────────────────────────────────────────
  console.log("\n=== Strategy Registration Summary ===");
  if (results.length === 0) {
    console.log("[ERROR] No strategies were registered successfully.");
    process.exit(1);
  }

  for (const r of results) {
    console.log(`\n  ${r.name}`);
    console.log(`    Adaptor  : ${r.adaptorAddress}`);
    console.log(`    Strategy : ${r.strategyAddress}`);
    console.log(`    Tx       : ${r.txSignature}`);
  }

  console.log(`\nRegistered ${results.length}/${ADAPTORS.length} strategies.`);

  if (results.length < ADAPTORS.length) {
    console.warn(
      "\n[WARN] Some strategies failed to register. " +
      "Re-run this script to retry the failed ones."
    );
    process.exit(1);
  }

  console.log("\nNext step: set VAULT_ADDRESS in .env and start the keeper bot.");
}

main().catch((err) => {
  console.error(`\n[ERROR] ${err}`);
  process.exit(1);
});

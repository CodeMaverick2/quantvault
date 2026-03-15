/**
 * VaultManager — Voltr SDK vault manager for QuantVault.
 *
 * Responsibilities:
 *   - Allocate USDC from the vault into Kamino lending strategy
 *   - Allocate USDC from the vault into Drift Spot lending strategy
 *   - Withdraw from strategies back to the vault
 *   - Read current vault strategy positions / balances
 *
 * The Voltr API returns unsigned serialized VersionedTransactions as
 * base58 strings. This manager deserializes, signs, and broadcasts them.
 */

import axios, { AxiosInstance } from "axios";
import bs58 from "bs58";
import {
  Connection,
  Keypair,
  VersionedTransaction,
  TransactionSignature,
  PublicKey,
  Commitment,
} from "@solana/web3.js";
import logger from "./logger";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface StrategyAllocation {
  strategyAddress: string;
  allocationPct: number;      // 0–1
  currentValueUsd: number;
}

interface VoltrStrategyPosition {
  strategy_address: string;
  allocation_pct: number;
  current_value_usd: number;
}

interface VoltrVaultState {
  vault_address: string;
  total_nav_usd: number;
  strategies: VoltrStrategyPosition[];
}

interface VoltrTxResponse {
  /** Unsigned serialized VersionedTransaction in base58 */
  transaction: string;
}

interface VoltrDepositParams {
  vault_address: string;
  strategy_address: string;
  amount_usdc: number;
}

interface VoltrWithdrawParams {
  vault_address: string;
  strategy_address: string;
  amount_usdc: number;
}

interface VoltrRebalanceParams {
  vault_address: string;
  targets: Record<string, number>;   // strategyAddress → target pct (0–1)
}

// ── Constants ─────────────────────────────────────────────────────────────────

const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 1_500;
const TX_CONFIRMATION_COMMITMENT: Commitment = "confirmed";

// ── Helpers ───────────────────────────────────────────────────────────────────

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Deserialize a base58-encoded serialized VersionedTransaction returned by
 * the Voltr API, sign it with the keeper wallet, and broadcast it.
 *
 * Returns the confirmed transaction signature.
 */
async function signAndSendVersionedTx(
  connection: Connection,
  keypair: Keypair,
  txBase58: string
): Promise<TransactionSignature> {
  const txBytes = bs58.decode(txBase58);
  const tx = VersionedTransaction.deserialize(txBytes);

  // Sign with the keeper keypair
  tx.sign([keypair]);

  const rawTx = tx.serialize();
  const sig = await connection.sendRawTransaction(rawTx, {
    skipPreflight: false,
    preflightCommitment: TX_CONFIRMATION_COMMITMENT,
    maxRetries: 3,
  });

  logger.info(`Tx submitted: ${sig}, awaiting confirmation...`);

  const latestBlockhash = await connection.getLatestBlockhash(TX_CONFIRMATION_COMMITMENT);
  const result = await connection.confirmTransaction(
    {
      signature: sig,
      blockhash: latestBlockhash.blockhash,
      lastValidBlockHeight: latestBlockhash.lastValidBlockHeight,
    },
    TX_CONFIRMATION_COMMITMENT
  );

  if (result.value.err) {
    throw new Error(`Transaction ${sig} failed on-chain: ${JSON.stringify(result.value.err)}`);
  }

  logger.info(`Tx confirmed: ${sig}`);
  return sig;
}

// ── VaultManager ─────────────────────────────────────────────────────────────

export class VaultManager {
  private readonly http: AxiosInstance;
  private readonly vaultAddress: PublicKey;

  constructor(
    private readonly connection: Connection,
    private readonly keypair: Keypair,
    vaultAddress: string,
    apiBaseUrl: string = "https://api.voltr.xyz"
  ) {
    if (!vaultAddress) {
      logger.warn("VAULT_ADDRESS not configured — vault lending rebalances will be skipped");
      // Use a placeholder public key; rebalanceToTargets will be a no-op when vaultAddress is empty
      this.vaultAddress = new PublicKey("11111111111111111111111111111111");
    } else {
      this.vaultAddress = new PublicKey(vaultAddress);
    }

    this.http = axios.create({
      baseURL: apiBaseUrl,
      timeout: 30_000,
      headers: { "Content-Type": "application/json" },
    });

    // Log every failed request
    this.http.interceptors.response.use(
      (res) => res,
      (err) => {
        logger.error(
          `Voltr API request failed: ${err.message} ` +
          `(url: ${err.config?.url ?? "unknown"}, status: ${err.response?.status ?? "N/A"})`
        );
        return Promise.reject(err);
      }
    );
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Fetch the current per-strategy allocations from the Voltr API.
   */
  async getStrategyAllocations(): Promise<StrategyAllocation[]> {
    return this.withRetry("getStrategyAllocations", async () => {
      const { data } = await this.http.get<VoltrVaultState>(
        `/vault/${this.vaultAddress.toBase58()}/state`
      );
      return data.strategies.map((s) => ({
        strategyAddress: s.strategy_address,
        allocationPct: s.allocation_pct,
        currentValueUsd: s.current_value_usd,
      }));
    });
  }

  /**
   * Get the total vault NAV in USD.
   */
  async getVaultNAV(): Promise<number> {
    return this.withRetry("getVaultNAV", async () => {
      const { data } = await this.http.get<VoltrVaultState>(
        `/vault/${this.vaultAddress.toBase58()}/state`
      );
      return data.total_nav_usd;
    });
  }

  /**
   * Rebalance vault holdings to the supplied target allocations.
   *
   * @param targets  Map of strategyAddress → target fraction (0–1, must sum ≤ 1)
   */
  async rebalanceToTargets(targets: Record<string, number>): Promise<void> {
    if (this.vaultAddress.toBase58() === "11111111111111111111111111111111") {
      logger.debug("Skipping vault rebalance — VAULT_ADDRESS not configured");
      return;
    }

    const totalPct = Object.values(targets).reduce((s, v) => s + v, 0);
    if (totalPct > 1.001) {
      throw new Error(
        `rebalanceToTargets: target allocations sum to ${totalPct.toFixed(4)}, must be ≤ 1`
      );
    }

    logger.info(
      `Rebalancing vault ${this.vaultAddress.toBase58()} to targets: ` +
      JSON.stringify(targets)
    );

    const params: VoltrRebalanceParams = {
      vault_address: this.vaultAddress.toBase58(),
      targets,
    };

    await this.withRetry("rebalanceToTargets", async () => {
      const { data } = await this.http.post<VoltrTxResponse>("/vault/rebalance", params);
      const sig = await signAndSendVersionedTx(this.connection, this.keypair, data.transaction);
      logger.info(`Rebalance tx confirmed: ${sig}`);
    });
  }

  /**
   * Deposit a specific USDC amount into a strategy via the Voltr API.
   *
   * @returns confirmed transaction signature
   */
  async depositToStrategy(strategyAddress: string, amountUsdc: number): Promise<string> {
    if (amountUsdc <= 0) {
      throw new Error(`depositToStrategy: amountUsdc must be positive, got ${amountUsdc}`);
    }

    logger.info(
      `Depositing $${amountUsdc.toFixed(2)} USDC into strategy ${strategyAddress}`
    );

    const params: VoltrDepositParams = {
      vault_address: this.vaultAddress.toBase58(),
      strategy_address: strategyAddress,
      amount_usdc: amountUsdc,
    };

    return this.withRetry("depositToStrategy", async () => {
      const { data } = await this.http.post<VoltrTxResponse>("/vault/strategy/deposit", params);
      const sig = await signAndSendVersionedTx(this.connection, this.keypair, data.transaction);
      logger.info(`Deposit to ${strategyAddress} confirmed: ${sig}`);
      return sig;
    });
  }

  /**
   * Withdraw a specific USDC amount from a strategy back to the vault.
   *
   * @returns confirmed transaction signature
   */
  async withdrawFromStrategy(strategyAddress: string, amountUsdc: number): Promise<string> {
    if (amountUsdc <= 0) {
      throw new Error(`withdrawFromStrategy: amountUsdc must be positive, got ${amountUsdc}`);
    }

    logger.info(
      `Withdrawing $${amountUsdc.toFixed(2)} USDC from strategy ${strategyAddress}`
    );

    const params: VoltrWithdrawParams = {
      vault_address: this.vaultAddress.toBase58(),
      strategy_address: strategyAddress,
      amount_usdc: amountUsdc,
    };

    return this.withRetry("withdrawFromStrategy", async () => {
      const { data } = await this.http.post<VoltrTxResponse>("/vault/strategy/withdraw", params);
      const sig = await signAndSendVersionedTx(this.connection, this.keypair, data.transaction);
      logger.info(`Withdrawal from ${strategyAddress} confirmed: ${sig}`);
      return sig;
    });
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  /**
   * Execute `fn` with up to MAX_RETRIES attempts, logging each failure.
   * Uses exponential back-off: RETRY_DELAY_MS * 2^attempt.
   */
  private async withRetry<T>(label: string, fn: () => Promise<T>): Promise<T> {
    let lastErr: unknown;
    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      try {
        return await fn();
      } catch (err) {
        lastErr = err;
        const delay = RETRY_DELAY_MS * Math.pow(2, attempt);
        logger.warn(
          `VaultManager.${label}: attempt ${attempt + 1}/${MAX_RETRIES} failed ` +
          `(${err}). Retrying in ${delay}ms...`
        );
        await sleep(delay);
      }
    }
    logger.error(`VaultManager.${label}: all ${MAX_RETRIES} attempts failed`);
    throw lastErr;
  }
}

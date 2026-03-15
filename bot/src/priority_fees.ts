/**
 * priority_fees.ts — Dynamic Solana priority fee estimation.
 *
 * CRITICAL for mainnet: transactions without priority fees have >95% drop
 * rate during congestion. This module estimates fees based on:
 *   1. Recent account-level priority fee history (most accurate)
 *   2. Fallback to conservative hardcoded minimums
 *
 * Research findings (2025 mainnet):
 *   Routine keeper:        1,000 – 5,000 microlamports/CU
 *   Standard operation:   10,000 – 50,000 microlamports/CU
 *   Near settlement:     100,000 – 500,000 microlamports/CU
 *   Emergency exit:    1,000,000+ microlamports/CU (use Jito bundles)
 *
 * Usage:
 *   const feeMgr = new PriorityFeeManager(connection, [driftProgramId]);
 *   const fees = await feeMgr.estimate(urgency);
 *   // Prepend to every transaction:
 *   //   ComputeBudgetProgram.setComputeUnitLimit({ units: fees.cuLimit })
 *   //   ComputeBudgetProgram.setComputeUnitPrice({ microLamports: fees.microLamportsPerCU })
 */

import {
  Connection,
  ComputeBudgetProgram,
  PublicKey,
  TransactionInstruction,
  TransactionMessage,
  VersionedTransaction,
} from "@solana/web3.js";
import logger from "./logger";

// ── Types ─────────────────────────────────────────────────────────────────────

export type FeeUrgency = "routine" | "normal" | "urgent" | "emergency";

export interface FeeEstimate {
  microLamportsPerCU: number;
  cuLimit: number;
  totalLamports: number;
  urgency: FeeUrgency;
}

// ── Constants ─────────────────────────────────────────────────────────────────

// Conservative minimums per urgency level (microlamports/CU)
const FLOOR_FEES: Record<FeeUrgency, number> = {
  routine:   1_000,
  normal:   10_000,
  urgent:  200_000,
  emergency: 1_000_000,
};

// Typical CU usage for Drift operations (simulate + add 20% buffer)
const DEFAULT_CU_LIMITS: Record<string, number> = {
  openPerp:           400_000,  // perp open + funding settlement + cross-collateral
  closePerp:          350_000,
  rebalance:          450_000,  // open + close + lending rebalance in one tx
  emergencyExit:      500_000,
  default:            300_000,
};

const BASE_LAMPORTS_PER_SIGNATURE = 5_000;
const MICROLAMPORTS_PER_LAMPORT   = 1_000_000;

// ── PriorityFeeManager ────────────────────────────────────────────────────────

export class PriorityFeeManager {
  private cachedFees: number[] = [];
  private lastFetchAt = 0;
  private readonly cacheTtlMs = 10_000;  // refresh every 10s

  constructor(
    private readonly connection: Connection,
    private readonly watchAccounts: PublicKey[] = [],   // accounts that will be write-locked
    private readonly percentile: number = 75,           // use 75th percentile
  ) {}

  /**
   * Get fee estimate for the given urgency level.
   * Fetches recent prioritization fees from the RPC and takes the
   * configured percentile. Falls back to floor values on error.
   */
  async estimate(
    urgency: FeeUrgency = "normal",
    opType: keyof typeof DEFAULT_CU_LIMITS = "default",
  ): Promise<FeeEstimate> {
    const microLamports = await this._getPercentileFee(urgency);
    const cuLimit = DEFAULT_CU_LIMITS[opType] ?? DEFAULT_CU_LIMITS.default;

    // Total fee = base_fee + (CU_limit × CU_price / 1_000_000)
    const priorityLamports = Math.ceil(cuLimit * microLamports / MICROLAMPORTS_PER_LAMPORT);
    const totalLamports = BASE_LAMPORTS_PER_SIGNATURE + priorityLamports;

    return { microLamportsPerCU: microLamports, cuLimit, totalLamports, urgency };
  }

  /**
   * Build ComputeBudget instructions to prepend to every transaction.
   * ALWAYS call this and prepend to tx instructions — never skip.
   */
  async buildComputeBudgetInstructions(
    urgency: FeeUrgency = "normal",
    opType: keyof typeof DEFAULT_CU_LIMITS = "default",
  ): Promise<TransactionInstruction[]> {
    const fee = await this.estimate(urgency, opType);
    return [
      ComputeBudgetProgram.setComputeUnitLimit({ units: fee.cuLimit }),
      ComputeBudgetProgram.setComputeUnitPrice({ microLamports: fee.microLamportsPerCU }),
    ];
  }

  /**
   * Simulate a transaction and return the actual CU consumed + 20% buffer.
   * Use this to set a tight CU limit instead of the default estimate.
   */
  async simulateAndGetCULimit(
    instructions: TransactionInstruction[],
    payer: PublicKey,
    lookupTables: any[] = [],
  ): Promise<number> {
    try {
      const { blockhash } = await this.connection.getLatestBlockhash("confirmed");
      const msg = new TransactionMessage({
        payerKey: payer,
        recentBlockhash: blockhash,
        instructions,
      }).compileToV0Message(lookupTables);
      const vtx = new VersionedTransaction(msg);

      const sim = await this.connection.simulateTransaction(vtx, {
        replaceRecentBlockhash: true,
        commitment: "confirmed",
      });

      if (sim.value.err) {
        logger.warn("simulateAndGetCULimit: simulation error:", sim.value.err);
        return DEFAULT_CU_LIMITS.default;
      }

      const consumed = sim.value.unitsConsumed ?? DEFAULT_CU_LIMITS.default;
      return Math.ceil(consumed * 1.2);  // 20% buffer
    } catch (err) {
      logger.warn("simulateAndGetCULimit failed:", err);
      return DEFAULT_CU_LIMITS.default;
    }
  }

  // ── Private ─────────────────────────────────────────────────────────────────

  private async _getPercentileFee(urgency: FeeUrgency): Promise<number> {
    const floor = FLOOR_FEES[urgency];

    // Use cache if fresh
    const now = Date.now();
    if (this.cachedFees.length > 0 && now - this.lastFetchAt < this.cacheTtlMs) {
      return Math.max(floor, this._percentile(this.cachedFees));
    }

    try {
      const recentFees = await this.connection.getRecentPrioritizationFees(
        this.watchAccounts.length > 0 ? { lockedWritableAccounts: this.watchAccounts } : undefined
      );

      const fees = recentFees
        .map((f: any) => f.prioritizationFee)
        .filter((f: number) => f > 0);

      if (fees.length === 0) {
        return floor;
      }

      this.cachedFees = fees;
      this.lastFetchAt = now;

      const p75 = this._percentile(fees);
      const result = Math.max(floor, p75);

      logger.debug(
        `PriorityFees: p${this.percentile}=${p75} floor=${floor} using=${result} (${urgency})`
      );

      return result;
    } catch (err) {
      logger.warn("Failed to fetch priority fees, using floor:", err);
      return floor;
    }
  }

  private _percentile(values: number[]): number {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const idx = Math.floor(sorted.length * (this.percentile / 100));
    return sorted[Math.min(idx, sorted.length - 1)];
  }
}

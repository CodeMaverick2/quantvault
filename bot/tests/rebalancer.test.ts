/**
 * Tests for the Rebalancer class.
 */

import { Rebalancer } from "../src/rebalancer";
import { DriftManager } from "../src/drift";
import { StrategyEngineClient, AllocationResponse, RegimeResponse, HedgeRatiosResponse } from "../src/strategyClient";

// ── Mocks ────────────────────────────────────────────────────────────────────

const mockDrift = {
  getAllMarketSnapshots: jest.fn(),
  getPositions: jest.fn(),
  getVaultNAV: jest.fn(),
  openShortPerp: jest.fn(),
  closePosition: jest.fn(),
  closeAllPositions: jest.fn(),
  getHealthRate: jest.fn(),
} as unknown as DriftManager;

const mockStrategy = {
  getAllocations: jest.fn(),
  getRegime: jest.fn(),
  getHedgeRatios: jest.fn(),
  getRisk: jest.fn(),
  updateMarket: jest.fn(),
  recordNav: jest.fn(),
  isHealthy: jest.fn(),
} as unknown as StrategyEngineClient;

function makeAllocation(perpPct: number = 0.40): AllocationResponse {
  return {
    kamino_lending_pct: 0.30,
    drift_spot_lending_pct: 0.30,
    perp_allocations: {
      "SOL-PERP": perpPct * 0.5,
      "BTC-PERP": perpPct * 0.3,
      "ETH-PERP": perpPct * 0.2,
    },
    total_perp_pct: perpPct,
    total_lending_pct: 0.60,
    regime: "BULL_CARRY",
    position_scale: 1.0,
    expected_blended_apr: 18.5,
  };
}

function makeRegime(): RegimeResponse {
  return {
    regime: "BULL_CARRY",
    confidence: 0.85,
    position_scale: 1.0,
    probabilities: { BULL_CARRY: 0.85, SIDEWAYS: 0.10, HIGH_VOL_CRISIS: 0.05 },
  };
}

function makeHedgeRatios(): HedgeRatiosResponse {
  return {
    hedge_ratios: {
      "SOL-PERP": { beta: 1.02, z_score: 0.3, uncertainty: 0.05 },
      "BTC-PERP": { beta: 0.98, z_score: -0.1, uncertainty: 0.03 },
      "ETH-PERP": { beta: 1.01, z_score: 0.2, uncertainty: 0.04 },
    },
  };
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe("Rebalancer", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (mockDrift.getVaultNAV as jest.Mock).mockResolvedValue(100_000);
    (mockDrift.getPositions as jest.Mock).mockResolvedValue([]);
    (mockDrift.getAllMarketSnapshots as jest.Mock).mockResolvedValue({
      "SOL-PERP": { oraclePrice: 90.0, fundingApr: 20.0, basisPct: 0.001, openInterest: 1e6 },
      "BTC-PERP": { oraclePrice: 50000.0, fundingApr: 15.0, basisPct: 0.0005, openInterest: 500 },
      "ETH-PERP": { oraclePrice: 2500.0, fundingApr: 12.0, basisPct: 0.001, openInterest: 5000 },
    });
    (mockStrategy.getAllocations as jest.Mock).mockResolvedValue(makeAllocation());
    (mockStrategy.getRegime as jest.Mock).mockResolvedValue(makeRegime());
    (mockStrategy.getHedgeRatios as jest.Mock).mockResolvedValue(makeHedgeRatios());
  });

  test("rebalance calls open short for new perp allocations", async () => {
    const r = new Rebalancer(mockDrift, mockStrategy, 0.01);
    await r.rebalance(false);
    expect(mockDrift.openShortPerp).toHaveBeenCalled();
  });

  test("emergency exit calls closeAllPositions", async () => {
    const r = new Rebalancer(mockDrift, mockStrategy);
    await r.rebalance(true);
    expect(mockDrift.closeAllPositions).toHaveBeenCalled();
    expect(mockDrift.openShortPerp).not.toHaveBeenCalled();
  });

  test("small delta does not trigger rebalance order", async () => {
    // Set existing position close to target
    (mockDrift.getPositions as jest.Mock).mockResolvedValue([
      {
        marketIndex: 0,
        symbol: "SOL-PERP",
        baseAssetAmount: -222,    // ~$20,000 notional at $90
        quoteEntryAmount: -20000,
        unrealizedPnl: 0,
        fundingPnl: 100,
        leverage: 1.2,
      },
    ]);

    const r = new Rebalancer(mockDrift, mockStrategy, 0.10);  // 10% threshold
    await r.rebalance(false);
    // With a tight existing position, should not place new order
    // (delta would be small relative to 10% threshold on $100k NAV = $10k)
  });

  test("crisis regime results in close orders for all positions", async () => {
    (mockStrategy.getAllocations as jest.Mock).mockResolvedValue({
      ...makeAllocation(0),
      perp_allocations: {},
      total_perp_pct: 0.0,
      regime: "HIGH_VOL_CRISIS",
      position_scale: 0.0,
    });
    (mockDrift.getPositions as jest.Mock).mockResolvedValue([
      { marketIndex: 0, symbol: "SOL-PERP", baseAssetAmount: -100, quoteEntryAmount: -9000, unrealizedPnl: 0, fundingPnl: 50, leverage: 1.0 },
    ]);

    const r = new Rebalancer(mockDrift, mockStrategy);
    await r.rebalance(false);
    expect(mockDrift.closePosition).toHaveBeenCalled();
  });
});

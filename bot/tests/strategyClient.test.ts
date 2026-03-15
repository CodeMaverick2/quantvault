/**
 * Tests for StrategyEngineClient.
 * Uses mocked axios responses.
 */

import axios from "axios";
import { StrategyEngineClient } from "../src/strategyClient";

jest.mock("axios", () => {
  const mockAxios = {
    create: jest.fn(() => mockAxios),
    get: jest.fn(),
    post: jest.fn(),
    interceptors: {
      response: { use: jest.fn() },
    },
  };
  return { default: mockAxios, ...mockAxios };
});

const mockAxiosInstance = axios as any;

describe("StrategyEngineClient", () => {
  let client: StrategyEngineClient;

  beforeEach(() => {
    client = new StrategyEngineClient("http://localhost:8000");
    jest.clearAllMocks();
  });

  test("health() returns ok status", async () => {
    mockAxiosInstance.get.mockResolvedValue({
      data: { status: "ok", hmm_fitted: true, circuit_breaker: "NORMAL", drawdown_halted: false, ts: 1700000000 },
    });
    const h = await client.health();
    expect(h.status).toBe("ok");
    expect(mockAxiosInstance.get).toHaveBeenCalledWith("/health");
  });

  test("getRegime() returns regime data", async () => {
    mockAxiosInstance.get.mockResolvedValue({
      data: {
        regime: "BULL_CARRY",
        confidence: 0.85,
        position_scale: 1.0,
        probabilities: { BULL_CARRY: 0.85, SIDEWAYS: 0.10, HIGH_VOL_CRISIS: 0.05 },
      },
    });
    const r = await client.getRegime();
    expect(r.regime).toBe("BULL_CARRY");
    expect(r.confidence).toBe(0.85);
  });

  test("getAllocations() returns allocation data", async () => {
    mockAxiosInstance.get.mockResolvedValue({
      data: {
        kamino_lending_pct: 0.30,
        drift_spot_lending_pct: 0.30,
        perp_allocations: { "SOL-PERP": 0.20 },
        total_perp_pct: 0.20,
        total_lending_pct: 0.80,
        regime: "BULL_CARRY",
        position_scale: 1.0,
        expected_blended_apr: 15.0,
      },
    });
    const a = await client.getAllocations();
    expect(a.total_perp_pct).toBe(0.20);
    expect(a.expected_blended_apr).toBe(15.0);
  });

  test("updateMarket() posts to /update-market", async () => {
    mockAxiosInstance.post.mockResolvedValue({
      data: {
        cascade_risk: 0.25,
        cascade_triggered: false,
        cascade_recommendation: "LOW_RISK: Normal operation.",
        circuit_breaker: "NORMAL",
        cb_triggered: [],
      },
    });
    const r = await client.updateMarket({
      symbol: "SOL-PERP",
      funding_apr: 20.0,
      oracle_price: 90.0,
    });
    expect(r.cascade_triggered).toBe(false);
    expect(mockAxiosInstance.post).toHaveBeenCalledWith("/update-market", expect.objectContaining({ symbol: "SOL-PERP" }));
  });

  test("isHealthy() returns true when status is ok", async () => {
    mockAxiosInstance.get.mockResolvedValue({
      data: { status: "ok", hmm_fitted: true, circuit_breaker: "NORMAL", drawdown_halted: false, ts: 0 },
    });
    expect(await client.isHealthy()).toBe(true);
  });

  test("isHealthy() returns false on error", async () => {
    mockAxiosInstance.get.mockRejectedValue(new Error("Connection refused"));
    expect(await client.isHealthy()).toBe(false);
  });
});

/**
 * HTTP client for the Python strategy engine.
 * Abstracts all communication with the FastAPI server.
 */

import axios, { AxiosInstance } from "axios";
import logger from "./logger";

export interface RegimeResponse {
  regime: "BULL_CARRY" | "SIDEWAYS" | "HIGH_VOL_CRISIS";
  confidence: number;
  position_scale: number;
  probabilities: Record<string, number>;
  note?: string;
}

export interface HedgeRatiosResponse {
  hedge_ratios: Record<
    string,
    { beta: number; z_score: number; uncertainty: number }
  >;
}

export interface AllocationResponse {
  kamino_lending_pct: number;
  drift_spot_lending_pct: number;
  perp_allocations: Record<string, number>;
  perp_directions: Record<string, "SHORT" | "LONG">;
  total_perp_pct: number;
  total_lending_pct: number;
  regime: string;
  position_scale: number;
  expected_blended_apr: number;
}

export interface MarketUpdateRequest {
  symbol: string;
  funding_apr: number;
  ob_imbalance?: number;
  basis_pct?: number;
  oracle_price?: number;
  open_interest?: number;
  book_depth?: number;
  lending_apr?: number;
  liq_volume_1h?: number;   // liquidation volume last hour (USD) — feeds leading indicator engine
}

export interface MarketUpdateResponse {
  cascade_risk: number;
  cascade_triggered: boolean;
  cascade_recommendation: string;
  circuit_breaker: string;
  cb_triggered: string[];
}

export interface RiskResponse {
  circuit_breaker_state: string;
  circuit_breaker_scale: number;
  drawdown_halted: boolean;
  drawdown_hwm: number;
  active_circuit_breaker_events: Array<{
    trigger: string;
    triggered_at: number;
    duration_secs: number;
    value: number;
    threshold: number;
  }>;
  market_cascade_risks: Record<string, number>;
}

export interface HealthResponse {
  status: string;
  hmm_fitted: boolean;
  circuit_breaker: string;
  drawdown_halted: boolean;
  ts: number;
}

export class StrategyEngineClient {
  private client: AxiosInstance;
  private readonly baseUrl: string;

  constructor(baseUrl: string = "http://localhost:8000") {
    this.baseUrl = baseUrl;
    this.client = axios.create({
      baseURL: baseUrl,
      timeout: 10_000,
      headers: { "Content-Type": "application/json" },
    });

    this.client.interceptors.response.use(
      (res) => res,
      (err) => {
        logger.error(
          `Strategy engine request failed: ${err.message} (url: ${err.config?.url})`
        );
        return Promise.reject(err);
      }
    );
  }

  async health(): Promise<HealthResponse> {
    const { data } = await this.client.get<HealthResponse>("/health");
    return data;
  }

  async getRegime(): Promise<RegimeResponse> {
    const { data } = await this.client.get<RegimeResponse>("/regime");
    return data;
  }

  async getHedgeRatios(): Promise<HedgeRatiosResponse> {
    const { data } =
      await this.client.get<HedgeRatiosResponse>("/hedge-ratios");
    return data;
  }

  async getAllocations(): Promise<AllocationResponse> {
    const { data } = await this.client.get<AllocationResponse>("/allocations");
    return data;
  }

  async updateMarket(req: MarketUpdateRequest): Promise<MarketUpdateResponse> {
    const { data } = await this.client.post<MarketUpdateResponse>(
      "/update-market",
      req
    );
    return data;
  }

  async getRisk(): Promise<RiskResponse> {
    const { data } = await this.client.get<RiskResponse>("/risk");
    return data;
  }

  async recordNav(navUsd: number): Promise<void> {
    await this.client.post("/record-nav", { nav_usd: navUsd });
  }

  async updateLendingRates(kaminoApr: number, driftSpotApr: number): Promise<void> {
    await this.client.post("/lending-rates", {
      kamino_apr: kaminoApr,
      drift_spot_apr: driftSpotApr,
    });
  }

  async isHealthy(): Promise<boolean> {
    try {
      const h = await this.health();
      return h.status === "ok";
    } catch {
      return false;
    }
  }
}

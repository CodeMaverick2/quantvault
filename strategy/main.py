"""
QuantVault Strategy Engine — FastAPI REST server.

Endpoints consumed by the TypeScript keeper bot:
  GET  /health           — health check
  GET  /regime           — current market regime + confidence
  GET  /hedge-ratios     — Kalman filter hedge ratios per market
  GET  /allocations      — target portfolio allocations
  POST /update-market    — push new market data (funding rate, cascade signals)
  GET  /risk             — current risk status (drawdown, circuit breaker state)
  POST /record-nav       — record NAV for drawdown tracking
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .models.hmm_regime import HMMRegimeClassifier, MarketRegime, RegimePrediction
from .models.kalman_hedge import MultiAssetHedgeManager
from .optimization.allocation import (
    AllocationConfig,
    AllocationResult,
    DynamicAllocationOptimizer,
    MarketYieldData,
)
from .risk.circuit_breakers import CircuitBreaker, CircuitBreakerConfig
from .risk.drawdown_control import DrawdownController
from .signals.cascade_risk import CascadeRiskScorer
from .signals.drift_data import DriftDataClient
from .signals.funding_features import build_features, get_hmm_feature_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("strategy_engine")

# ── Config ──────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent.parent / "config" / "strategy.yaml"
MODEL_DIR = Path(__file__).parent.parent / "models"

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

SYMBOLS = [m["symbol"] for m in CFG["markets"]["perp"]]

# ── State ────────────────────────────────────────────────────────────────────

hmm = HMMRegimeClassifier(
    n_states=CFG["hmm"]["n_states"],
    n_iter=CFG["hmm"]["n_iter"],
    covariance_type=CFG["hmm"]["covariance_type"],
    model_path=MODEL_DIR / "hmm_regime.pkl",
)

hedge_manager = MultiAssetHedgeManager(
    symbols=SYMBOLS,
    process_noise=CFG["kalman"]["process_noise"],
    observation_noise=CFG["kalman"]["observation_noise"],
)

cascade_scorer = CascadeRiskScorer(
    trigger_threshold=CFG["risk"]["cascade_risk_threshold"]
)

circuit_breaker = CircuitBreaker(
    config=CircuitBreakerConfig(
        max_negative_funding_apr=CFG["risk"]["max_negative_funding_apr"],
        max_basis_pct=CFG["risk"]["max_basis_pct"],
        max_oracle_deviation_pct=CFG["risk"]["max_oracle_deviation_pct"],
        cascade_risk_threshold=CFG["risk"]["cascade_risk_threshold"],
        min_book_depth_ratio=0.30,
        cooldown_secs=CFG["risk"]["circuit_breaker_cooldown_minutes"] * 60,
    )
)

drawdown_ctrl = DrawdownController(
    daily_halt_pct=CFG["risk"]["daily_drawdown_halt"],
    weekly_halt_pct=CFG["risk"]["weekly_drawdown_halt"],
    monthly_review_pct=CFG["risk"]["monthly_drawdown_review"],
)

optimizer = DynamicAllocationOptimizer(
    config=AllocationConfig(
        min_lending_pct=CFG["allocation"]["min_lending_pct"],
        max_perp_pct=CFG["allocation"]["max_perp_pct"],
        max_single_perp_pct=CFG["markets"]["perp"][0]["max_allocation_pct"],
        target_funding_apr_threshold=CFG["strategy"]["target_funding_apr_threshold"],
    )
)

# Runtime market state (updated by /update-market endpoint)
_market_state: dict[str, dict] = {
    sym: {
        "funding_apr": 0.0,
        "lending_apr": 0.0,
        "cascade_risk": 0.0,
        "ob_imbalance": 0.0,
        "basis_pct": 0.0,
        "oracle_price": 0.0,
        "updated_at": 0.0,
    }
    for sym in SYMBOLS
}
_latest_regime: Optional[RegimePrediction] = None
_kamino_apr: float = 5.0
_drift_spot_apr: float = 7.0
_hmm_feature_buffer: list[np.ndarray] = []  # rolling window for HMM predictions
HMM_BUFFER_SIZE = 48  # 48 hours of hourly data


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Try to load pre-trained model
    try:
        hmm.load()
        logger.info("Loaded pre-trained HMM model")
    except FileNotFoundError:
        logger.info("No pre-trained HMM found — will train on first data fetch")

    # Background task: fetch market data every 10 minutes
    task = asyncio.create_task(_market_data_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="QuantVault Strategy Engine",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class MarketUpdateRequest(BaseModel):
    symbol: str
    funding_apr: float
    ob_imbalance: float = 0.0
    basis_pct: float = 0.0
    oracle_price: float = 0.0
    open_interest: float = 0.0
    book_depth: float = 1.0
    lending_apr: float = 0.0


class NavUpdateRequest(BaseModel):
    nav_usd: float
    timestamp: Optional[float] = None


class LendingRatesRequest(BaseModel):
    kamino_apr: float
    drift_spot_apr: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "hmm_fitted": hmm.is_fitted,
        "circuit_breaker": circuit_breaker.state.value,
        "drawdown_halted": drawdown_ctrl.is_halted,
        "ts": int(time.time()),
    }


@app.get("/regime")
async def get_regime():
    if not hmm.is_fitted or len(_hmm_feature_buffer) < 10:
        return {
            "regime": "SIDEWAYS",
            "confidence": 0.33,
            "position_scale": 0.5,
            "probabilities": {"BULL_CARRY": 0.33, "SIDEWAYS": 0.34, "HIGH_VOL_CRISIS": 0.33},
            "note": "Model not yet fitted or insufficient data",
        }

    try:
        X = np.array(_hmm_feature_buffer[-HMM_BUFFER_SIZE:])
        prediction = hmm.predict(X)
        global _latest_regime
        _latest_regime = prediction
        return {
            "regime": prediction.regime.name,
            "confidence": prediction.confidence,
            "position_scale": prediction.position_scale,
            "probabilities": prediction.probabilities,
        }
    except Exception as exc:
        logger.exception("Regime prediction failed: %s", exc)
        return {"regime": "SIDEWAYS", "confidence": 0.5, "position_scale": 0.5, "error": str(exc)}


@app.get("/hedge-ratios")
async def get_hedge_ratios():
    ratios = hedge_manager.get_hedge_ratios()
    states = {}
    for sym in SYMBOLS:
        state = hedge_manager.get_state(sym)
        states[sym] = {
            "beta": ratios.get(sym, 1.0),
            "z_score": state.z_score if state else 0.0,
            "uncertainty": float(np.sqrt(hedge_manager._trackers[sym].state_covariance[0, 0]))
            if sym in hedge_manager._trackers else 1.0,
        }
    return {"hedge_ratios": states}


@app.get("/allocations")
async def get_allocations():
    global _latest_regime

    if _latest_regime is None:
        regime = MarketRegime.SIDEWAYS
        confidence = 0.5
    else:
        regime = _latest_regime.regime
        confidence = _latest_regime.confidence

    dd_state = drawdown_ctrl.record_nav(
        _get_current_nav(), time.time()
    ) if _get_current_nav() > 0 else None
    dd_scale = dd_state.position_scale if dd_state else 1.0

    cb_scale = circuit_breaker.get_position_multiplier()

    markets = [
        MarketYieldData(
            symbol=sym,
            funding_apr=_market_state[sym]["funding_apr"],
            lending_apr=_market_state[sym]["lending_apr"],
            is_perp=True,
            cascade_risk=_market_state[sym]["cascade_risk"],
        )
        for sym in SYMBOLS
    ]

    result = optimizer.compute(
        markets=markets,
        regime=regime,
        regime_confidence=confidence,
        drawdown_scale=dd_scale,
        cb_scale=cb_scale,
        kamino_apr=_kamino_apr,
        drift_spot_apr=_drift_spot_apr,
    )

    return {
        "kamino_lending_pct": result.kamino_lending_pct,
        "drift_spot_lending_pct": result.drift_spot_lending_pct,
        "perp_allocations": result.perp_allocations,
        "total_perp_pct": result.total_perp_pct,
        "total_lending_pct": result.total_lending_pct,
        "regime": result.regime,
        "position_scale": result.position_scale,
        "expected_blended_apr": result.expected_blended_apr,
    }


@app.post("/update-market")
async def update_market(req: MarketUpdateRequest):
    if req.symbol not in _market_state:
        raise HTTPException(status_code=400, detail=f"Unknown symbol: {req.symbol}")

    # Update cascade risk scorer history
    cascade_scorer.update_history(
        funding_rate=req.funding_apr / (365.25 * 24),
        open_interest=req.open_interest,
        liquidation_volume=0.0,
        book_depth=req.book_depth,
    )

    cascade_input = cascade_scorer.build_input_from_market(
        ob_imbalance=req.ob_imbalance,
        funding_rate=req.funding_apr / (365.25 * 24),
        open_interest=req.open_interest,
        book_depth=req.book_depth,
        basis_pct=req.basis_pct,
    )
    cascade_result = cascade_scorer.score(cascade_input)

    _market_state[req.symbol].update({
        "funding_apr": req.funding_apr,
        "ob_imbalance": req.ob_imbalance,
        "basis_pct": req.basis_pct,
        "oracle_price": req.oracle_price,
        "lending_apr": req.lending_apr,
        "cascade_risk": cascade_result.score,
        "updated_at": time.time(),
    })

    # Update Kalman filter with new price
    if req.oracle_price > 0:
        spread = req.basis_pct * req.oracle_price
        state = hedge_manager.update(req.symbol, req.oracle_price, spread)

    # Update circuit breaker
    avg_funding = np.mean([v["funding_apr"] for v in _market_state.values()])
    avg_basis = np.mean([abs(v["basis_pct"]) for v in _market_state.values()])
    cb_state, triggered = circuit_breaker.check(
        funding_apr=avg_funding,
        basis_pct=avg_basis,
        oracle_deviation_pct=abs(req.basis_pct),
        cascade_risk_score=cascade_result.score,
        book_depth_ratio=req.book_depth,
    )

    return {
        "cascade_risk": cascade_result.score,
        "cascade_triggered": cascade_result.triggered,
        "cascade_recommendation": cascade_result.recommendation,
        "circuit_breaker": cb_state.value,
        "cb_triggered": triggered,
    }


@app.get("/risk")
async def get_risk():
    active_events = circuit_breaker.active_events
    return {
        "circuit_breaker_state": circuit_breaker.state.value,
        "circuit_breaker_scale": circuit_breaker.get_position_multiplier(),
        "drawdown_halted": drawdown_ctrl.is_halted,
        "drawdown_hwm": drawdown_ctrl.high_water_mark,
        "active_circuit_breaker_events": [
            {
                "trigger": e.trigger_name,
                "triggered_at": e.triggered_at,
                "duration_secs": e.duration_secs,
                "value": e.value,
                "threshold": e.threshold,
            }
            for e in active_events
        ],
        "market_cascade_risks": {
            sym: state["cascade_risk"]
            for sym, state in _market_state.items()
        },
    }


@app.post("/record-nav")
async def record_nav(req: NavUpdateRequest):
    state = drawdown_ctrl.record_nav(req.nav_usd, req.timestamp)
    return {
        "nav": state.current_nav,
        "hwm": state.high_water_mark,
        "daily_drawdown_pct": state.daily_drawdown_pct,
        "weekly_drawdown_pct": state.weekly_drawdown_pct,
        "position_scale": state.position_scale,
        "is_halted": state.is_halted,
        "halt_reason": state.halt_reason,
    }


@app.post("/lending-rates")
async def update_lending_rates(req: LendingRatesRequest):
    global _kamino_apr, _drift_spot_apr
    _kamino_apr = req.kamino_apr
    _drift_spot_apr = req.drift_spot_apr
    return {"kamino_apr": _kamino_apr, "drift_spot_apr": _drift_spot_apr}


# ── Background task ───────────────────────────────────────────────────────────

async def _market_data_loop():
    """Fetch market data from Drift API and update HMM feature buffer."""
    while True:
        try:
            await _fetch_and_update_market_data()
        except Exception as exc:
            logger.exception("Market data fetch failed: %s", exc)
        await asyncio.sleep(600)  # 10 minutes


async def _fetch_and_update_market_data():
    async with DriftDataClient() as client:
        dfs = await client.get_multi_market_funding(SYMBOLS, limit=HMM_BUFFER_SIZE + 50)

    for sym, df in dfs.items():
        if df.empty:
            continue

        from .signals.funding_features import build_features, get_hmm_feature_matrix
        enriched = build_features(df)
        X, idx = get_hmm_feature_matrix(enriched)

        if len(X) >= 10:
            # Update HMM feature buffer with latest row
            _hmm_feature_buffer.append(X[-1])
            if len(_hmm_feature_buffer) > HMM_BUFFER_SIZE * len(SYMBOLS):
                _hmm_feature_buffer.pop(0)

            # Train HMM if we have enough data and it's not yet fitted
            if not hmm.is_fitted and len(X) >= 50:
                logger.info("Training HMM on %d observations for %s", len(X), sym)
                hmm.fit(X)
                hmm.save()

        # Update market state with latest funding rate
        if not df.empty:
            latest = df.iloc[-1]
            apr = float(latest.get("apr", 0.0))
            basis = float(latest.get("basis_pct", 0.0))
            if sym in _market_state:
                _market_state[sym]["funding_apr"] = apr
                _market_state[sym]["basis_pct"] = basis
                _market_state[sym]["updated_at"] = time.time()


def _get_current_nav() -> float:
    """Get the most recently recorded NAV (from history)."""
    if drawdown_ctrl._nav_history:
        return drawdown_ctrl._nav_history[-1][1]
    return 0.0


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "strategy.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )

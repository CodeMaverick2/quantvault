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
import collections
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
from .signals.funding_persistence import FundingPersistenceScorer
from .signals.ar_funding_predictor import ARFundingPredictor
from .signals.tod_optimizer import TimeOfDayOptimizer
from .signals.multi_horizon_forecaster import MultiHorizonForecaster
from .signals.regime_transition import RegimeTransitionForecaster
from .signals.leading_indicators import LeadingIndicatorEngine

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


def _validate_config(cfg: dict) -> None:
    """Validate required config keys are present and have sane values."""
    required_keys = [
        ("hmm", "n_states"),
        ("hmm", "n_iter"),
        ("hmm", "covariance_type"),
        ("kalman", "process_noise"),
        ("kalman", "observation_noise"),
        ("risk", "cascade_risk_threshold"),
        ("risk", "max_negative_funding_apr"),
        ("risk", "max_basis_pct"),
        ("risk", "max_oracle_deviation_pct"),
        ("risk", "daily_drawdown_halt"),
        ("risk", "weekly_drawdown_halt"),
        ("risk", "monthly_drawdown_review"),
        ("risk", "circuit_breaker_cooldown_minutes"),
        ("allocation", "min_lending_pct"),
        ("allocation", "max_perp_pct"),
        ("strategy", "target_funding_apr_threshold"),
        ("markets", "perp"),
    ]
    for section, key in required_keys:
        if section not in cfg or key not in cfg[section]:
            raise KeyError(f"Missing required config: [{section}][{key}]")
    if not cfg["markets"]["perp"]:
        raise ValueError("markets.perp must contain at least one market")
    if cfg["hmm"]["n_states"] < 3:
        raise ValueError(f"hmm.n_states must be >= 3, got {cfg['hmm']['n_states']}")


_validate_config(CFG)

SYMBOLS = [m["symbol"] for m in CFG["markets"]["perp"]]

# ── State ────────────────────────────────────────────────────────────────────

# Slow HMM: trained on weekly rolling window (primary regime signal)
hmm = HMMRegimeClassifier(
    n_states=CFG["hmm"]["n_states"],
    n_iter=CFG["hmm"]["n_iter"],
    covariance_type=CFG["hmm"]["covariance_type"],
    model_path=MODEL_DIR / "hmm_regime.pkl",
)

# Fast HMM: 4-hour buffer captures intraday regime shifts
# Only enter perps when both fast + slow agree on BULL_CARRY
hmm_fast = HMMRegimeClassifier(
    n_states=CFG["hmm"]["n_states"],
    n_iter=200,   # fewer iterations for speed
    covariance_type=CFG["hmm"]["covariance_type"],
    model_path=MODEL_DIR / "hmm_fast.pkl",
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
_latest_fast_regime: Optional[RegimePrediction] = None
_kamino_apr: float = 5.0
_drift_spot_apr: float = 7.0

HMM_BUFFER_SIZE = 48       # slow HMM: 48 hours
HMM_FAST_BUFFER_SIZE = 6   # fast HMM: 6 hours (intraday shifts)
_hmm_feature_buffer: collections.deque = collections.deque(maxlen=HMM_BUFFER_SIZE)
_hmm_fast_buffer: collections.deque = collections.deque(maxlen=HMM_FAST_BUFFER_SIZE)

# Funding persistence scorer (tracks consecutive positive funding per symbol)
_persistence_scorer = FundingPersistenceScorer()

# AR(4) funding predictor — only enter when AR prediction exceeds breakeven
_ar_predictor = ARFundingPredictor(breakeven_apr=22.0)

# Time-of-day optimizer — concentrates positions in historically rich UTC windows
_tod_optimizer = TimeOfDayOptimizer()

# Predictive signal stack (Layers 8–10)
# Feed these every cycle; graceful degradation if not enough history
_mh_forecaster = MultiHorizonForecaster()        # AR(4) iterative 1/6/24/72h forecast
_transition_forecaster = RegimeTransitionForecaster()  # HMM A^N transition probs
_leading_engine = LeadingIndicatorEngine()        # OI + basis + liquidation leads

# Threading/async safety locks
_market_state_lock = asyncio.Lock()
_regime_lock = asyncio.Lock()
_hmm_lock = asyncio.Lock()

# Periodic HMM retraining tracker (retrain weekly)
_hmm_last_retrain_ts: float = 0.0
HMM_RETRAIN_INTERVAL_SECS = 7 * 86400  # 1 week


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Try to load pre-trained models
    try:
        hmm.load()
        logger.info("Loaded pre-trained slow HMM model")
    except FileNotFoundError:
        logger.info("No pre-trained slow HMM found — will train on first data fetch")

    try:
        hmm_fast.load()
        logger.info("Loaded pre-trained fast HMM model")
    except FileNotFoundError:
        logger.info("No pre-trained fast HMM found — will train on first data fetch")

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
    liq_volume_1h: float = 0.0   # liquidation volume in last hour (USD, optional)


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
        X = np.array(list(_hmm_feature_buffer)[-HMM_BUFFER_SIZE:])
        prediction = hmm.predict(X)
        global _latest_regime
        _latest_regime = prediction

        # Fast HMM prediction (intraday)
        fast_info: dict = {}
        global _latest_fast_regime
        if hmm_fast.is_fitted and len(_hmm_fast_buffer) >= 3:
            X_fast = np.array(_hmm_fast_buffer)
            fast_pred = hmm_fast.predict(X_fast)
            _latest_fast_regime = fast_pred
            fast_info = {
                "fast_regime": fast_pred.regime.name,
                "fast_confidence": fast_pred.confidence,
                "dual_agreement": fast_pred.regime == prediction.regime,
            }

        return {
            "regime": prediction.regime.name,
            "confidence": prediction.confidence,
            "position_scale": prediction.position_scale,
            "probabilities": prediction.probabilities,
            **fast_info,
        }
    except Exception as exc:
        logger.exception("Regime prediction failed: %s", exc)
        return {"regime": "SIDEWAYS", "confidence": 0.5, "position_scale": 0.5, "error": str(exc)}


@app.get("/signals")
async def get_signals():
    """Returns full signal stack state: persistence, dual-HMM, circuit breakers."""
    persistence_results = _persistence_scorer.score_all(SYMBOLS)
    return {
        "funding_persistence": {
            sym: {
                "persistence_score": r.persistence_score,
                "momentum_quality": r.momentum_quality,
                "basis_alignment": r.basis_alignment,
                "consecutive_positive": r.consecutive_positive,
                "entry_quality": r.entry_quality,
                "allow_entry": r.allow_entry,
            }
            for sym, r in persistence_results.items()
        },
        "ar_predictions": {
            sym: {
                "predicted_apr": p.predicted_apr,
                "lower_95": p.lower_95,
                "upper_95": p.upper_95,
                "prediction_std": p.prediction_std,
                "ar_coefficients": p.ar_coefficients,
                "allow_entry": p.allow_entry,
            }
            for sym, p in _ar_predictor.predict_all(SYMBOLS).items()
        },
        "slow_regime": _latest_regime.regime.name if _latest_regime else "UNKNOWN",
        "fast_regime": _latest_fast_regime.regime.name if _latest_fast_regime else "UNKNOWN",
        "dual_agreement": (
            _latest_fast_regime.regime == _latest_regime.regime
            if _latest_fast_regime and _latest_regime else False
        ),
        "tod": {
            "multiplier": _tod_optimizer.current_multiplier(),
            **vars(_tod_optimizer.get_multiplier()),
        },
        "circuit_breaker": circuit_breaker.state.value,
        "cb_scale": circuit_breaker.get_position_multiplier(),
        "predictive_signals": {
            sym: {
                "trajectory": mhf.trajectory.value if hasattr(mhf, "trajectory") else "FLAT",
                "pre_position": mhf.pre_position_signal if hasattr(mhf, "pre_position_signal") else False,
                "exit_signal": mhf.exit_signal if hasattr(mhf, "exit_signal") else False,
                "f1h": mhf.forecasts[1].predicted_apr if 1 in mhf.forecasts else None,
                "f6h": mhf.forecasts[6].predicted_apr if 6 in mhf.forecasts else None,
                "f24h": mhf.forecasts[24].predicted_apr if 24 in mhf.forecasts else None,
            }
            for sym, mhf in _mh_forecaster.forecast_all(SYMBOLS).items()
        },
        "regime_transition": (lambda f: {
            "warning": f.warning.value,
            "trans_6h": f.transition_probs.get(6, 0.0),
            "trans_24h": f.transition_probs.get(24, 0.0),
            "expected_hours": f.expected_transition_hours,
            "crisis_approach_24h": f.crisis_approach_prob_24h,
        })(_transition_forecaster.forecast()),
        "leading_indicators": {
            sym: {
                "signal": r.signal.value,
                "composite_score": r.composite_score,
                "pre_position_carry": r.pre_position_carry,
                "pre_exit_carry": r.pre_exit_carry,
                "pre_position_inverse": r.pre_position_inverse,
            }
            for sym, r in _leading_engine.analyze_all(SYMBOLS).items()
        },
        "market_states": {
            sym: {
                "funding_apr": state["funding_apr"],
                "cascade_risk": state["cascade_risk"],
                "basis_pct": state["basis_pct"],
                "updated_at": state["updated_at"],
            }
            for sym, state in _market_state.items()
        },
    }


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

    # Gather persistence + AR predictions for each market
    persistence_results = _persistence_scorer.score_all(SYMBOLS)
    ar_predictions = _ar_predictor.predict_all(SYMBOLS)

    markets = [
        MarketYieldData(
            symbol=sym,
            # Use AR predicted APR when available (reduces entering on spikes)
            funding_apr=ar_predictions[sym].predicted_apr
                if ar_predictions[sym].allow_entry
                else _market_state[sym]["funding_apr"],
            lending_apr=_market_state[sym]["lending_apr"],
            is_perp=True,
            cascade_risk=_market_state[sym]["cascade_risk"],
            persistence_score=persistence_results[sym].entry_quality if sym in persistence_results else 1.0,
            consecutive_positive=persistence_results[sym].consecutive_positive if sym in persistence_results else 0,
        )
        for sym in SYMBOLS
    ]

    # Fast regime for dual-timeframe consensus
    fast_r = _latest_fast_regime.regime if _latest_fast_regime else None
    fast_c = _latest_fast_regime.confidence if _latest_fast_regime else 1.0

    # Time-of-day multiplier: concentrate perp sizing in high-yield UTC windows
    tod_mult = _tod_optimizer.current_multiplier()

    # Update regime transition forecaster with current regime (feeds A^N matrix)
    if _latest_regime is not None:
        _transition_forecaster.update(
            regime=_latest_regime.regime.name,
            confidence=_latest_regime.confidence,
        )

    # Gather predictive signals (all gracefully degrade on cold start)
    mh_forecasts = _mh_forecaster.forecast_all(SYMBOLS)
    transition_forecast = _transition_forecaster.forecast()
    leading_signals = _leading_engine.analyze_all(SYMBOLS)

    result = optimizer.compute(
        markets=markets,
        regime=regime,
        regime_confidence=confidence,
        drawdown_scale=dd_scale,
        cb_scale=cb_scale,
        kamino_apr=_kamino_apr,
        drift_spot_apr=_drift_spot_apr,
        fast_regime=fast_r,
        fast_confidence=fast_c,
        tod_multiplier=tod_mult,
        multi_horizon_forecasts=mh_forecasts,
        regime_transition=transition_forecast,
        leading_indicators=leading_signals,
    )

    return {
        "kamino_lending_pct": result.kamino_lending_pct,
        "drift_spot_lending_pct": result.drift_spot_lending_pct,
        "perp_allocations": result.perp_allocations,
        "perp_directions": result.perp_directions,  # SHORT or LONG (inverse carry)
        "total_perp_pct": result.total_perp_pct,
        "total_lending_pct": result.total_lending_pct,
        "regime": result.regime,
        "position_scale": result.position_scale,
        "expected_blended_apr": result.expected_blended_apr,
        "tod_multiplier": tod_mult,
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
        oracle_price=req.oracle_price,
        symbol=req.symbol,
    )

    # Feed leading indicators (OI + basis + liquidation pre-signals)
    if req.open_interest > 0 or req.oracle_price > 0:
        perp_price = req.oracle_price * (1.0 + req.basis_pct)
        _leading_engine.update(
            symbol=req.symbol,
            oi=req.open_interest,
            perp_price=perp_price if perp_price > 0 else req.oracle_price,
            spot_price=req.oracle_price,
            liq_volume_1h=req.liq_volume_1h,
        )

    # Keep multi-horizon forecaster current with live funding
    _mh_forecaster.update(req.symbol, req.funding_apr)

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
    global _hmm_last_retrain_ts

    async with DriftDataClient() as client:
        dfs = await client.get_multi_market_funding(SYMBOLS, limit=HMM_BUFFER_SIZE + 50)

    for sym, df in dfs.items():
        if df.empty:
            continue

        enriched = build_features(df)
        X, idx = get_hmm_feature_matrix(enriched)

        if len(X) >= 10:
            async with _hmm_lock:
                # Slow HMM buffer (48h)
                _hmm_feature_buffer.append(X[-1])
                # Fast HMM buffer (6h) — append last min(6, len(X)) rows
                for row in X[-HMM_FAST_BUFFER_SIZE:]:
                    _hmm_fast_buffer.append(row)

                now = time.time()
                # Retrain slow HMM weekly
                should_retrain = (
                    not hmm.is_fitted
                    or (now - _hmm_last_retrain_ts) >= HMM_RETRAIN_INTERVAL_SECS
                )
                if should_retrain and len(X) >= 50:
                    logger.info(
                        "Retraining slow HMM on %d observations (sym=%s)",
                        len(X), sym,
                    )
                    hmm.fit(X)
                    hmm.save()
                    _hmm_last_retrain_ts = now

                # Train fast HMM whenever we have enough data (low n_iter, fast)
                if not hmm_fast.is_fitted and len(X) >= 20:
                    hmm_fast.fit(X)
                    hmm_fast.save()

        # Update market state, persistence scorer, and AR predictor
        if sym in _market_state:
            latest = df.iloc[-1]
            apr = float(latest.get("apr", 0.0))
            basis = float(latest.get("basis_pct", 0.0))
            z_score = float(latest.get("fr_z_24h", 0.0)) if "fr_z_24h" in latest else 0.0

            # Feed all historical APRs to AR predictor and multi-horizon forecaster for warmup
            for _, row in df.iterrows():
                row_apr = float(row.get("apr", 0.0))
                _ar_predictor.update(sym, row_apr)
                _mh_forecaster.update(sym, row_apr)

            # Update persistence scorer and time-of-day optimizer
            _persistence_scorer.update(sym, apr, basis_pct=basis, z_score=z_score)
            _tod_optimizer.update(apr)

            async with _market_state_lock:
                _market_state[sym]["funding_apr"] = apr
                _market_state[sym]["basis_pct"] = basis
                _market_state[sym]["updated_at"] = time.time()


def _get_current_nav() -> float:
    """Get the most recently recorded NAV (from history)."""
    if drawdown_ctrl._nav_history:
        return drawdown_ctrl._nav_history[-1][1]
    return 0.0


# ── Email report endpoint ─────────────────────────────────────────────────────

import smtplib
import datetime
import urllib.request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def _fetch_prometheus_metrics() -> dict:
    """Fetch Prometheus metrics from keeper-bot container."""
    metrics_url = os.getenv("KEEPER_METRICS_URL", "http://localhost:9090")
    result = {}
    try:
        with urllib.request.urlopen(f"{metrics_url}/metrics", timeout=5) as resp:
            for line in resp.read().decode().splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        result[parts[0]] = float(parts[-1])
                    except Exception:
                        pass
    except Exception as e:
        logger.warning("Could not fetch Prometheus metrics: %s", e)
    return result


def _build_report_html() -> tuple[str, str]:
    """Build HTML + plain-text report from current in-memory state."""
    now = datetime.datetime.utcnow()
    ts = now.strftime("%Y-%m-%d %H:%M UTC")

    # ── Regime ────────────────────────────────────────────────────────────────
    if _latest_regime:
        regime_name = _latest_regime.regime.name
        regime_conf = _latest_regime.confidence
        regime_probs = {s.name: p for s, p in (_latest_regime.probabilities or {}).items()}
    else:
        regime_name, regime_conf, regime_probs = "UNKNOWN", 0.0, {}

    cb_state = circuit_breaker.state.value
    dd_halted = drawdown_ctrl.is_halted
    dd_state = drawdown_ctrl.record_nav(_get_current_nav(), time.time()) if _get_current_nav() > 0 else None
    dd_scale = dd_state.position_scale if dd_state else 1.0
    pos_scale = circuit_breaker.get_position_multiplier() * dd_scale

    # ── Market state (cascade_risk already stored by update-market) ───────────
    funding_rates = {sym: s.get("funding_apr", 0) for sym, s in _market_state.items()}
    cascade_risks = {sym: float(s.get("cascade_risk", 0)) for sym, s in _market_state.items()}

    # ── Prometheus (keeper bot metrics) ───────────────────────────────────────
    prom = _fetch_prometheus_metrics()
    nav_usd       = prom.get("quantvault_vault_nav_usd", _get_current_nav())
    rebal_total   = int(prom.get("quantvault_rebalances_total", 0))
    rebal_errors  = int(prom.get("quantvault_rebalance_errors_total", 0))
    error_rate    = (rebal_errors / rebal_total * 100) if rebal_total > 0 else 0.0

    # ── Allocation snapshot ───────────────────────────────────────────────────
    try:
        persistence_results = _persistence_scorer.score_all(list(_market_state.keys()))
        alloc_result = optimizer.compute(
            markets=[
                MarketYieldData(
                    symbol=sym,
                    funding_apr=s.get("funding_apr", 0),
                    lending_apr=s.get("lending_apr", 0),
                    is_perp=True,
                    cascade_risk=float(s.get("cascade_risk", 0)),
                    persistence_score=persistence_results[sym].persistence_score if sym in persistence_results else 0.5,
                    consecutive_positive=persistence_results[sym].consecutive_positive if sym in persistence_results else 0,
                )
                for sym, s in _market_state.items()
            ],
            regime=_latest_regime.regime if _latest_regime else MarketRegime.SIDEWAYS,
            regime_confidence=regime_conf,
            drawdown_scale=dd_scale,
            cb_scale=circuit_breaker.get_position_multiplier(),
            kamino_apr=5.0,
            drift_spot_apr=5.0,
        )
        exp_apr     = alloc_result.expected_blended_apr
        perp_allocs = alloc_result.perp_allocations
        k_pct       = alloc_result.kamino_lending_pct
        d_pct       = alloc_result.drift_spot_lending_pct
    except Exception:
        exp_apr, perp_allocs, k_pct, d_pct = 0.0, {}, 0.0, 0.0

    # ── Status ────────────────────────────────────────────────────────────────
    REGIME_EMOJI = {"BULL_CARRY": "🟢", "SIDEWAYS": "🟡", "HIGH_VOL_CRISIS": "🔴"}
    r_emoji = REGIME_EMOJI.get(regime_name, "⚪")
    overall = ("✅ OPERATIONAL" if cb_state == "NORMAL" and not dd_halted
               else "⚠️ DEGRADED" if cb_state != "TRIGGERED" else "🔴 HALTED")

    # ── Build HTML ────────────────────────────────────────────────────────────
    fund_rows = "".join(
        f"<tr><td>{sym}</td>"
        f"<td class=\"{'green' if apr > 0.05 else 'yellow'}\">{apr*100:.2f}%</td>"
        f"<td class=\"{'red' if cascade_risks.get(sym,0) > 0.7 else 'yellow' if cascade_risks.get(sym,0) > 0.5 else 'green'}\">{cascade_risks.get(sym,0):.2f}</td>"
        f"<td>{perp_allocs.get(sym, 0)*100:.1f}%</td></tr>"
        for sym, apr in sorted(funding_rates.items(), key=lambda x: -x[1])
    )
    alloc_bars = "".join(
        f"<div class='signal-row'><span class='signal-label'>{sym} Perp</span>"
        f"<span class='signal-value'>{pct*100:.1f}%</span></div>"
        f"<div class='alloc-bar'><div class='alloc-fill' style='width:{pct*100:.1f}%'></div></div><br/>"
        for sym, pct in perp_allocs.items()
    )

    html = f"""<!DOCTYPE html><html><head><style>
body{{font-family:-apple-system,Arial,sans-serif;background:#0a0a0f;color:#e0e0e0;margin:0;padding:0}}
.container{{max-width:700px;margin:0 auto;padding:20px}}
.header{{background:linear-gradient(135deg,#1a1a2e,#16213e);padding:24px;border-radius:12px;margin-bottom:20px;border:1px solid #2a2a4a}}
.header h1{{margin:0;font-size:22px;color:#a78bfa}}.header .sub{{color:#6b7280;font-size:13px;margin-top:4px}}
.card{{background:#111827;border:1px solid #1f2937;border-radius:10px;padding:18px;margin-bottom:16px}}
.card h2{{margin:0 0 14px 0;font-size:15px;color:#9ca3af;text-transform:uppercase;letter-spacing:.05em}}
.metric-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.metric{{background:#0f172a;border-radius:8px;padding:12px}}
.metric .label{{font-size:11px;color:#6b7280;margin-bottom:4px}}
.metric .value{{font-size:20px;font-weight:bold;color:#f3f4f6}}
.green{{color:#4ade80!important}}.yellow{{color:#fbbf24!important}}.red{{color:#f87171!important}}
.signal-row{{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #1f2937;font-size:13px}}
.signal-row:last-child{{border-bottom:none}}.signal-label{{color:#9ca3af}}.signal-value{{color:#f3f4f6;font-weight:500}}
.alloc-bar{{background:#1f2937;border-radius:4px;height:8px;margin-top:6px;overflow:hidden}}
.alloc-fill{{height:100%;border-radius:4px;background:linear-gradient(90deg,#7c3aed,#a78bfa)}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;color:#6b7280;font-weight:normal;padding:6px 8px;border-bottom:1px solid #1f2937}}
td{{padding:8px;border-bottom:1px solid #111827}}
.footer{{text-align:center;color:#4b5563;font-size:11px;padding:16px}}
</style></head><body><div class="container">
<div class="header"><h1>⚡ QuantVault System Report</h1>
<div class="sub">{ts} &nbsp;|&nbsp; {overall}</div></div>
<div class="card"><h2>Portfolio Overview</h2><div class="metric-grid">
<div class="metric"><div class="label">VAULT NAV</div><div class="value green">${nav_usd:,.2f}</div></div>
<div class="metric"><div class="label">EXPECTED APR</div><div class="value {'green' if exp_apr >= 10 else 'yellow'}">{exp_apr:.1f}%</div></div>
<div class="metric"><div class="label">REBALANCES</div><div class="value">{rebal_total}</div><div style="font-size:11px;color:#6b7280">Error rate: {error_rate:.1f}%</div></div>
<div class="metric"><div class="label">CIRCUIT BREAKER</div><div class="value {'green' if cb_state=='NORMAL' else 'red'}">{cb_state}</div></div>
</div></div>
<div class="card"><h2>Market Regime</h2>
<div class="signal-row"><span class="signal-label">Active Regime</span><span class="signal-value">{r_emoji} {regime_name} ({regime_conf*100:.0f}%)</span></div>
<div class="signal-row"><span class="signal-label">Position Scale</span><span class="signal-value">{pos_scale:.2f}x</span></div>
{"".join(f'<div class="signal-row"><span class="signal-label">P({r})</span><span class="signal-value">{v*100:.1f}%</span></div>' for r,v in regime_probs.items())}
</div>
<div class="card"><h2>Live Funding Rates</h2>
<table><tr><th>Market</th><th>Funding APR</th><th>Cascade Risk</th><th>Target Alloc</th></tr>
{fund_rows}</table></div>
<div class="card"><h2>Current Allocation</h2>
<div class="signal-row"><span class="signal-label">Kamino Lending</span><span class="signal-value">{k_pct*100:.1f}%</span></div>
<div class="alloc-bar"><div class="alloc-fill" style="width:{k_pct*100:.1f}%"></div></div><br/>
<div class="signal-row"><span class="signal-label">Drift Spot Lending</span><span class="signal-value">{d_pct*100:.1f}%</span></div>
<div class="alloc-bar"><div class="alloc-fill" style="width:{d_pct*100:.1f}%"></div></div><br/>
{alloc_bars}</div>
<div class="footer">QuantVault Keeper Bot &nbsp;|&nbsp; Generated {ts}<br/>Next report in ~1 hour</div>
</div></body></html>"""

    text = (f"QuantVault Report — {ts}\n\nSTATUS: {overall}\n"
            f"NAV: ${nav_usd:,.2f} | APR: {exp_apr:.1f}%\n"
            f"Regime: {regime_name} ({regime_conf*100:.0f}%) | Scale: {pos_scale:.2f}x\n"
            f"CB: {cb_state} | DD Halt: {dd_halted}\n"
            f"Rebalances: {rebal_total} | Error rate: {error_rate:.1f}%\n\n"
            f"FUNDING:\n" +
            "\n".join(f"  {sym}: {apr*100:.2f}% APR" for sym, apr in funding_rates.items()) +
            f"\n\nALLOCATION:\n  Kamino: {k_pct*100:.1f}%  Drift Spot: {d_pct*100:.1f}%\n" +
            "\n".join(f"  {sym}: {pct*100:.1f}%" for sym, pct in perp_allocs.items()))

    return html, text, ts


@app.post("/send-report")
async def send_report():
    """Build and email the hourly performance report. Called by cron-job.org."""
    to_email   = os.getenv("REPORT_EMAIL_TO")
    from_email = os.getenv("REPORT_EMAIL_FROM")
    email_pass = os.getenv("REPORT_EMAIL_PASS")

    if not all([to_email, from_email, email_pass]):
        raise HTTPException(status_code=503, detail="Email env vars not configured (REPORT_EMAIL_TO / FROM / PASS)")

    try:
        html, text, ts = _build_report_html()
    except Exception as e:
        logger.exception("Report build failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Report build failed: {e}")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"QuantVault Report {ts}"
    msg["From"]    = from_email
    msg["To"]      = to_email
    msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.login(from_email, email_pass)
            server.sendmail(from_email, to_email, msg.as_string())
        logger.info("Hourly report sent to %s", to_email)
        return {"status": "ok", "sent_to": to_email, "ts": ts}
    except Exception as e:
        logger.error("Failed to send report email: %s", e)
        raise HTTPException(status_code=500, detail=f"SMTP error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "strategy.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )

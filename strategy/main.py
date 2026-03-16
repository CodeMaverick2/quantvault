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
from .signals.cointegration import CointegrationEngine

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
        # Env var takes priority over YAML — lets devnet bypass the CB on garbage rates
        max_negative_funding_apr=float(os.getenv(
            "MAX_NEGATIVE_FUNDING_APR",
            str(CFG["risk"]["max_negative_funding_apr"]),
        )),
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
# Rolling 6h funding rate history per symbol for peak/deterioration tracking.
# At 1 update/minute, 360 entries = 6h.
_funding_history_6h: dict[str, collections.deque] = {
    sym: collections.deque(maxlen=360) for sym in SYMBOLS
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

# Stat arb: Johansen cointegration across correlated perp pairs
_cointegration_engine = CointegrationEngine(
    entry_z=CFG["cointegration"]["entry_z_score"],
    exit_z=CFG["cointegration"]["exit_z_score"],
    stop_z=CFG["cointegration"]["stop_loss_z_score"],
)
_stat_arb_signals: dict[str, object] = {}   # latest signals keyed by pair

# Threading/async safety locks
_market_state_lock = asyncio.Lock()
_regime_lock = asyncio.Lock()
_hmm_lock = asyncio.Lock()

# Periodic HMM retraining tracker (retrain weekly)
_hmm_last_retrain_ts: float = 0.0
HMM_RETRAIN_INTERVAL_SECS = 7 * 86400  # 1 week

# Report analytics: NAV snapshots for hourly change tracking
_engine_start_time: float = time.time()
_report_nav_snapshots: collections.deque = collections.deque(maxlen=48)  # 48h of hourly history


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
        "stat_arb": {
            pair: {
                "z_score": float(sig.z_score),
                "action": sig.action,
                "beta": float(sig.beta),
                "confidence": float(sig.confidence),
            }
            for pair, sig in _stat_arb_signals.items()
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
            funding_peak_6h=_market_state[sym].get("funding_peak_6h"),
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

    # ── Stat arb overlay ─────────────────────────────────────────────────────
    # Carve out up to max_stat_arb_pct from lending when a pair has an active signal.
    # Confidence-weighted so only high-conviction trades get meaningful allocation.
    stat_arb_allocs: dict[str, float] = {}
    stat_arb_budget_used = 0.0
    max_stat_arb = optimizer.config.max_stat_arb_pct  # 15% max
    for pair, sig in _stat_arb_signals.items():
        if sig.action in ("ENTER_LONG_SPREAD", "ENTER_SHORT_SPREAD") and sig.confidence > 0.5:
            alloc = min(max_stat_arb * sig.confidence, max_stat_arb - stat_arb_budget_used)
            if alloc > 0.01:
                stat_arb_allocs[pair] = alloc
                stat_arb_budget_used += alloc

    # Reduce lending proportionally to fund stat arb
    if stat_arb_budget_used > 0:
        lending_reduction = min(stat_arb_budget_used, result.total_lending_pct - optimizer.config.min_lending_pct)
        if lending_reduction > 0:
            ratio = 1.0 - (lending_reduction / result.total_lending_pct)
            result.kamino_lending_pct *= ratio
            result.drift_spot_lending_pct *= ratio
            result.total_lending_pct *= ratio
        result.stat_arb_allocations = stat_arb_allocs
        logger.info("StatArb overlay: %d active pairs, budget=%.1f%%",
                    len(stat_arb_allocs), stat_arb_budget_used * 100)

    return {
        "kamino_lending_pct": result.kamino_lending_pct,
        "drift_spot_lending_pct": result.drift_spot_lending_pct,
        "perp_allocations": result.perp_allocations,
        "perp_directions": result.perp_directions,  # SHORT or LONG (inverse carry)
        "stat_arb_allocations": result.stat_arb_allocations,
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

    # Track 6h rolling funding history for deterioration detection
    if req.symbol in _funding_history_6h and req.funding_apr != 0.0:
        _funding_history_6h[req.symbol].append(req.funding_apr)
    funding_peak_6h: Optional[float] = (
        max(_funding_history_6h[req.symbol])
        if req.symbol in _funding_history_6h and len(_funding_history_6h[req.symbol]) >= 3
        else None
    )

    _market_state[req.symbol].update({
        "funding_apr": req.funding_apr,
        "ob_imbalance": req.ob_imbalance,
        "basis_pct": req.basis_pct,
        "oracle_price": req.oracle_price,
        "lending_apr": req.lending_apr,
        "cascade_risk": cascade_result.score,
        "funding_peak_6h": funding_peak_6h,
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

    # Update stat arb engine with latest oracle prices across all pairs
    if req.oracle_price > 0:
        _prices = {sym: s.get("oracle_price", 0.0) for sym, s in _market_state.items()}
        _prices[req.symbol] = req.oracle_price
        for pair in CFG["cointegration"]["pairs"]:
            sym_a, sym_b = pair[0], pair[1]
            pa, pb = _prices.get(sym_a, 0.0), _prices.get(sym_b, 0.0)
            if pa > 0 and pb > 0:
                sig = _cointegration_engine.update(
                    sym_a, sym_b,
                    float(np.log(pa)), float(np.log(pb)),
                )
                _stat_arb_signals[sig.pair] = sig

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


def _gemini_commentary(state_dict: dict) -> str:
    """Call Gemini 2.0 Flash for a plain-English summary of current strategy state."""
    api_key = os.getenv("GEMINI_API_KEY")
    logger.info("Gemini: API key %s", "SET" if api_key else "NOT SET — skipping commentary")
    if not api_key:
        return ""
    import urllib.request as _urlreq, json as _json
    lines = [
        f"Vault NAV: ${state_dict['nav']:.2f} ({state_dict['nav_change_str']})",
        f"Regime: {state_dict['regime']} at {state_dict['regime_conf']*100:.0f}% confidence",
        f"Circuit Breaker: {state_dict['cb_state']}",
        f"Expected APR: {state_dict['exp_apr']:.1f}%",
        f"Position scale: {state_dict['pos_scale']:.2f}x (0=all cash, 1=full exposure)",
        f"Allocation: {state_dict['k_pct']*100:.0f}% Kamino lending, {state_dict['d_pct']*100:.0f}% Drift spot, rest in perps",
        f"Funding rates: {', '.join(f'{s}: {r}' for s,r in state_dict['funding_rates'].items())}",
        f"Rebalances completed: {state_dict['rebal_total']} | Error rate: {state_dict['error_rate']:.1f}%",
    ]
    prompt = (
        "You are a DeFi fund manager writing a short hourly update for QuantVault, a delta-neutral yield vault on Solana. "
        "The reader is a non-technical investor who wants to understand what their money is doing right now.\n\n"
        "Write a clear, conversational update in 4-6 sentences covering:\n"
        "- What the vault is doing right now and what yield it is targeting\n"
        "- Why it made this allocation decision (mention the regime, risk signals, funding rates)\n"
        "- Whether performance improved or declined vs last hour and why\n"
        "- Any active risks or circuit breakers and what they mean in plain English\n"
        "- A one-line outlook for the next hour based on current signals\n\n"
        "Be specific with numbers. Write in plain English, no jargon, no markdown, no bullet points — just paragraphs.\n\n"
        "Current data:\n"
        + "\n".join(lines)
    )
    payload = _json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.4},
    }).encode()
    model = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    logger.info("Gemini: calling model=%s", model)
    req = _urlreq.Request(
        url, data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "QuantVault/1.0"},
        method="POST",
    )
    try:
        with _urlreq.urlopen(req, timeout=15) as resp:
            data = _json.loads(resp.read())
        candidate = data["candidates"][0]
        finish_reason = candidate.get("finishReason", "UNKNOWN")
        text = candidate["content"]["parts"][0]["text"].strip()
        logger.info("Gemini: commentary generated (%d chars, finishReason=%s)", len(text), finish_reason)
        return text
    except _urlreq.HTTPError as http_err:
        body = http_err.read().decode("utf-8", errors="replace")
        logger.error("Gemini HTTP %s: %s", http_err.code, body)
        return ""
    except Exception as exc:
        logger.error("Gemini unexpected error: %s", exc)
        return ""


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

    # ── Market state ──────────────────────────────────────────────────────────
    funding_rates = {sym: s.get("funding_apr", 0) for sym, s in _market_state.items()}
    cascade_risks = {sym: float(s.get("cascade_risk", 0)) for sym, s in _market_state.items()}

    # ── Prometheus ────────────────────────────────────────────────────────────
    prom = _fetch_prometheus_metrics()
    nav_usd      = prom.get("quantvault_vault_nav_usd", _get_current_nav())
    rebal_total  = int(prom.get("quantvault_rebalances_total", 0))
    rebal_errors = int(prom.get("quantvault_rebalance_errors_total", 0))
    error_rate   = (rebal_errors / rebal_total * 100) if rebal_total > 0 else 0.0

    # ── NAV change vs last snapshot ───────────────────────────────────────────
    nav_change_usd = 0.0
    nav_change_pct = 0.0
    if _report_nav_snapshots:
        prev_nav = _report_nav_snapshots[-1][1]
        if prev_nav > 0:
            nav_change_usd = nav_usd - prev_nav
            nav_change_pct = nav_change_usd / prev_nav * 100
    _report_nav_snapshots.append((time.time(), nav_usd))

    # Build change string: "+$2.10 (+1.2%)" or "-$0.50 (-0.3%)"
    if nav_change_usd == 0.0 and not _report_nav_snapshots:
        nav_change_str = "first report"
    else:
        sign = "+" if nav_change_usd >= 0 else ""
        nav_change_str = f"{sign}${nav_change_usd:,.2f} ({sign}{nav_change_pct:.2f}%) vs last hour"

    # ── Uptime ────────────────────────────────────────────────────────────────
    uptime_secs = time.time() - _engine_start_time
    uptime_h = int(uptime_secs // 3600)
    uptime_m = int((uptime_secs % 3600) // 60)
    uptime_str = f"{uptime_h}h {uptime_m}m"

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

    # ── Labels ────────────────────────────────────────────────────────────────
    REGIME_DESC = {
        "BULL_CARRY":      "Longs are paying shorts to hold positions. Optimal conditions for funding capture.",
        "SIDEWAYS":        "Funding rates are low or mixed. Capital rotated to safer lending positions.",
        "HIGH_VOL_CRISIS": "High volatility detected. Strategy is in inverse carry mode, collecting negative funding.",
        "UNKNOWN":         "Insufficient data to classify regime. Holding lending-only defensive posture.",
    }
    CB_DESC = {
        "NORMAL":       "All systems nominal.",
        "WARNING":      "Minor signal triggered. Monitoring closely.",
        "TRIGGERED":    "Risk threshold breached. All derivative positions closed.",
        "COOLING_DOWN": "Post-event cooldown. Positions scaling back up linearly.",
    }
    SCALE_DESC = (
        "Full deployment" if pos_scale >= 0.95
        else f"{pos_scale*100:.0f}% deployment (risk controls active)"
        if pos_scale > 0
        else "0% — all capital in lending (risk halt)"
    )
    overall_status = ("OPERATIONAL" if cb_state == "NORMAL" and not dd_halted
                      else "DEGRADED" if cb_state != "TRIGGERED" else "HALTED")
    status_color = "#1a7f4b" if overall_status == "OPERATIONAL" else ("#b45309" if overall_status == "DEGRADED" else "#991b1b")
    nav_dir = "+" if nav_change_usd >= 0 else ""
    nav_dir_color = "#16a34a" if nav_change_usd >= 0 else "#dc2626"

    # ── AR predictions per symbol ─────────────────────────────────────────────
    ar_preds = {}
    for sym in list(_market_state.keys()):
        try:
            pred = _ar_predictor.predict(sym)
            ar_preds[sym] = pred.predicted_apr if pred else None
        except Exception:
            ar_preds[sym] = None

    # ── Persistence per symbol ────────────────────────────────────────────────
    persist_scores = {}
    consec_positive = {}
    try:
        pr = _persistence_scorer.score_all(list(_market_state.keys()))
        for sym, res in pr.items():
            persist_scores[sym] = res.persistence_score
            consec_positive[sym] = res.consecutive_positive
    except Exception:
        pass

    # ── CB recent events ──────────────────────────────────────────────────────
    recent_cb = [e.trigger_name for e in circuit_breaker.recent_events(3) if e.is_active]

    # ── Gemini AI commentary ──────────────────────────────────────────────────
    ai_commentary = _gemini_commentary({
        "nav": nav_usd,
        "nav_change_str": nav_change_str,
        "nav_change_pct": nav_change_pct,
        "regime": regime_name,
        "regime_conf": regime_conf,
        "regime_probs": {k: f"{v*100:.0f}%" for k, v in regime_probs.items()},
        "cb_state": cb_state,
        "recent_cb_events": recent_cb or ["none"],
        "exp_apr": exp_apr,
        "pos_scale": pos_scale,
        "k_pct": k_pct,
        "d_pct": d_pct,
        "perp_allocs": {k: f"{v*100:.1f}%" for k, v in perp_allocs.items()},
        "funding_rates": {s: f"{r*100:.2f}%" for s, r in funding_rates.items()},
        "ar_predictions": {s: (f"{v*100:.2f}%" if v is not None else "unavailable") for s, v in ar_preds.items()},
        "persistence_scores": {s: f"{v:.2f}" for s, v in persist_scores.items()},
        "consecutive_positive_hours": consec_positive,
        "cascade_risks": {s: f"{v:.2f}" for s, v in cascade_risks.items()},
        "rebal_total": rebal_total,
        "error_rate": error_rate,
        "uptime": uptime_str,
        "dd_halted": dd_halted,
        "dd_scale": dd_scale,
    })

    # ── Fund rows ─────────────────────────────────────────────────────────────
    fund_rows = ""
    for sym, apr in sorted(funding_rates.items(), key=lambda x: -x[1]):
        apr_pct = apr * 100
        cr = cascade_risks.get(sym, 0)
        alloc = perp_allocs.get(sym, 0) * 100
        ar_str = f"{ar_preds[sym]*100:.2f}%" if ar_preds.get(sym) is not None else "—"
        consec = consec_positive.get(sym, 0)
        apr_color = "#16a34a" if apr_pct > 5 else ("#d97706" if apr_pct >= 0 else "#dc2626")
        cr_color  = "#dc2626" if cr > 0.7 else ("#d97706" if cr > 0.5 else "#16a34a")
        fund_rows += (
            f"<tr>"
            f"<td style='font-weight:600;color:#1e293b'>{sym}</td>"
            f"<td style='color:{apr_color};font-weight:600'>{apr_pct:.2f}%</td>"
            f"<td style='color:{cr_color}'>{cr:.2f}</td>"
            f"<td style='color:#475569'>{ar_str}</td>"
            f"<td style='color:#475569'>{consec}h</td>"
            f"<td style='font-weight:600;color:#1e293b'>{alloc:.1f}%</td>"
            f"</tr>"
        )

    alloc_rows = (
        f"<tr><td style='color:#475569'>Kamino Lending</td><td style='color:#475569'>~5% APY — low-risk supply</td><td style='font-weight:600;color:#1e293b;text-align:right'>{k_pct*100:.1f}%</td></tr>"
        f"<tr><td style='color:#475569'>Drift Spot Lending</td><td style='color:#475569'>~5% APY — low-risk supply</td><td style='font-weight:600;color:#1e293b;text-align:right'>{d_pct*100:.1f}%</td></tr>"
        + "".join(
            f"<tr><td style='color:#475569'>{sym} Perp Short</td><td style='color:#475569'>Funding capture position</td><td style='font-weight:600;color:#1e293b;text-align:right'>{pct*100:.1f}%</td></tr>"
            for sym, pct in perp_allocs.items()
        )
    )

    regime_rows = "".join(
        f"<tr><td style='color:#475569'>{r}</td><td style='text-align:right;font-weight:600;color:#1e293b'>{v*100:.1f}%</td></tr>"
        for r, v in sorted(regime_probs.items(), key=lambda x: -x[1])
    )

    ai_section = (
        f"""<tr><td colspan="2" style="padding:0">
  <table width="100%" cellpadding="0" cellspacing="0" border="0">
    <tr><td style="background:#f0f9ff;border-left:3px solid #0369a1;padding:16px 18px;border-radius:0 4px 4px 0">
      <p style="margin:0 0 6px 0;font-size:11px;font-weight:700;color:#0369a1;text-transform:uppercase;letter-spacing:.05em">Strategy Analyst — AI Commentary</p>
      <p style="margin:0;font-size:13px;line-height:1.7;color:#1e3a5f">{ai_commentary}</p>
    </td></tr>
  </table>
</td></tr>"""
        if ai_commentary else ""
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<style>
  body {{margin:0;padding:0;background:#f1f5f9;font-family:Arial,Helvetica,sans-serif}}
  .wrap {{max-width:660px;margin:0 auto;padding:24px 16px}}
  .header-bar {{background:#0f172a;padding:20px 28px;border-radius:8px 8px 0 0}}
  .header-bar h1 {{margin:0;font-size:18px;font-weight:700;color:#f8fafc;letter-spacing:.01em}}
  .header-bar .meta {{margin:6px 0 0;font-size:12px;color:#94a3b8}}
  .status-tag {{display:inline-block;padding:2px 10px;border-radius:3px;font-size:11px;font-weight:700;
    background:{status_color};color:#fff;letter-spacing:.04em;text-transform:uppercase}}
  .body-wrap {{background:#ffffff;border-radius:0 0 8px 8px;padding:24px 28px}}
  .section-title {{font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.06em;
    margin:24px 0 10px;padding-bottom:6px;border-bottom:1px solid #e2e8f0}}
  .kpi-grid {{display:table;width:100%;border-collapse:separate;border-spacing:8px}}
  .kpi {{display:table-cell;background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:14px 16px;width:25%}}
  .kpi .lbl {{font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}}
  .kpi .val {{font-size:20px;font-weight:700;color:#0f172a}}
  .kpi .sub {{font-size:11px;color:#64748b;margin-top:4px}}
  table.data {{width:100%;border-collapse:collapse;font-size:13px}}
  table.data th {{text-align:left;font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;
    letter-spacing:.05em;padding:8px 10px;border-bottom:2px solid #e2e8f0}}
  table.data td {{padding:10px 10px;border-bottom:1px solid #f1f5f9;color:#334155;font-size:13px}}
  table.data tr:last-child td {{border-bottom:none}}
  .footer {{text-align:center;font-size:11px;color:#94a3b8;margin-top:20px;padding-top:16px;border-top:1px solid #e2e8f0}}
  .divider {{height:1px;background:#f1f5f9;margin:4px 0}}
</style>
</head><body><div class="wrap">

<div class="header-bar">
  <h1>QuantVault — Hourly Strategy Report</h1>
  <div class="meta">
    {ts} &nbsp;&nbsp;|&nbsp;&nbsp; Uptime: {uptime_str} &nbsp;&nbsp;|&nbsp;&nbsp;
    <span class="status-tag">{overall_status}</span>
  </div>
</div>

<div class="body-wrap">

  <div class="section-title">Portfolio Overview</div>
  <table width="100%" cellpadding="0" cellspacing="8" border="0">
    <tr>
      <td width="25%" style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:14px 16px;vertical-align:top">
        <div style="font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px">Vault NAV</div>
        <div style="font-size:20px;font-weight:700;color:#0f172a">${nav_usd:,.2f}</div>
        <div style="font-size:11px;color:{nav_dir_color};margin-top:4px">{nav_dir}${abs(nav_change_usd):,.2f} ({nav_dir}{nav_change_pct:.2f}%) vs prior</div>
      </td>
      <td width="25%" style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:14px 16px;vertical-align:top">
        <div style="font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px">Expected APR</div>
        <div style="font-size:20px;font-weight:700;color:{'#16a34a' if exp_apr >= 10 else '#d97706'}">{exp_apr:.1f}%</div>
        <div style="font-size:11px;color:#64748b;margin-top:4px">Blended yield</div>
      </td>
      <td width="25%" style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:14px 16px;vertical-align:top">
        <div style="font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px">Risk Status</div>
        <div style="font-size:15px;font-weight:700;color:{'#16a34a' if cb_state=='NORMAL' else '#d97706' if cb_state=='COOLING_DOWN' else '#dc2626'}">{cb_state}</div>
        <div style="font-size:11px;color:#64748b;margin-top:4px">{CB_DESC.get(cb_state,'')}</div>
      </td>
      <td width="25%" style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:14px 16px;vertical-align:top">
        <div style="font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px">Rebalances</div>
        <div style="font-size:20px;font-weight:700;color:#0f172a">{rebal_total}</div>
        <div style="font-size:11px;color:#64748b;margin-top:4px">Error rate: {error_rate:.1f}%</div>
      </td>
    </tr>
  </table>

  <div class="section-title">Strategy Commentary</div>
  <table width="100%" cellpadding="0" cellspacing="0" border="0">
    {ai_section if ai_commentary else f'<tr><td style="font-size:13px;color:#64748b;padding:12px 0">Commentary unavailable (AI quota exceeded — report data above is accurate).</td></tr>'}
  </table>

  <div class="section-title">Market Regime</div>
  <table width="100%" cellpadding="0" cellspacing="0" border="0">
    <tr>
      <td width="60%" style="vertical-align:top;padding-right:16px">
        <table class="data" width="100%">
          <tr><th>Signal</th><th style="text-align:right">Value</th></tr>
          <tr><td>Active Regime</td><td style="text-align:right;font-weight:600;color:#0f172a">{regime_name}</td></tr>
          <tr><td>Confidence</td><td style="text-align:right;font-weight:600;color:#0f172a">{regime_conf*100:.0f}%</td></tr>
          <tr><td>Position Scale</td><td style="text-align:right;font-weight:600;color:#0f172a">{pos_scale:.2f}x</td></tr>
          <tr><td colspan="2" style="font-size:12px;color:#64748b;padding-top:6px">{REGIME_DESC.get(regime_name,'')}</td></tr>
        </table>
      </td>
      <td width="40%" style="vertical-align:top">
        <table class="data" width="100%">
          <tr><th>Regime</th><th style="text-align:right">Probability</th></tr>
          {regime_rows}
        </table>
      </td>
    </tr>
  </table>

  <div class="section-title">Perpetual Futures — Funding Rates</div>
  <table class="data" width="100%">
    <tr>
      <th>Market</th>
      <th>Funding APR</th>
      <th>Cascade Risk</th>
      <th>AR Prediction</th>
      <th>Positive Hours</th>
      <th style="text-align:right">Allocation</th>
    </tr>
    {fund_rows}
  </table>
  <p style="font-size:11px;color:#94a3b8;margin:8px 0 0">Funding APR: annualized rate collected by shorts. AR Prediction: model estimate for next period. Cascade Risk: composite market-stress score (above 0.50 = reduce exposure).</p>

  <div class="section-title">Portfolio Allocation</div>
  <table class="data" width="100%">
    <tr><th>Position</th><th>Description</th><th style="text-align:right">Weight</th></tr>
    {alloc_rows}
  </table>

  <div class="footer">
    QuantVault Strategy Engine &nbsp;|&nbsp; {ts}<br/>
    Next scheduled report in approximately 1 hour
  </div>

</div>
</div></body></html>"""

    text = (
        f"QUANTVAULT HOURLY STRATEGY REPORT\n"
        f"{'='*50}\n"
        f"Generated: {ts}  |  Uptime: {uptime_str}  |  Status: {overall_status}\n\n"
        f"PORTFOLIO\n"
        f"  NAV:          ${nav_usd:,.2f}  ({nav_dir}${abs(nav_change_usd):,.2f} / {nav_dir}{nav_change_pct:.2f}% vs prior)\n"
        f"  Expected APR: {exp_apr:.1f}%\n"
        f"  Rebalances:   {rebal_total}  (error rate: {error_rate:.1f}%)\n\n"
        f"RISK\n"
        f"  Circuit Breaker: {cb_state} — {CB_DESC.get(cb_state,'')}\n"
        f"  Position Scale:  {pos_scale:.2f}x — {SCALE_DESC}\n"
        f"  Drawdown Halt:   {'YES' if dd_halted else 'No'}\n\n"
        f"REGIME\n"
        f"  {regime_name} ({regime_conf*100:.0f}% confidence)\n"
        f"  {REGIME_DESC.get(regime_name,'')}\n\n"
        f"FUNDING RATES\n"
        + "\n".join(f"  {sym}: {apr*100:.2f}% APR  |  cascade: {cascade_risks.get(sym,0):.2f}  |  {consec_positive.get(sym,0)}h consecutive positive" for sym, apr in sorted(funding_rates.items(), key=lambda x: -x[1]))
        + f"\n\nALLOCATION\n"
        f"  Kamino Lending:      {k_pct*100:.1f}%\n"
        f"  Drift Spot Lending:  {d_pct*100:.1f}%\n"
        + "\n".join(f"  {sym} Perp Short:    {pct*100:.1f}%" for sym, pct in perp_allocs.items())
        + (f"\n\nAI COMMENTARY\n  {ai_commentary}" if ai_commentary else "")
        + "\n"
    )

    return html, text, ts


@app.post("/send-report")
async def send_report():
    """Build and email the hourly performance report. Called by cron-job.org."""
    to_email   = os.getenv("REPORT_EMAIL_TO")
    from_email = os.getenv("REPORT_EMAIL_FROM")
    email_pass = os.getenv("REPORT_EMAIL_PASS")
    resend_key = os.getenv("RESEND_API_KEY")

    logger.info("=== /send-report called ===")
    logger.info("REPORT_EMAIL_TO    : %s", to_email or "NOT SET")
    logger.info("REPORT_EMAIL_FROM  : %s", from_email or "NOT SET")
    logger.info("REPORT_EMAIL_PASS  : %s", "SET" if email_pass else "NOT SET")
    logger.info("RESEND_API_KEY     : %s", "SET" if resend_key else "NOT SET")

    if not to_email:
        raise HTTPException(status_code=503, detail="REPORT_EMAIL_TO not set")
    if not resend_key and not (from_email and email_pass):
        raise HTTPException(status_code=503, detail="No email method configured — set RESEND_API_KEY or REPORT_EMAIL_FROM+PASS")

    logger.info("Building report HTML...")
    try:
        html, text, ts = _build_report_html()
        logger.info("Report built successfully for ts=%s", ts)
    except Exception as e:
        logger.exception("Report build failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Report build failed: {e}")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"QuantVault Report {ts}"
    msg["From"]    = from_email
    msg["To"]      = to_email
    msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(html, "html"))

    logger.info("Sending email via %s...", "Resend API" if resend_key else "SMTP")
    try:
        if resend_key:
            # Use Resend API (HTTPS — works on all cloud platforms)
            import urllib.request as _urlreq, json as _json
            payload = _json.dumps({
                "from": "QuantVault <onboarding@resend.dev>",
                "to": [to_email],
                "subject": f"QuantVault Report {ts}",
                "html": html,
                "text": text,
            }).encode()
            req = _urlreq.Request(
                "https://api.resend.com/emails",
                data=payload,
                headers={"Authorization": f"Bearer {resend_key}", "Content-Type": "application/json", "User-Agent": "QuantVault/1.0"},
                method="POST",
            )
            try:
                with _urlreq.urlopen(req, timeout=15) as resp:
                    result = _json.loads(resp.read())
                logger.info("Hourly report sent via Resend to %s (id=%s)", to_email, result.get("id"))
            except _urlreq.HTTPError as http_err:
                body = http_err.read().decode("utf-8", errors="replace")
                logger.error("Resend API error %s: %s", http_err.code, body)
                raise RuntimeError(f"Resend {http_err.code}: {body}")
        else:
            # Fallback: SMTP
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.ehlo()
                server.starttls()
                server.login(from_email, email_pass)
                server.sendmail(from_email, to_email, msg.as_string())
            logger.info("Hourly report sent via SMTP to %s", to_email)
        return {"status": "ok", "sent_to": to_email, "ts": ts}
    except Exception as e:
        logger.error("Failed to send report email: %s", e)
        raise HTTPException(status_code=500, detail=f"Email error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "strategy.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )

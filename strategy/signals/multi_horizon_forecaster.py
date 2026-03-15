"""
Multi-Horizon Funding Rate Forecaster.

Extends AR(4) from single-step to a full forecast curve:
  t+1h  — tactical:    do we enter this hour?
  t+6h  — sizing:      how large a position?
  t+24h — strategic:   which direction is the regime heading?
  t+72h — positioning: what's coming in 3 days?

The forecast curve gives us trajectory shape:
  RISING   → enter now, you'll collect more as rates rise
  PEAKING  → don't enter, rate about to collapse
  FALLING  → exit or don't enter
  TROUGH   → watch for reversal; consider pre-positioning for inverse carry

Multi-step prediction method:
  Iterate the AR(4) model forward using predicted values as inputs.
  This is the standard "plug-in" multi-step AR forecast.
  Uncertainty grows with horizon (compounding prediction error).

Pre-positioning signal:
  When trajectory=RISING and OI/basis confirm, pre-position BEFORE
  the funding rate peaks — capturing the full spike, not just the tail.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

AR_LAGS    = 4
MIN_SAMPLES = 16
HORIZONS   = [1, 6, 24, 72]   # hours ahead to forecast


class FundingTrajectory(str, Enum):
    RISING   = "RISING"    # forecast curve slopes up   → enter now
    PEAKING  = "PEAKING"   # near peak, about to fall   → avoid new entries
    FALLING  = "FALLING"   # curve slopes down          → exit or hold off
    TROUGH   = "TROUGH"    # near bottom, may reverse   → watch for inverse carry
    FLAT     = "FLAT"      # no clear direction         → neutral


@dataclass
class HorizonForecast:
    horizon_hours: int
    predicted_apr: float   # forecast APR (%)
    lower_95: float        # lower confidence bound
    upper_95: float        # upper confidence bound
    cumulative_std: float  # uncertainty grows with horizon


@dataclass
class MultiHorizonForecast:
    symbol: str
    forecasts: dict[int, HorizonForecast]   # horizon → forecast
    trajectory: FundingTrajectory
    peak_hour: int                           # estimated hours until peak (0 = already peaked)
    trough_hour: int                         # estimated hours until trough
    pre_position_signal: bool                # enter now ahead of rising funding
    exit_signal: bool                        # exit now ahead of falling funding
    ar_coefficients: list[float]
    confidence: float                        # model fit quality [0,1]

    def forecast_at(self, horizon: int) -> Optional[HorizonForecast]:
        return self.forecasts.get(horizon)

    def predicted_apr_at(self, horizon: int) -> float:
        f = self.forecast_at(horizon)
        return f.predicted_apr if f else 0.0


def _fit_ar(y: np.ndarray, n_lags: int) -> tuple[np.ndarray, float]:
    """
    OLS AR(n_lags) fit. Returns (coefficients, sigma).
    coefficients: [ar1, ar2, ..., ar_k, intercept]
    """
    n = len(y)
    rows = n - n_lags
    if rows < 2:
        return np.zeros(n_lags + 1), 1.0

    X = np.column_stack([
        y[n_lags - lag - 1: n - lag - 1] for lag in range(n_lags)
    ] + [np.ones(rows)])
    Y = y[n_lags:]

    XtX = X.T @ X + np.eye(X.shape[1]) * 1e-8
    beta = np.linalg.solve(XtX, X.T @ Y)
    residuals = Y - X @ beta
    sigma = float(np.std(residuals)) if len(residuals) > 1 else 1.0
    return beta, sigma


def _forecast_curve(
    history: np.ndarray,
    beta: np.ndarray,
    sigma: float,
    horizons: list[int],
    n_lags: int,
) -> dict[int, HorizonForecast]:
    """
    Iterative multi-step forecast.
    At each step we plug in the previous prediction as the new observation.
    Uncertainty accumulates (sigma grows with sqrt(horizon) roughly).
    """
    # Sliding window: start with last n_lags real observations
    window = list(history[-n_lags:])
    forecasts: dict[int, HorizonForecast] = {}
    cumulative_var = 0.0

    max_h = max(horizons)
    for h in range(1, max_h + 1):
        x_pred = np.array(list(reversed(window[-n_lags:])) + [1.0])
        pred = float(x_pred @ beta)
        cumulative_var += sigma ** 2  # accumulate variance step-by-step
        cum_std = float(np.sqrt(cumulative_var))
        z = 1.96

        if h in horizons:
            forecasts[h] = HorizonForecast(
                horizon_hours=h,
                predicted_apr=round(pred, 2),
                lower_95=round(pred - z * cum_std, 2),
                upper_95=round(pred + z * cum_std, 2),
                cumulative_std=round(cum_std, 3),
            )
        window.append(pred)
        if len(window) > n_lags * 2:
            window.pop(0)

    return forecasts


def _classify_trajectory(
    forecasts: dict[int, HorizonForecast],
    current_apr: float,
) -> tuple[FundingTrajectory, int, int]:
    """
    Classify the shape of the forecast curve.
    Returns (trajectory, peak_hour, trough_hour).
    """
    curve = [(0, current_apr)] + [
        (h, f.predicted_apr) for h, f in sorted(forecasts.items())
    ]
    values = [v for _, v in curve]
    hours  = [h for h, _ in curve]

    if len(values) < 3:
        return FundingTrajectory.FLAT, 0, 0

    peak_idx   = int(np.argmax(values))
    trough_idx = int(np.argmin(values))
    peak_hour  = hours[peak_idx]
    trough_hour = hours[trough_idx]

    # Slope from t=0 to t+6h and t+6h to t+24h
    v1h  = forecasts.get(1,  curve[1][1] if len(curve) > 1 else current_apr)
    v6h  = forecasts.get(6,  None)
    v24h = forecasts.get(24, None)

    f1  = v1h.predicted_apr  if isinstance(v1h,  HorizonForecast) else float(v1h)
    f6  = v6h.predicted_apr  if isinstance(v6h,  HorizonForecast) else current_apr
    f24 = v24h.predicted_apr if isinstance(v24h, HorizonForecast) else current_apr

    short_slope = f6  - current_apr   # direction over next 6h
    long_slope  = f24 - current_apr   # direction over next 24h

    threshold = 2.0  # APR % — minimum meaningful slope

    if short_slope > threshold and long_slope > threshold:
        return FundingTrajectory.RISING, peak_hour, trough_hour
    if short_slope < -threshold and long_slope < -threshold:
        # Check if near trough (already deeply negative)
        if current_apr < -5.0:
            return FundingTrajectory.TROUGH, peak_hour, trough_hour
        return FundingTrajectory.FALLING, peak_hour, trough_hour
    if short_slope > threshold and long_slope < -threshold:
        # Rising short-term but will reverse → peaking
        return FundingTrajectory.PEAKING, peak_hour, trough_hour
    if short_slope < -threshold and long_slope > threshold:
        # Falling short-term but will recover
        return FundingTrajectory.RISING, peak_hour, trough_hour

    return FundingTrajectory.FLAT, peak_hour, trough_hour


class MultiHorizonForecaster:
    """
    Per-symbol multi-horizon funding rate forecaster.

    Usage:
        forecaster = MultiHorizonForecaster()
        forecaster.update("SOL-PERP", 15.2)   # call each hour
        result = forecaster.forecast("SOL-PERP")

        if result.pre_position_signal:
            # Enter now — funding is rising, capture from the base
            ...
        if result.exit_signal:
            # Exit now — funding will fall, don't wait for it to collapse
            ...
    """

    def __init__(
        self,
        n_lags:    int = AR_LAGS,
        window:    int = 72,
        horizons:  list[int] = None,
        pre_position_threshold: float = 3.0,   # APR rise needed to pre-position
        exit_threshold:         float = -3.0,  # APR fall needed to trigger exit signal
    ):
        self.n_lags   = n_lags
        self.window   = window
        self.horizons = horizons or HORIZONS
        self.pre_position_threshold = pre_position_threshold
        self.exit_threshold = exit_threshold
        self._buffers: dict[str, deque] = {}

    def update(self, symbol: str, funding_apr: float) -> None:
        if symbol not in self._buffers:
            self._buffers[symbol] = deque(maxlen=self.window)
        self._buffers[symbol].append(funding_apr)

    def forecast(self, symbol: str) -> MultiHorizonForecast:
        buf = list(self._buffers.get(symbol, []))
        current_apr = buf[-1] if buf else 0.0

        if len(buf) < MIN_SAMPLES:
            return self._neutral(symbol, current_apr)

        y = np.array(buf)
        effective_lags = min(self.n_lags, len(y) - 2)
        if effective_lags < 1:
            return self._neutral(symbol, current_apr)

        beta, sigma = _fit_ar(y, effective_lags)
        forecasts = _forecast_curve(y, beta, sigma, self.horizons, effective_lags)

        trajectory, peak_hour, trough_hour = _classify_trajectory(forecasts, current_apr)

        # Model fit quality: R² on in-sample
        n = len(y)
        rows = n - effective_lags
        X = np.column_stack([
            y[effective_lags - lag - 1: n - lag - 1] for lag in range(effective_lags)
        ] + [np.ones(rows)])
        y_hat = X @ beta
        ss_res = np.sum((y[effective_lags:] - y_hat) ** 2)
        ss_tot = np.sum((y[effective_lags:] - y[effective_lags:].mean()) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
        confidence = max(0.0, min(1.0, r2))

        # Pre-position: funding is rising AND 6h forecast significantly above current
        f6 = forecasts.get(6)
        f24 = forecasts.get(24)
        pre_position = (
            trajectory == FundingTrajectory.RISING
            and f6 is not None
            and (f6.predicted_apr - current_apr) >= self.pre_position_threshold
        )

        # Exit signal: funding is falling and 6h forecast significantly below current
        exit_sig = (
            trajectory in (FundingTrajectory.FALLING, FundingTrajectory.PEAKING)
            and f6 is not None
            and (f6.predicted_apr - current_apr) <= self.exit_threshold
        )

        return MultiHorizonForecast(
            symbol=symbol,
            forecasts=forecasts,
            trajectory=trajectory,
            peak_hour=peak_hour,
            trough_hour=trough_hour,
            pre_position_signal=pre_position,
            exit_signal=exit_sig,
            ar_coefficients=[round(float(c), 4) for c in beta[:-1]],
            confidence=round(confidence, 3),
        )

    def forecast_all(self, symbols: list[str]) -> dict[str, MultiHorizonForecast]:
        return {sym: self.forecast(sym) for sym in symbols}

    def _neutral(self, symbol: str, current_apr: float) -> MultiHorizonForecast:
        neutral_forecasts = {
            h: HorizonForecast(
                horizon_hours=h,
                predicted_apr=current_apr,
                lower_95=current_apr - 10.0,
                upper_95=current_apr + 10.0,
                cumulative_std=10.0,
            )
            for h in self.horizons
        }
        return MultiHorizonForecast(
            symbol=symbol,
            forecasts=neutral_forecasts,
            trajectory=FundingTrajectory.FLAT,
            peak_hour=0,
            trough_hour=0,
            pre_position_signal=False,
            exit_signal=False,
            ar_coefficients=[],
            confidence=0.0,
        )

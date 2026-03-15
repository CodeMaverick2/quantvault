"""
Statistical arbitrage: Johansen cointegration + Kalman filter spread trading
across correlated Drift perpetual markets.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

logger = logging.getLogger(__name__)


@dataclass
class PairState:
    symbol_a: str
    symbol_b: str
    beta: float          # hedge ratio (how many B contracts per A)
    spread_mean: float   # rolling mean of spread
    spread_std: float    # rolling std of spread
    z_score: float       # current z-score
    cointegrated: bool   # whether pair is currently cointegrated
    p_value: float       # Engle-Granger p-value
    last_fit_ts: int     # timestamp of last refit


@dataclass
class StatArbSignal:
    pair: str            # e.g. "SOL-BTC"
    z_score: float
    action: str          # "ENTER_LONG_SPREAD", "ENTER_SHORT_SPREAD", "EXIT", "HOLD"
    beta: float
    confidence: float    # 0–1


class KalmanPairTracker:
    """
    Dynamic hedge ratio tracking using the Kalman Filter.
    Treats the hedge ratio β as a random walk latent state:
        y_t = β_t * x_t + α_t + ε_t
        β_t = β_{t-1} + η_t
    """

    def __init__(
        self,
        process_noise: float = 1e-4,
        observation_noise: float = 1e-3,
        initial_beta: float = 1.0,
        initial_cov: float = 1.0,
    ):
        # State: [beta, alpha]
        self.state = np.array([initial_beta, 0.0])
        self.P = np.eye(2) * initial_cov
        self.Q = np.eye(2) * process_noise   # process noise covariance
        self.R = observation_noise            # observation noise variance

    def update(self, x: float, y: float) -> tuple[float, float]:
        """
        Update Kalman state with new observation (x, y).
        Returns (hedge_ratio, z_score) where z_score is the standardized innovation.
        """
        # Observation matrix: H = [x, 1]
        H = np.array([x, 1.0])

        # Prediction
        x_pred = self.state
        P_pred = self.P + self.Q

        # Innovation
        y_pred = H @ x_pred
        innovation = y - y_pred
        S = H @ P_pred @ H.T + self.R

        # Kalman gain
        K = P_pred @ H.T / S

        # Update
        self.state = x_pred + K * innovation
        self.P = (np.eye(2) - np.outer(K, H)) @ P_pred

        z_score = innovation / np.sqrt(S)
        return float(self.state[0]), float(z_score)

    @property
    def hedge_ratio(self) -> float:
        return float(self.state[0])

    @property
    def intercept(self) -> float:
        return float(self.state[1])


class CointegrationEngine:
    """
    Manages cointegration pairs and generates stat arb entry/exit signals.
    """

    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
        min_history: int = 168,   # 7 days of hourly data
        refit_every: int = 720,   # refit every 30 days
        coint_pvalue_threshold: float = 0.05,
        process_noise: float = 1e-4,
        observation_noise: float = 1e-3,
    ):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.min_history = min_history
        self.refit_every = refit_every
        self.pvalue_threshold = coint_pvalue_threshold
        self.process_noise = process_noise
        self.observation_noise = observation_noise

        self._trackers: dict[str, KalmanPairTracker] = {}
        self._pair_states: dict[str, PairState] = {}
        self._tick_counts: dict[str, int] = {}

    def pair_key(self, a: str, b: str) -> str:
        return f"{a}|{b}"

    def test_cointegration(
        self, log_prices_a: np.ndarray, log_prices_b: np.ndarray
    ) -> tuple[bool, float, float]:
        """
        Engle-Granger cointegration test.
        Returns (is_cointegrated, p_value, static_beta).
        """
        try:
            from statsmodels.regression.linear_model import OLS
            from statsmodels.tools import add_constant

            _, p_value, _ = coint(log_prices_a, log_prices_b)

            x = add_constant(log_prices_b)
            model = OLS(log_prices_a, x).fit()
            static_beta = float(model.params[1])

            return p_value < self.pvalue_threshold, float(p_value), static_beta
        except Exception as exc:
            logger.warning("Cointegration test failed: %s", exc)
            return False, 1.0, 1.0

    def johansen_test(
        self, price_matrix: np.ndarray, n_series: int = 3
    ) -> tuple[bool, np.ndarray]:
        """
        Johansen test for n-asset cointegration (e.g. SOL/BTC/ETH triplet).
        Returns (is_cointegrated, cointegrating_vector).
        """
        try:
            result = coint_johansen(price_matrix, det_order=0, k_ar_diff=1)
            # Test statistic vs. 95% critical value for first cointegrating vector
            is_cointegrated = result.lr1[0] > result.cvt[0, 1]
            return is_cointegrated, result.evec[:, 0]
        except Exception as exc:
            logger.warning("Johansen test failed: %s", exc)
            return False, np.ones(n_series) / n_series

    def update(
        self,
        symbol_a: str,
        symbol_b: str,
        log_price_a: float,
        log_price_b: float,
        historical_a: np.ndarray | None = None,
        historical_b: np.ndarray | None = None,
        current_ts: int = 0,
    ) -> StatArbSignal:
        """
        Feed a new price observation and get an updated signal.
        Pass historical arrays on first call or periodically for refit.
        """
        key = self.pair_key(symbol_a, symbol_b)
        tick = self._tick_counts.get(key, 0)
        self._tick_counts[key] = tick + 1

        # Initialize or periodically refit the Kalman tracker
        if key not in self._trackers or (
            tick % self.refit_every == 0
            and historical_a is not None
            and len(historical_a) >= self.min_history
        ):
            is_coint, p_val, static_beta = (
                self.test_cointegration(historical_a, historical_b)
                if historical_a is not None
                else (True, 0.0, 1.0)
            )
            initial_beta = static_beta if is_coint else 1.0
            tracker = KalmanPairTracker(
                process_noise=self.process_noise,
                observation_noise=self.observation_noise,
                initial_beta=initial_beta,
            )
            # Warm up tracker on historical data
            if historical_a is not None:
                for xa, xb in zip(historical_a[-self.min_history :], historical_b[-self.min_history :]):
                    tracker.update(xb, xa)
            self._trackers[key] = tracker
            self._pair_states[key] = PairState(
                symbol_a=symbol_a,
                symbol_b=symbol_b,
                beta=initial_beta,
                spread_mean=0.0,
                spread_std=1.0,
                z_score=0.0,
                cointegrated=is_coint,
                p_value=p_val if historical_a is not None else 0.05,
                last_fit_ts=current_ts,
            )

        tracker = self._trackers[key]
        beta, z_score = tracker.update(log_price_b, log_price_a)

        state = self._pair_states[key]
        state.beta = beta
        state.z_score = z_score

        action = self._decide_action(z_score, state.cointegrated)

        return StatArbSignal(
            pair=f"{symbol_a}-{symbol_b}",
            z_score=z_score,
            action=action,
            beta=beta,
            confidence=max(0.0, 1.0 - state.p_value / self.pvalue_threshold),
        )

    def _decide_action(self, z_score: float, cointegrated: bool) -> str:
        if not cointegrated:
            return "HOLD"
        abs_z = abs(z_score)
        if abs_z >= self.stop_z:
            return "STOP_LOSS"
        if abs_z <= self.exit_z:
            return "EXIT"
        if z_score >= self.entry_z:
            return "ENTER_SHORT_SPREAD"   # spread too high → expect mean reversion down
        if z_score <= -self.entry_z:
            return "ENTER_LONG_SPREAD"    # spread too low → expect mean reversion up
        return "HOLD"

    def get_all_signals(self) -> dict[str, StatArbSignal | None]:
        return {
            key: StatArbSignal(
                pair=state.symbol_a + "-" + state.symbol_b,
                z_score=state.z_score,
                action=self._decide_action(state.z_score, state.cointegrated),
                beta=state.beta,
                confidence=max(0.0, 1.0 - state.p_value / self.pvalue_threshold),
            )
            for key, state in self._pair_states.items()
        }

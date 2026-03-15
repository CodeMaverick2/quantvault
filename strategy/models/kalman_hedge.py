"""
Kalman Filter for dynamic delta-neutral hedge ratio estimation.

Models the time-varying hedge ratio β_t between a perp short and
the USDC collateral + spot hedge leg.

State-space formulation:
    Observation: spread_t = β_t * price_t + α_t + ε_t
    Transition:  β_t = β_{t-1} + η_t  (random walk prior)

The Kalman gain dynamically weights new vs. prior observations,
allowing the hedge ratio to track slowly changing market structure.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HedgeState:
    beta: float           # current hedge ratio
    alpha: float          # current intercept
    cov: np.ndarray       # state covariance matrix (2x2)
    innovation: float     # last prediction error
    innovation_var: float # variance of prediction error
    z_score: float        # standardized innovation (signal quality)


class KalmanHedgeRatio:
    """
    Univariate Kalman Filter tracking a 2D state [β, α].

    Usage:
        tracker = KalmanHedgeRatio()
        for price, collateral_value in observations:
            state = tracker.update(price, collateral_value)
            hedge_ratio = state.beta
    """

    def __init__(
        self,
        process_noise: float = 1e-4,
        observation_noise: float = 1e-3,
        initial_beta: float = 1.0,
        initial_alpha: float = 0.0,
        initial_cov: float = 1.0,
    ):
        self._state = np.array([initial_beta, initial_alpha], dtype=float)
        self._P = np.eye(2, dtype=float) * initial_cov

        # Process noise covariance Q: how fast β and α drift
        self._Q = np.eye(2, dtype=float) * process_noise

        # Observation noise variance R
        self._R = float(observation_noise)

        self._last_state: Optional[HedgeState] = None
        self._update_count: int = 0

    def update(self, price: float, spread_value: float) -> HedgeState:
        """
        Update state with a new observation.

        Args:
            price: The perp/spot price (x variable in the linear model)
            spread_value: The observed spread value (y variable)

        Returns:
            HedgeState with updated beta, alpha, and diagnostic metrics
        """
        # Observation matrix H = [price, 1]
        H = np.array([price, 1.0], dtype=float)

        # --- Predict ---
        # State prediction: x_{t|t-1} = x_{t-1|t-1}  (random walk)
        x_pred = self._state.copy()
        P_pred = self._P + self._Q

        # --- Update ---
        y_pred = float(H @ x_pred)
        innovation = spread_value - y_pred
        S = float(H @ P_pred @ H.T) + self._R  # innovation variance

        # Kalman gain
        K = P_pred @ H.T / S  # shape (2,)

        # Update state
        self._state = x_pred + K * innovation
        self._P = (np.eye(2) - np.outer(K, H)) @ P_pred

        # Symmetrize P to prevent numerical drift
        self._P = (self._P + self._P.T) / 2.0

        z_score = innovation / (np.sqrt(S) + 1e-12)
        self._update_count += 1

        state = HedgeState(
            beta=float(self._state[0]),
            alpha=float(self._state[1]),
            cov=self._P.copy(),
            innovation=innovation,
            innovation_var=S,
            z_score=float(z_score),
        )
        self._last_state = state
        return state

    def update_batch(
        self, prices: np.ndarray, spreads: np.ndarray
    ) -> list[HedgeState]:
        """Batch update for historical data warm-up."""
        return [self.update(float(p), float(s)) for p, s in zip(prices, spreads)]

    @property
    def beta(self) -> float:
        return float(self._state[0])

    @property
    def alpha(self) -> float:
        return float(self._state[1])

    @property
    def state_covariance(self) -> np.ndarray:
        return self._P.copy()

    @property
    def beta_uncertainty(self) -> float:
        """1σ uncertainty on current beta estimate."""
        return float(np.sqrt(self._P[0, 0]))

    @property
    def update_count(self) -> int:
        return self._update_count

    def reset(
        self,
        initial_beta: float = 1.0,
        initial_alpha: float = 0.0,
        initial_cov: float = 1.0,
    ) -> None:
        self._state = np.array([initial_beta, initial_alpha], dtype=float)
        self._P = np.eye(2, dtype=float) * initial_cov
        self._update_count = 0
        self._last_state = None


class MultiAssetHedgeManager:
    """
    Manages separate Kalman trackers for each perp market.
    Provides a unified interface for the delta-neutral strategy.
    """

    def __init__(
        self,
        symbols: list[str],
        process_noise: float = 1e-4,
        observation_noise: float = 1e-3,
    ):
        self._trackers: dict[str, KalmanHedgeRatio] = {
            sym: KalmanHedgeRatio(
                process_noise=process_noise,
                observation_noise=observation_noise,
            )
            for sym in symbols
        }
        self._symbols = symbols

    def update(
        self,
        symbol: str,
        price: float,
        spread_value: float,
    ) -> HedgeState:
        """Update hedge ratio for a specific market."""
        if symbol not in self._trackers:
            logger.warning("Unknown symbol %s, initializing tracker", symbol)
            self._trackers[symbol] = KalmanHedgeRatio()
        return self._trackers[symbol].update(price, spread_value)

    def get_hedge_ratios(self) -> dict[str, float]:
        return {sym: t.beta for sym, t in self._trackers.items()}

    def get_state(self, symbol: str) -> Optional[HedgeState]:
        tracker = self._trackers.get(symbol)
        if tracker is None:
            return None
        return tracker._last_state

    def warmup(
        self,
        symbol: str,
        prices: np.ndarray,
        spreads: np.ndarray,
    ) -> None:
        """Warm up the Kalman tracker with historical data."""
        if symbol not in self._trackers:
            self._trackers[symbol] = KalmanHedgeRatio()
        self._trackers[symbol].update_batch(prices, spreads)
        logger.info(
            "Warmed up Kalman tracker for %s (β=%.4f) over %d obs",
            symbol,
            self._trackers[symbol].beta,
            len(prices),
        )

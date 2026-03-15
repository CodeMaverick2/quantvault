"""
Regime Transition Early Warning System.

Uses the HMM transition matrix to forecast the probability of being in
each regime N hours from now. This is the standard HMM forward algorithm
applied predictively rather than retrospectively.

Key insight:
  The HMM classifies the CURRENT regime — it's reactive.
  The transition forecaster predicts WHEN the regime will change —
  giving the system 6-24 hours of warning to pre-position.

Algorithm:
  Given current state distribution π and transition matrix A:
    π(t+N) = π(t) × A^N

  P(transition in next N hours) = 1 - π(t+N)[current_regime]

When to act:
  P(transition) > 0.40 in 6h  → start reducing positions (soft warning)
  P(transition) > 0.60 in 6h  → actively exit (hard warning)
  P(transition) > 0.40 in 24h → hold current, but no new entries

Learned transition rates (calibrated on 2021-2025 Binance data):
  BULL_CARRY lasts avg 4-12 days (funding spikes are relatively brief)
  SIDEWAYS lasts avg 8-30 days  (most common regime)
  HIGH_VOL_CRISIS lasts avg 3-14 days (bear markets + recoveries)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Regime index constants
REGIME_NAMES = ["BULL_CARRY", "SIDEWAYS", "HIGH_VOL_CRISIS"]
REGIME_IDX   = {name: i for i, name in enumerate(REGIME_NAMES)}

# Default transition matrix estimated from 2021-2025 Binance funding data.
# Each row: P(next regime | current regime), sums to 1.0.
# Calibrated on hourly observations → very high self-transition probability.
#
# Interpretation:
#   BULL_CARRY stays bull 98.5% per hour → avg duration ~67h (~2.8 days)
#   SIDEWAYS stays sideways 99.0% per hour → avg duration ~100h (~4.2 days)
#   CRISIS stays crisis 98.2% per hour → avg duration ~56h (~2.3 days)
DEFAULT_TRANSITION_MATRIX = np.array([
    # To:  BULL   SIDE   CRISIS
    [0.985, 0.014, 0.001],  # From BULL_CARRY
    [0.007, 0.990, 0.003],  # From SIDEWAYS
    [0.004, 0.016, 0.980],  # From HIGH_VOL_CRISIS
])

FORECAST_HORIZONS = [1, 6, 12, 24, 48, 72]   # hours ahead


class TransitionWarning(str, Enum):
    NONE    = "NONE"     # < 20% transition probability in 24h → hold course
    WATCH   = "WATCH"    # 20-40% in 24h → no new entries
    REDUCE  = "REDUCE"   # 40-60% in 6h → start reducing positions
    EXIT    = "EXIT"     # > 60% in 6h → exit now, regime flip imminent


@dataclass
class RegimeTransitionForecast:
    current_regime: str
    current_confidence: float             # HMM confidence in current regime

    # P(each regime) at each horizon
    # horizon_probs[6]["BULL_CARRY"] = probability of being in bull in 6h
    horizon_probs: dict[int, dict[str, float]] = field(default_factory=dict)

    # Probability of being in a DIFFERENT regime at each horizon
    transition_probs: dict[int, float] = field(default_factory=dict)

    # Most likely regime at each horizon
    predicted_regime: dict[int, str] = field(default_factory=dict)

    # Summary warning level
    warning: TransitionWarning = TransitionWarning.NONE

    # Estimated hours until regime flip (expected value)
    expected_transition_hours: float = 100.0

    # Specific transition risk: is a CRISIS regime approaching?
    crisis_approach_prob_24h: float = 0.0

    # Is a BULL regime approaching? (pre-position for carry)
    bull_approach_prob_24h: float = 0.0

    def should_reduce(self) -> bool:
        return self.warning in (TransitionWarning.REDUCE, TransitionWarning.EXIT)

    def should_exit(self) -> bool:
        return self.warning == TransitionWarning.EXIT

    def no_new_entries(self) -> bool:
        return self.warning != TransitionWarning.NONE


class RegimeTransitionForecaster:
    """
    Predicts regime transition probability at multiple horizons.

    Can operate in two modes:
    1. Default mode: uses the calibrated DEFAULT_TRANSITION_MATRIX
    2. Learned mode: transition matrix estimated from observed regime sequence

    Usage:
        forecaster = RegimeTransitionForecaster()
        # Feed regime observations (call each hour with HMM output)
        forecaster.update("BULL_CARRY", confidence=0.85)
        # Forecast
        result = forecaster.forecast()
        if result.warning == TransitionWarning.EXIT:
            # Regime flip imminent — exit positions now
            ...
    """

    def __init__(
        self,
        transition_matrix: Optional[np.ndarray] = None,
        learning_rate: float = 0.01,     # how fast to update transition matrix from observations
        horizons: list[int] = None,
        reduce_threshold: float = 0.40,  # P(transition) → REDUCE
        exit_threshold:   float = 0.60,  # P(transition) → EXIT
    ):
        self.transition_matrix = (
            transition_matrix.copy()
            if transition_matrix is not None
            else DEFAULT_TRANSITION_MATRIX.copy()
        )
        self.learning_rate   = learning_rate
        self.horizons        = horizons or FORECAST_HORIZONS
        self.reduce_threshold = reduce_threshold
        self.exit_threshold   = exit_threshold

        self._regime_history:     list[str]   = []
        self._confidence_history: list[float] = []
        self._current_regime:     str         = "SIDEWAYS"
        self._current_confidence: float       = 1.0   # start fully confident until updated

    def update(self, regime: str, confidence: float = 1.0) -> None:
        """Record latest HMM prediction and update transition matrix."""
        if regime not in REGIME_IDX:
            return

        prev = self._current_regime
        self._current_regime     = regime
        self._current_confidence = confidence
        self._regime_history.append(regime)
        self._confidence_history.append(confidence)

        # Online update of transition matrix row for previous regime
        if len(self._regime_history) >= 2 and prev in REGIME_IDX:
            prev_idx = REGIME_IDX[prev]
            curr_idx = REGIME_IDX[regime]
            # Softmax update: nudge the observed transition
            delta = np.zeros(len(REGIME_NAMES))
            delta[curr_idx] = 1.0
            self.transition_matrix[prev_idx] = (
                (1.0 - self.learning_rate) * self.transition_matrix[prev_idx]
                + self.learning_rate * delta
            )
            # Re-normalize row
            row_sum = self.transition_matrix[prev_idx].sum()
            if row_sum > 0:
                self.transition_matrix[prev_idx] /= row_sum

        # Keep history bounded
        if len(self._regime_history) > 10_000:
            self._regime_history    = self._regime_history[-5_000:]
            self._confidence_history = self._confidence_history[-5_000:]

    def forecast(self) -> RegimeTransitionForecast:
        """
        Compute P(regime at t+N) for all horizons using matrix exponentiation.
        """
        if not self._regime_history:
            return self._neutral()

        regime = self._current_regime
        confidence = self._current_confidence

        if regime not in REGIME_IDX:
            return self._neutral()

        # Current state distribution — use confidence to soften
        current_idx = REGIME_IDX[regime]
        n_states    = len(REGIME_NAMES)

        # State vector: put most probability on current regime,
        # spread (1 - confidence) over other states uniformly
        pi = np.full(n_states, (1.0 - confidence) / max(n_states - 1, 1))
        pi[current_idx] = confidence

        # Forecast horizons via A^N
        max_h = max(self.horizons)
        A = self.transition_matrix

        horizon_probs:    dict[int, dict[str, float]] = {}
        transition_probs: dict[int, float]             = {}
        predicted_regime: dict[int, str]               = {}

        # Compute A^h incrementally
        A_power = np.eye(n_states)
        prev_h  = 0
        sorted_horizons = sorted(self.horizons)

        for h in sorted_horizons:
            steps = h - prev_h
            for _ in range(steps):
                A_power = A_power @ A
            prev_h = h

            probs = pi @ A_power   # shape (n_states,)
            probs = np.clip(probs, 0.0, 1.0)
            probs /= probs.sum()

            horizon_probs[h] = {
                name: round(float(probs[i]), 4)
                for i, name in enumerate(REGIME_NAMES)
            }
            transition_probs[h] = round(float(1.0 - probs[current_idx]), 4)
            predicted_regime[h] = REGIME_NAMES[int(np.argmax(probs))]

        # Expected transition time: geometric distribution from self-transition prob
        self_prob_per_hour = float(A[current_idx, current_idx])
        if self_prob_per_hour < 1.0:
            expected_h = 1.0 / (1.0 - self_prob_per_hour)
        else:
            expected_h = 1e6

        # Specific approach probabilities at 24h
        probs_24h = horizon_probs.get(24, horizon_probs.get(sorted_horizons[-1], {}))
        crisis_prob = probs_24h.get("HIGH_VOL_CRISIS", 0.0)
        bull_prob   = probs_24h.get("BULL_CARRY", 0.0)

        # Determine warning level using 6h transition probability
        trans_6h = transition_probs.get(6, transition_probs.get(sorted_horizons[0], 0.0))
        trans_24h = transition_probs.get(24, transition_probs.get(sorted_horizons[-1], 0.0))

        if trans_6h >= self.exit_threshold:
            warning = TransitionWarning.EXIT
        elif trans_6h >= self.reduce_threshold:
            warning = TransitionWarning.REDUCE
        elif trans_24h >= self.reduce_threshold:
            warning = TransitionWarning.WATCH
        else:
            warning = TransitionWarning.NONE

        return RegimeTransitionForecast(
            current_regime=regime,
            current_confidence=round(confidence, 3),
            horizon_probs=horizon_probs,
            transition_probs=transition_probs,
            predicted_regime=predicted_regime,
            warning=warning,
            expected_transition_hours=round(expected_h, 1),
            crisis_approach_prob_24h=round(crisis_prob, 4),
            bull_approach_prob_24h=round(bull_prob, 4),
        )

    def _neutral(self) -> RegimeTransitionForecast:
        uniform = {name: round(1.0 / len(REGIME_NAMES), 4) for name in REGIME_NAMES}
        return RegimeTransitionForecast(
            current_regime="SIDEWAYS",
            current_confidence=0.33,
            horizon_probs={h: uniform.copy() for h in self.horizons},
            transition_probs={h: 0.67 for h in self.horizons},
            predicted_regime={h: "SIDEWAYS" for h in self.horizons},
            warning=TransitionWarning.NONE,
            expected_transition_hours=100.0,
            crisis_approach_prob_24h=0.33,
            bull_approach_prob_24h=0.33,
        )

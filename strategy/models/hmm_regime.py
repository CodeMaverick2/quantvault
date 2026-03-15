"""
3-state Hidden Markov Model for market regime classification.

States:
  0 — BULL_CARRY: persistently positive funding, trending market
  1 — SIDEWAYS: mixed/mean-reverting funding, choppy price action
  2 — HIGH_VOL_CRISIS: elevated volatility, negative or erratic funding

The model is trained on Drift funding rate features and retrained weekly.
"""

import logging
import pickle
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional

import numpy as np
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)


class MarketRegime(IntEnum):
    BULL_CARRY = 0
    SIDEWAYS = 1
    HIGH_VOL_CRISIS = 2

    def position_scale(self) -> float:
        """How much of max perp allocation to use in this regime."""
        if self == MarketRegime.BULL_CARRY:
            return 1.0
        if self == MarketRegime.SIDEWAYS:
            return 0.5
        return 0.0  # exit all perp positions in crisis


@dataclass
class RegimePrediction:
    regime: MarketRegime
    probabilities: dict[str, float]  # regime_name → probability
    confidence: float                # probability of predicted regime
    position_scale: float


class HMMRegimeClassifier:
    """
    Fits a GaussianHMM on funding rate + price features and classifies
    the current market regime.

    The class handles:
    - Training (fit) with walk-forward validation
    - State labeling (mapping HMM states to economic regimes)
    - Prediction with posterior probabilities
    - Model persistence (save/load)
    """

    N_FEATURES = 9  # must match get_hmm_feature_matrix() output

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 2000,
        covariance_type: str = "full",
        random_state: int = 42,
        model_path: Optional[Path] = None,
    ):
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model_path = model_path or Path("models/hmm_regime.pkl")

        self._model: Optional[GaussianHMM] = None
        self._state_map: dict[int, MarketRegime] = {}  # raw HMM state → MarketRegime
        self._is_fitted: bool = False

    def fit(self, X: np.ndarray, lengths: list[int] | None = None) -> "HMMRegimeClassifier":
        """
        Fit the HMM on feature matrix X.

        X shape: (n_samples, n_features)
        lengths: list of sequence lengths for non-contiguous data
        """
        if X.shape[0] < self.n_states * 10:
            raise ValueError(
                f"Need at least {self.n_states * 10} samples, got {X.shape[0]}"
            )

        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )
        model.fit(X, lengths)
        self._model = model
        self._is_fitted = True

        # Label states by their economic meaning
        self._label_states(X)
        logger.info(
            "HMM fitted. State map: %s",
            {k: v.name for k, v in self._state_map.items()},
        )
        return self

    def _label_states(self, X: np.ndarray) -> None:
        """
        Assign economic labels to HMM states based on the mean of key features.

        Feature index 1 = fr_z_24h (funding rate z-score relative to 24h mean)
        Feature index 8 = norm_range (proxy for realized volatility)
        """
        hidden_states = self._model.predict(X)
        state_stats: list[dict] = []

        for s in range(self.n_states):
            mask = hidden_states == s
            if mask.sum() == 0:
                state_stats.append({"mean_fr_z": 0, "mean_vol": 0, "count": 0})
                continue
            state_X = X[mask]
            mean_fr_z = float(state_X[:, 1].mean()) if X.shape[1] > 1 else 0.0
            mean_vol = float(state_X[:, 8].mean()) if X.shape[1] > 8 else 0.0
            state_stats.append({
                "mean_fr_z": mean_fr_z,
                "mean_vol": mean_vol,
                "count": int(mask.sum()),
            })

        # Sort rules:
        # BULL_CARRY = highest mean funding z-score
        # HIGH_VOL_CRISIS = highest mean volatility
        # SIDEWAYS = the remaining state
        fr_zs = [s["mean_fr_z"] for s in state_stats]
        vols = [s["mean_vol"] for s in state_stats]

        bull_state = int(np.argmax(fr_zs))
        crisis_state = int(np.argmax(vols))
        sideways_state = next(
            i for i in range(self.n_states)
            if i != bull_state and i != crisis_state
        )
        if bull_state == crisis_state:
            # Edge case: reassign
            sorted_by_fr = sorted(range(self.n_states), key=lambda i: fr_zs[i])
            bull_state = sorted_by_fr[-1]
            crisis_state = sorted_by_fr[-2]
            sideways_state = sorted_by_fr[0]

        self._state_map = {
            bull_state: MarketRegime.BULL_CARRY,
            crisis_state: MarketRegime.HIGH_VOL_CRISIS,
            sideways_state: MarketRegime.SIDEWAYS,
        }

    def predict(self, X: np.ndarray) -> RegimePrediction:
        """
        Classify the most recent observation.

        X shape: (n_samples, n_features) — uses the LAST sample for prediction.
        Uses posterior probabilities from the full sequence.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Use the full window for Viterbi decoding, predict on last timestep
        posteriors = self._model.predict_proba(X)
        last_probs = posteriors[-1]  # shape (n_states,)

        raw_state = int(np.argmax(last_probs))
        regime = self._state_map.get(raw_state, MarketRegime.SIDEWAYS)
        confidence = float(last_probs[raw_state])

        # Build probability dict with regime names
        probs_by_regime: dict[str, float] = {}
        for raw_s, reg in self._state_map.items():
            probs_by_regime[reg.name] = float(last_probs[raw_s])

        return RegimePrediction(
            regime=regime,
            probabilities=probs_by_regime,
            confidence=confidence,
            position_scale=regime.position_scale(),
        )

    def predict_sequence(self, X: np.ndarray) -> list[MarketRegime]:
        """Viterbi decoding over entire sequence (for backtesting)."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        raw_states = self._model.predict(X)
        return [self._state_map.get(s, MarketRegime.SIDEWAYS) for s in raw_states]

    def score(self, X: np.ndarray) -> float:
        """Log-likelihood of the observation sequence under the fitted model."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        return float(self._model.score(X))

    def save(self, path: Optional[Path] = None) -> None:
        path = path or self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "state_map": self._state_map}, f)
        logger.info("HMM model saved to %s", path)

    def load(self, path: Optional[Path] = None) -> "HMMRegimeClassifier":
        path = path or self.model_path
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self._state_map = data["state_map"]
        self._is_fitted = True
        logger.info("HMM model loaded from %s", path)
        return self

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

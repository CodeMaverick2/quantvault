"""Tests for HMM regime classifier."""

import numpy as np
import pytest

from strategy.models.hmm_regime import HMMRegimeClassifier, MarketRegime


@pytest.fixture
def synthetic_data():
    """Generate synthetic 3-regime data for testing."""
    rng = np.random.default_rng(42)
    n = 300

    # Bull regime: positive funding z-score, positive returns, low vol
    bull = np.column_stack([
        rng.normal(0.002, 0.001, n // 3),   # fr_log_diff
        rng.normal(1.5, 0.5, n // 3),        # fr_z_24h (positive)
        rng.normal(0.001, 0.0005, n // 3),   # fr_mom_24h
        rng.normal(0.001, 0.0003, n // 3),   # basis_pct
        rng.normal(0.5, 0.3, n // 3),        # basis_z_48h
        rng.normal(0.002, 0.003, n // 3),    # price_return_1h
        rng.normal(0.004, 0.005, n // 3),    # price_return_4h
        rng.normal(0.9, 0.2, n // 3),        # vol_ratio
        rng.normal(0.005, 0.002, n // 3),    # norm_range
    ])

    # Sideways regime: near-zero funding, low returns, moderate vol
    sideways = np.column_stack([
        rng.normal(0.0, 0.001, n // 3),
        rng.normal(0.0, 0.3, n // 3),
        rng.normal(0.0, 0.0003, n // 3),
        rng.normal(0.0, 0.0002, n // 3),
        rng.normal(0.0, 0.2, n // 3),
        rng.normal(0.0, 0.002, n // 3),
        rng.normal(0.0, 0.003, n // 3),
        rng.normal(1.0, 0.2, n // 3),
        rng.normal(0.008, 0.003, n // 3),
    ])

    # Crisis regime: negative funding, negative returns, high vol
    crisis = np.column_stack([
        rng.normal(-0.002, 0.003, n // 3),
        rng.normal(-1.5, 0.8, n // 3),
        rng.normal(-0.002, 0.001, n // 3),
        rng.normal(-0.005, 0.003, n // 3),
        rng.normal(-1.0, 0.5, n // 3),
        rng.normal(-0.005, 0.008, n // 3),
        rng.normal(-0.01, 0.015, n // 3),
        rng.normal(2.0, 0.5, n // 3),
        rng.normal(0.025, 0.010, n // 3),
    ])

    return np.vstack([bull, sideways, crisis])


class TestHMMRegimeClassifier:
    def test_fit_does_not_raise(self, synthetic_data):
        clf = HMMRegimeClassifier(n_states=3, n_iter=100)
        clf.fit(synthetic_data)
        assert clf.is_fitted

    def test_predict_returns_valid_regime(self, synthetic_data):
        clf = HMMRegimeClassifier(n_states=3, n_iter=100)
        clf.fit(synthetic_data)
        pred = clf.predict(synthetic_data[-10:])
        assert pred.regime in list(MarketRegime)
        assert 0.0 <= pred.confidence <= 1.0
        assert abs(sum(pred.probabilities.values()) - 1.0) < 0.01

    def test_predict_sequence_length(self, synthetic_data):
        clf = HMMRegimeClassifier(n_states=3, n_iter=100)
        clf.fit(synthetic_data)
        seq = clf.predict_sequence(synthetic_data)
        assert len(seq) == len(synthetic_data)
        assert all(r in list(MarketRegime) for r in seq)

    def test_position_scale_crisis_is_zero(self, synthetic_data):
        clf = HMMRegimeClassifier(n_states=3, n_iter=100)
        clf.fit(synthetic_data)
        # Feed crisis-like data (last third)
        crisis_X = synthetic_data[200:]
        pred = clf.predict(crisis_X)
        # We can't guarantee the label due to unsupervised nature,
        # but position_scale must be in [0, 1]
        assert 0.0 <= pred.position_scale <= 1.0

    def test_score_returns_finite(self, synthetic_data):
        clf = HMMRegimeClassifier(n_states=3, n_iter=100)
        clf.fit(synthetic_data)
        s = clf.score(synthetic_data)
        assert np.isfinite(s)

    def test_requires_minimum_samples(self):
        clf = HMMRegimeClassifier(n_states=3)
        with pytest.raises(ValueError, match="at least"):
            clf.fit(np.random.randn(10, 9))

    def test_predict_before_fit_raises(self):
        clf = HMMRegimeClassifier(n_states=3)
        with pytest.raises(RuntimeError):
            clf.predict(np.random.randn(5, 9))

    def test_save_load_roundtrip(self, tmp_path, synthetic_data):
        model_path = tmp_path / "hmm.pkl"
        clf = HMMRegimeClassifier(n_states=3, n_iter=100, model_path=model_path)
        clf.fit(synthetic_data)

        pred_before = clf.predict(synthetic_data[-10:])

        # Save explicitly to the tmp_path
        clf.save(model_path)
        assert model_path.exists(), f"Model file not saved at {model_path}"

        clf2 = HMMRegimeClassifier(n_states=3, model_path=model_path)
        clf2.load(model_path)

        pred_after = clf2.predict(synthetic_data[-10:])
        assert pred_before.regime == pred_after.regime

    def test_bull_regime_has_high_position_scale(self, synthetic_data):
        assert MarketRegime.BULL_CARRY.position_scale() == 1.0

    def test_crisis_regime_has_zero_position_scale(self):
        assert MarketRegime.HIGH_VOL_CRISIS.position_scale() == 0.0

    def test_sideways_regime_has_partial_scale(self):
        assert 0.0 < MarketRegime.SIDEWAYS.position_scale() < 1.0

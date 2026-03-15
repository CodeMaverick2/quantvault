"""Tests for Kalman filter hedge ratio tracker."""

import numpy as np
import pytest

from strategy.models.kalman_hedge import KalmanHedgeRatio, MultiAssetHedgeManager


class TestKalmanHedgeRatio:
    def test_initial_state(self):
        k = KalmanHedgeRatio(initial_beta=1.0, initial_alpha=0.0)
        assert k.beta == pytest.approx(1.0)
        assert k.alpha == pytest.approx(0.0)
        assert k.update_count == 0

    def test_single_update_returns_state(self):
        k = KalmanHedgeRatio()
        state = k.update(price=100.0, spread_value=200.0)
        assert hasattr(state, "beta")
        assert hasattr(state, "z_score")
        assert np.isfinite(state.beta)
        assert np.isfinite(state.z_score)
        assert k.update_count == 1

    def test_beta_converges_to_true_value(self):
        """Given y = 2*x + noise, beta should converge to ~2."""
        rng = np.random.default_rng(42)
        k = KalmanHedgeRatio(initial_beta=1.0, process_noise=1e-4, observation_noise=1e-3)
        prices = rng.uniform(90, 110, 500)
        true_beta = 2.0
        spreads = true_beta * prices + rng.normal(0, 0.1, 500)

        for p, s in zip(prices, spreads):
            k.update(p, s)

        assert abs(k.beta - true_beta) < 0.3, f"Expected beta≈2.0, got {k.beta:.3f}"

    def test_beta_tracks_drift(self):
        """Beta should adapt when the true relationship changes."""
        rng = np.random.default_rng(0)
        k = KalmanHedgeRatio(process_noise=1e-3, observation_noise=1e-3)
        prices = np.ones(400) * 100.0

        # First 200: beta = 1.0
        for _ in range(200):
            s = 1.0 * 100.0 + rng.normal(0, 0.1)
            k.update(100.0, s)
        beta_phase1 = k.beta

        # Next 200: beta = 2.0
        for _ in range(200):
            s = 2.0 * 100.0 + rng.normal(0, 0.1)
            k.update(100.0, s)
        beta_phase2 = k.beta

        assert beta_phase2 > beta_phase1, "Kalman should adapt to higher beta"

    def test_z_score_is_normalized(self):
        """Z-scores should have roughly unit standard deviation."""
        rng = np.random.default_rng(7)
        k = KalmanHedgeRatio()
        z_scores = []
        for _ in range(200):
            p = rng.uniform(90, 110)
            s = p + rng.normal(0, 0.5)
            state = k.update(p, s)
            z_scores.append(state.z_score)

        z_array = np.array(z_scores[50:])  # skip warmup
        assert np.std(z_array) < 3.0  # should not explode

    def test_batch_update(self):
        k = KalmanHedgeRatio()
        prices = np.linspace(90, 110, 100)
        spreads = prices * 1.5 + np.random.randn(100) * 0.1
        states = k.update_batch(prices, spreads)
        assert len(states) == 100
        assert k.update_count == 100

    def test_covariance_is_positive_definite(self):
        k = KalmanHedgeRatio()
        for _ in range(50):
            k.update(100.0, 150.0 + np.random.randn())
        eigenvalues = np.linalg.eigvalsh(k.state_covariance)
        assert all(ev > 0 for ev in eigenvalues), "Covariance must remain positive definite"

    def test_reset(self):
        k = KalmanHedgeRatio(initial_beta=1.0)
        for _ in range(100):
            k.update(100.0, 200.0)
        k.reset(initial_beta=1.0)
        assert k.update_count == 0
        assert k.beta == pytest.approx(1.0)


class TestMultiAssetHedgeManager:
    def test_manage_multiple_symbols(self):
        mgr = MultiAssetHedgeManager(["SOL-PERP", "BTC-PERP", "ETH-PERP"])
        mgr.update("SOL-PERP", 90.0, 0.005 * 90.0)
        mgr.update("BTC-PERP", 50000.0, 0.001 * 50000.0)
        ratios = mgr.get_hedge_ratios()
        assert "SOL-PERP" in ratios
        assert "BTC-PERP" in ratios
        assert "ETH-PERP" in ratios

    def test_unknown_symbol_auto_initializes(self):
        mgr = MultiAssetHedgeManager([])
        state = mgr.update("NEW-PERP", 100.0, 10.0)
        assert state is not None

    def test_warmup(self):
        mgr = MultiAssetHedgeManager(["SOL-PERP"])
        prices = np.linspace(80, 100, 100)
        spreads = prices * 0.01
        mgr.warmup("SOL-PERP", prices, spreads)
        assert mgr._trackers["SOL-PERP"].update_count == 100

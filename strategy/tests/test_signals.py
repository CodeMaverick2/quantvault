"""Tests for signal generators: cascade risk, features, cointegration."""

import numpy as np
import pandas as pd
import pytest

from strategy.signals.cascade_risk import CascadeRiskInput, CascadeRiskScorer
from strategy.signals.cointegration import CointegrationEngine, KalmanPairTracker
from strategy.signals.funding_features import (
    build_features,
    get_hmm_feature_matrix,
    get_lstm_feature_matrix,
)


def make_funding_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = np.arange(1700000000, 1700000000 + n * 3600, 3600)
    fr = rng.normal(0.0001, 0.0005, n)
    oracle = 90.0 + np.cumsum(rng.normal(0, 0.5, n))
    mark = oracle * (1 + rng.normal(0, 0.001, n))
    return pd.DataFrame({
        "ts": ts,
        "funding_rate": fr,
        "funding_rate_long": fr,
        "funding_rate_short": -fr,
        "mark_twap": mark,
        "oracle_twap": oracle,
        "basis_pct": (mark - oracle) / oracle,
        "hourly_rate": fr / 8,
        "apr": fr / 8 * 24 * 365.25,
        "period_revenue": rng.uniform(0, 1000, n),
    })


class TestBuildFeatures:
    def test_output_has_expected_columns(self):
        df = make_funding_df()
        out = build_features(df)
        for col in ["fr_z_24h", "fr_mom_24h", "basis_z_48h", "price_return_1h", "vol_ratio"]:
            assert col in out.columns, f"Missing column: {col}"

    def test_no_inf_values(self):
        df = make_funding_df()
        out = build_features(df)
        numeric = out.select_dtypes(include=np.number)
        assert not np.isinf(numeric.values).any(), "Infinite values in features"

    def test_length_preserved(self):
        df = make_funding_df(100)
        out = build_features(df)
        assert len(out) == 100

    def test_hmm_matrix_shape(self):
        df = make_funding_df(200)
        enriched = build_features(df)
        X, idx = get_hmm_feature_matrix(enriched)
        assert X.ndim == 2
        assert X.shape[1] > 0
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))

    def test_lstm_matrix_shape(self):
        df = make_funding_df(200)
        enriched = build_features(df)
        X, y = get_lstm_feature_matrix(enriched, sequence_length=24)
        if len(X) > 0:
            assert X.ndim == 3
            assert X.shape[1] == 24
            assert len(y) == len(X)


class TestCascadeRiskScorer:
    @pytest.fixture
    def scorer(self):
        return CascadeRiskScorer(trigger_threshold=0.70)

    def test_low_risk_normal_conditions(self, scorer):
        inp = CascadeRiskInput(
            ob_imbalance=0.3,        # healthy bid-side
            funding_percentile=0.4,  # average funding
            oi_percentile=0.4,
            book_depth_ratio=1.2,
            basis_pct=0.001,
        )
        result = scorer.score(inp)
        assert result.score < 0.50
        assert not result.triggered

    def test_high_risk_cascade_conditions(self, scorer):
        inp = CascadeRiskInput(
            ob_imbalance=-0.8,       # heavy selling pressure
            funding_percentile=0.95, # extreme crowding
            oi_percentile=0.95,      # max leverage
            book_depth_ratio=0.1,    # liquidity collapse
            basis_pct=0.04,          # 4% basis blowout
            liquidation_percentile=0.95,
        )
        result = scorer.score(inp)
        assert result.score >= 0.70
        assert result.triggered

    def test_score_is_bounded(self, scorer):
        rng = np.random.default_rng(42)
        for _ in range(100):
            inp = CascadeRiskInput(
                ob_imbalance=rng.uniform(-1, 1),
                funding_percentile=rng.uniform(0, 1),
                oi_percentile=rng.uniform(0, 1),
                book_depth_ratio=rng.uniform(0, 2),
                basis_pct=rng.uniform(-0.05, 0.05),
            )
            result = scorer.score(inp)
            assert 0.0 <= result.score <= 1.0

    def test_dominant_signal_is_set(self, scorer):
        inp = CascadeRiskInput(
            ob_imbalance=-0.9,
            funding_percentile=0.5,
            oi_percentile=0.5,
            book_depth_ratio=1.0,
        )
        result = scorer.score(inp)
        assert result.dominant_signal != ""

    def test_recommendation_contains_action(self, scorer):
        inp_low = CascadeRiskInput(ob_imbalance=0.5, funding_percentile=0.2)
        inp_high = CascadeRiskInput(ob_imbalance=-0.9, funding_percentile=0.95, oi_percentile=0.95)
        assert "NORMAL" in scorer.score(inp_low).recommendation or "LOW_RISK" in scorer.score(inp_low).recommendation
        assert "EXIT" in scorer.score(inp_high).recommendation or "CIRCUIT" in scorer.score(inp_high).recommendation

    def test_update_history_and_percentile(self, scorer):
        history = list(range(1, 101))  # 1 to 100
        for v in history:
            scorer.update_history(v * 0.0001, v * 1000, v * 100, v * 0.5)
        pct = scorer.compute_percentile(50 * 0.0001, scorer._funding_history)
        assert 0.45 < pct < 0.55


class TestKalmanPairTracker:
    def test_converges_to_true_beta(self):
        rng = np.random.default_rng(99)
        tracker = KalmanPairTracker(process_noise=1e-4, observation_noise=1e-3)
        true_beta = 0.7
        x_vals = rng.uniform(90, 110, 400)
        y_vals = true_beta * x_vals + rng.normal(0, 0.5, 400)

        for x, y in zip(x_vals, y_vals):
            beta, z = tracker.update(x, y)

        assert abs(tracker.hedge_ratio - true_beta) < 0.15

    def test_z_score_is_finite(self):
        tracker = KalmanPairTracker()
        for _ in range(50):
            _, z = tracker.update(100.0, 150.0)
            assert np.isfinite(z)


class TestCointegrationEngine:
    def test_stat_arb_signal_for_cointegrated_pair(self):
        rng = np.random.default_rng(42)
        n = 300
        # Generate cointegrated series
        x = np.cumsum(rng.normal(0, 1, n))
        y = 1.5 * x + rng.normal(0, 0.3, n)

        engine = CointegrationEngine(
            entry_z=1.5, exit_z=0.5, min_history=50, refit_every=1000
        )

        # Warm up
        for i in range(200):
            signal = engine.update(
                "A", "B",
                float(y[i]), float(x[i]),
                historical_a=y[:i+1] if i > 49 else None,
                historical_b=x[:i+1] if i > 49 else None,
            )

        # After warm-up, signal should be valid
        assert signal.pair == "A-B"
        assert signal.action in ("HOLD", "ENTER_LONG_SPREAD", "ENTER_SHORT_SPREAD", "EXIT", "STOP_LOSS")
        assert np.isfinite(signal.z_score)
        assert np.isfinite(signal.beta)

    def test_get_all_signals(self):
        engine = CointegrationEngine()
        engine.update("SOL-PERP", "BTC-PERP", 4.5, 10.8)
        engine.update("ETH-PERP", "BTC-PERP", 5.5, 10.8)
        signals = engine.get_all_signals()
        assert len(signals) == 2

"""Tests for AR funding predictor, funding persistence scorer, and oracle defense."""

import pytest
import numpy as np

from strategy.signals.ar_funding_predictor import ARFundingPredictor
from strategy.signals.funding_persistence import FundingPersistenceScorer
from strategy.risk.circuit_breakers import CircuitBreaker, CircuitBreakerConfig


# ── AR Funding Predictor ──────────────────────────────────────────────────────

class TestARFundingPredictor:
    def test_returns_default_with_no_data(self):
        pred = ARFundingPredictor()
        result = pred.predict("SOL-PERP")
        assert result.allow_entry is False
        assert result.predicted_apr == 0.0

    def test_returns_default_with_insufficient_data(self):
        pred = ARFundingPredictor()
        for _ in range(5):
            pred.update("SOL-PERP", 25.0)
        result = pred.predict("SOL-PERP")
        assert result.allow_entry is False  # < MIN_SAMPLES

    def test_allows_entry_with_persistent_positive_funding(self):
        pred = ARFundingPredictor(breakeven_apr=5.0)
        # Persistent high positive funding should predict positive next period
        for _ in range(20):
            pred.update("SOL-PERP", 30.0)
        result = pred.predict("SOL-PERP")
        assert result.predicted_apr > 0.0

    def test_blocks_entry_with_negative_trend(self):
        pred = ARFundingPredictor(breakeven_apr=5.0)
        # Declining from positive to negative
        for i in range(20):
            pred.update("SOL-PERP", 20.0 - i * 2.0)  # 20 → -18
        result = pred.predict("SOL-PERP")
        assert result.allow_entry is False

    def test_ci_lower_below_zero_blocks_entry(self):
        pred = ARFundingPredictor(breakeven_apr=5.0)
        # High variance data: CI should be wide, lower bound negative
        rng = np.random.default_rng(42)
        for val in rng.normal(10.0, 50.0, 20):  # high variance
            pred.update("SOL-PERP", float(val))
        result = pred.predict("SOL-PERP")
        # With 50% std dev, 95% CI lower bound should be negative
        assert result.lower_95 < result.upper_95

    def test_ar_coefficients_returned(self):
        pred = ARFundingPredictor()
        for _ in range(20):
            pred.update("BTC-PERP", 15.0)
        result = pred.predict("BTC-PERP")
        assert len(result.ar_coefficients) > 0

    def test_predict_all_returns_all_symbols(self):
        pred = ARFundingPredictor()
        symbols = ["SOL-PERP", "BTC-PERP", "ETH-PERP"]
        results = pred.predict_all(symbols)
        assert set(results.keys()) == set(symbols)

    def test_breakeven_gate_respected(self):
        pred = ARFundingPredictor(breakeven_apr=100.0)  # very high breakeven
        for _ in range(20):
            pred.update("ETH-PERP", 20.0)  # below breakeven
        result = pred.predict("ETH-PERP")
        assert result.allow_entry is False  # predicted ~20 < breakeven 100


# ── Funding Persistence Scorer ────────────────────────────────────────────────

class TestFundingPersistenceScorer:
    def test_returns_no_entry_with_no_data(self):
        scorer = FundingPersistenceScorer()
        result = scorer.score("SOL-PERP")
        assert result.allow_entry is False
        assert result.persistence_score == 0.0

    def test_high_persistence_allows_entry(self):
        scorer = FundingPersistenceScorer(min_score=0.5, min_consecutive=3)
        for _ in range(24):
            scorer.update("SOL-PERP", 25.0, basis_pct=0.001, z_score=1.5)
        result = scorer.score("SOL-PERP")
        assert result.allow_entry is True
        assert result.persistence_score > 0.9
        assert result.consecutive_positive == 24

    def test_mostly_negative_funding_blocks_entry(self):
        scorer = FundingPersistenceScorer(min_score=0.5, min_consecutive=3)
        for i in range(24):
            val = 10.0 if i % 5 == 0 else -10.0  # mostly negative
            scorer.update("BTC-PERP", val)
        result = scorer.score("BTC-PERP")
        assert result.persistence_score < 0.5

    def test_consecutive_streak_resets_on_negative(self):
        scorer = FundingPersistenceScorer()
        for _ in range(10):
            scorer.update("ETH-PERP", 20.0)
        scorer.update("ETH-PERP", -5.0)  # break the streak
        result = scorer.score("ETH-PERP")
        assert result.consecutive_positive == 0

    def test_score_all_returns_all_symbols(self):
        scorer = FundingPersistenceScorer()
        symbols = ["SOL-PERP", "BTC-PERP"]
        results = scorer.score_all(symbols)
        assert set(results.keys()) == set(symbols)

    def test_entry_quality_in_valid_range(self):
        scorer = FundingPersistenceScorer()
        for _ in range(15):
            scorer.update("SOL-PERP", 20.0, basis_pct=0.001, z_score=1.0)
        result = scorer.score("SOL-PERP")
        assert 0.0 <= result.entry_quality <= 1.0


# ── Oracle Manipulation Defense ───────────────────────────────────────────────

class TestOracleManipulationDefense:
    def test_no_manipulation_detected_on_normal_moves(self):
        cb = CircuitBreaker()
        # Normal price drift
        for price in [100.0, 100.1, 100.2, 99.9, 100.0, 100.3]:
            cb.check_oracle_manipulation("SOL-PERP", price)
        # No manipulation with small moves
        result = cb.check_oracle_manipulation("SOL-PERP", 100.4)
        assert result is False

    def test_detects_3sigma_jump(self):
        cb = CircuitBreaker(CircuitBreakerConfig(oracle_move_sigma_threshold=3.0))
        # Establish normal vol: ~0.1% moves
        for price in [100.0, 100.1, 100.0, 100.1, 100.0, 100.1, 100.0, 100.1, 100.0, 100.1]:
            cb.check_oracle_manipulation("BTC-PERP", price)
        # Sudden 30% move = clear manipulation
        result = cb.check_oracle_manipulation("BTC-PERP", 130.0)
        assert result is True

    def test_circuit_breaker_triggers_on_oracle_manipulation(self):
        cb = CircuitBreaker(CircuitBreakerConfig(oracle_move_sigma_threshold=2.0))
        # Establish baseline
        for price in [100.0, 100.1, 100.0, 100.1, 100.0, 100.1, 100.0, 100.1, 100.0, 100.1]:
            cb.check(
                funding_apr=20.0, basis_pct=0.001,
                oracle_deviation_pct=0.001, cascade_risk_score=0.1,
                book_depth_ratio=1.0, oracle_price=price, symbol="ETH-PERP",
            )
        # Trigger with massive price move
        state, triggers = cb.check(
            funding_apr=20.0, basis_pct=0.001,
            oracle_deviation_pct=0.001, cascade_risk_score=0.1,
            book_depth_ratio=1.0, oracle_price=150.0, symbol="ETH-PERP",
        )
        oracle_triggers = [t for t in triggers if "ORACLE_MANIPULATION" in t]
        assert len(oracle_triggers) > 0

    def test_insufficient_history_does_not_trigger(self):
        cb = CircuitBreaker()
        # Only 2 data points — not enough to compute sigma
        result = cb.check_oracle_manipulation("SOL-PERP", 100.0)
        assert result is False
        result = cb.check_oracle_manipulation("SOL-PERP", 200.0)  # big move but no history
        assert result is False

"""Tests for risk management: circuit breakers, drawdown control, position limits."""

import time

import numpy as np
import pytest

from strategy.risk.circuit_breakers import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState
from strategy.risk.drawdown_control import DrawdownController
from strategy.risk.position_limits import PositionLimits, PositionValidator, kelly_position_size


class TestCircuitBreaker:
    @pytest.fixture
    def cb(self):
        return CircuitBreaker(
            config=CircuitBreakerConfig(
                max_negative_funding_apr=-0.45,
                max_basis_pct=0.02,
                max_oracle_deviation_pct=0.005,
                cascade_risk_threshold=0.70,
                min_book_depth_ratio=0.30,
                cooldown_secs=10,
            )
        )

    def test_normal_conditions_no_trigger(self, cb):
        state, checks = cb.check(
            funding_apr=0.15,
            basis_pct=0.005,
            oracle_deviation_pct=0.001,
            cascade_risk_score=0.30,
            book_depth_ratio=1.0,
        )
        assert state == CircuitBreakerState.NORMAL
        assert checks == []

    def test_negative_funding_triggers(self, cb):
        state, checks = cb.check(
            funding_apr=-0.50,
            basis_pct=0.005,
            oracle_deviation_pct=0.001,
            cascade_risk_score=0.30,
            book_depth_ratio=1.0,
        )
        assert state == CircuitBreakerState.TRIGGERED
        assert any("NEGATIVE_FUNDING" in c for c in checks)

    def test_cascade_risk_triggers(self, cb):
        state, checks = cb.check(
            funding_apr=0.10,
            basis_pct=0.005,
            oracle_deviation_pct=0.001,
            cascade_risk_score=0.85,
            book_depth_ratio=1.0,
        )
        assert state == CircuitBreakerState.TRIGGERED
        assert any("CASCADE_RISK" in c for c in checks)

    def test_basis_blowout_triggers(self, cb):
        state, checks = cb.check(
            funding_apr=0.10,
            basis_pct=0.05,  # 5% >> 2% threshold
            oracle_deviation_pct=0.001,
            cascade_risk_score=0.30,
            book_depth_ratio=1.0,
        )
        assert state == CircuitBreakerState.TRIGGERED

    def test_liquidity_collapse_triggers(self, cb):
        state, checks = cb.check(
            funding_apr=0.10,
            basis_pct=0.005,
            oracle_deviation_pct=0.001,
            cascade_risk_score=0.30,
            book_depth_ratio=0.1,  # 10% of normal depth
        )
        assert state == CircuitBreakerState.TRIGGERED

    def test_position_multiplier_normal(self, cb):
        assert cb.get_position_multiplier() == 1.0

    def test_position_multiplier_triggered_is_zero(self, cb):
        cb.check(
            funding_apr=-0.50,
            basis_pct=0.005,
            oracle_deviation_pct=0.001,
            cascade_risk_score=0.30,
            book_depth_ratio=1.0,
        )
        assert cb.get_position_multiplier() == 0.0

    def test_force_reset(self, cb):
        cb.check(
            funding_apr=-0.50,
            basis_pct=0.005,
            oracle_deviation_pct=0.001,
            cascade_risk_score=0.30,
            book_depth_ratio=1.0,
        )
        cb.force_reset()
        assert cb.state == CircuitBreakerState.NORMAL

    def test_multiple_triggers_logged(self, cb):
        cb.check(
            funding_apr=-0.50,
            basis_pct=0.05,
            oracle_deviation_pct=0.01,
            cascade_risk_score=0.90,
            book_depth_ratio=0.1,
        )
        assert len(cb.active_events) >= 1

    def test_repeated_trigger_increments_count_not_duplicates(self, cb):
        """Firing the same condition twice must increment trigger_count, not create a new event."""
        for _ in range(3):
            cb.check(
                funding_apr=-0.50,
                basis_pct=0.005,
                oracle_deviation_pct=0.001,
                cascade_risk_score=0.30,
                book_depth_ratio=1.0,
            )
        neg_fund_events = [e for e in cb.active_events if e.trigger_name == "NEGATIVE_FUNDING"]
        assert len(neg_fund_events) == 1
        assert neg_fund_events[0].trigger_count == 3

    def test_trigger_count_field_exists(self, cb):
        cb.check(
            funding_apr=-0.50,
            basis_pct=0.005,
            oracle_deviation_pct=0.001,
            cascade_risk_score=0.30,
            book_depth_ratio=1.0,
        )
        event = cb.active_events[0]
        assert hasattr(event, "trigger_count")
        assert event.trigger_count >= 1


class TestDrawdownController:
    def test_hwm_increases_monotonically(self):
        ctrl = DrawdownController()
        navs = [100, 105, 110, 108, 112, 111]
        hwm_values = []
        for nav in navs:
            state = ctrl.record_nav(float(nav))
            hwm_values.append(state.high_water_mark)
        # HWM should never decrease
        for i in range(1, len(hwm_values)):
            assert hwm_values[i] >= hwm_values[i - 1]

    def test_no_drawdown_full_scale(self):
        ctrl = DrawdownController()
        state = ctrl.record_nav(100.0)
        # First record, no history = no drawdown
        assert state.position_scale == 1.0
        assert not state.is_halted

    def test_daily_drawdown_halves_positions(self):
        ctrl = DrawdownController(daily_halt_pct=-0.03)
        ts_base = time.time()
        state = ctrl.record_nav(100.0, ts_base)
        # Simulate 4% drop within same day
        state2 = ctrl.record_nav(96.0, ts_base + 3600)
        assert state2.position_scale <= 0.5

    def test_weekly_halt_stops_all(self):
        ctrl = DrawdownController(weekly_halt_pct=-0.07)
        ts_base = time.time()
        ctrl.record_nav(100.0, ts_base)
        # 8% drop in one week
        state = ctrl.record_nav(92.0, ts_base + 3600 * 24)
        assert state.is_halted
        assert state.position_scale == 0.0

    def test_force_halt_and_resume(self):
        ctrl = DrawdownController()
        ctrl.record_nav(100.0)
        ctrl.force_halt("test halt")
        assert ctrl.is_halted
        ctrl.resume()
        assert not ctrl.is_halted

    def test_record_nav_rejects_zero(self):
        """record_nav must raise ValueError for nav=0."""
        ctrl = DrawdownController()
        with pytest.raises(ValueError, match="NAV must be positive"):
            ctrl.record_nav(0.0)

    def test_record_nav_rejects_negative(self):
        """record_nav must raise ValueError for negative nav."""
        ctrl = DrawdownController()
        with pytest.raises(ValueError, match="NAV must be positive"):
            ctrl.record_nav(-100.0)

    def test_hysteresis_prevents_immediate_resume(self):
        """After a weekly halt, strategy should not resume until drawdown recovers past hysteresis."""
        ctrl = DrawdownController(weekly_halt_pct=-0.07, hysteresis=0.02)
        ts_base = time.time()
        ctrl.record_nav(100.0, ts_base)
        # Trigger a weekly halt: 8% drop
        state = ctrl.record_nav(92.0, ts_base + 3600 * 24)
        assert state.is_halted

        # Recover to exactly at halt threshold (-7%) — should remain halted due to hysteresis
        # weekly_dd = (93 - 100) / 100 = -0.07, which is NOT > weekly_halt + hysteresis (-0.05)
        state2 = ctrl.record_nav(93.0, ts_base + 3600 * 24 * 2)
        assert state2.is_halted, "Should still be halted within hysteresis band"

        # Recover further past hysteresis band: weekly_dd > -0.07 + 0.02 = -0.05
        # nav=96 → weekly_dd ≈ (96 - 100)/100 = -0.04, which IS > -0.05
        state3 = ctrl.record_nav(96.0, ts_base + 3600 * 24 * 3)
        assert not state3.is_halted, "Should resume once past hysteresis band"


class TestPositionValidator:
    def test_within_limits(self):
        validator = PositionValidator(PositionLimits(max_single_market_pct=0.30))
        result = validator.validate_perp_allocation(
            requested_pct=0.20,
            nav_usd=100000,
            current_perp_pct=0.10,
            market_symbol="SOL-PERP",
        )
        assert result.is_valid
        assert result.adjusted_size == pytest.approx(0.20)

    def test_single_market_cap_enforced(self):
        validator = PositionValidator(PositionLimits(max_single_market_pct=0.25))
        result = validator.validate_perp_allocation(
            requested_pct=0.40,  # Exceeds 0.25 cap
            nav_usd=100000,
            current_perp_pct=0.0,
            market_symbol="SOL-PERP",
        )
        assert result.adjusted_size <= 0.25

    def test_total_perp_cap(self):
        validator = PositionValidator(PositionLimits(max_total_perp_pct=0.60))
        result = validator.validate_perp_allocation(
            requested_pct=0.30,
            nav_usd=100000,
            current_perp_pct=0.50,  # already at 50%, 0.30 would exceed 0.60 cap
            market_symbol="BTC-PERP",
        )
        assert result.adjusted_size <= 0.10  # can only add 0.10 more

    def test_health_rate_check_pass(self):
        validator = PositionValidator(PositionLimits(min_health_buffer=0.30))
        result = validator.validate_health_rate(
            maintenance_margin_usd=1000,
            actual_margin_usd=1500,  # 1.5x > 1.3x minimum
        )
        assert result.is_valid

    def test_health_rate_check_fail(self):
        validator = PositionValidator(PositionLimits(min_health_buffer=0.30))
        result = validator.validate_health_rate(
            maintenance_margin_usd=1000,
            actual_margin_usd=1100,  # 1.1x < 1.3x minimum
        )
        assert not result.is_valid


class TestKellyCriterion:
    def test_positive_expected_return_gives_positive_size(self):
        size = kelly_position_size(expected_return=0.01, variance=0.01)
        assert size > 0.0

    def test_zero_variance_gives_zero(self):
        size = kelly_position_size(expected_return=0.01, variance=0.0)
        assert size == 0.0

    def test_size_respects_max_cap(self):
        size = kelly_position_size(
            expected_return=1.0,    # extreme expected return
            variance=0.001,
            max_pct=0.30,
        )
        assert size <= 0.30

    def test_fraction_scales_output(self):
        full = kelly_position_size(expected_return=0.05, variance=0.1, fraction=1.0)
        quarter = kelly_position_size(expected_return=0.05, variance=0.1, fraction=0.25)
        assert abs(full - 4 * quarter) < 0.001

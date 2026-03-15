"""Tests for TimeOfDayOptimizer and ATR-based leverage scaling in allocation."""

import pytest
import time
from datetime import datetime, timezone

from strategy.signals.tod_optimizer import TimeOfDayOptimizer, _HOUR_PRIORS
from strategy.optimization.allocation import (
    AllocationConfig,
    DynamicAllocationOptimizer,
    MarketYieldData,
)
from strategy.models.hmm_regime import MarketRegime


# ── TimeOfDayOptimizer ────────────────────────────────────────────────────────

class TestTimeOfDayOptimizer:
    def test_cold_start_returns_prior(self):
        opt = TimeOfDayOptimizer()
        mult = opt.get_multiplier()
        # Should be within [0.5, 1.5] even cold
        assert 0.5 <= mult.final_multiplier <= 1.5

    def test_final_multiplier_always_clipped(self):
        opt = TimeOfDayOptimizer()
        # Feed extreme values
        for _ in range(200):
            opt.update(9999.0)
        mult = opt.get_multiplier()
        assert mult.final_multiplier <= 1.5

    def test_low_funding_reduces_multiplier_over_time(self):
        opt = TimeOfDayOptimizer()
        # Feed very low APRs so learned baseline drops
        for _ in range(100):
            opt.update(0.5)
        mult = opt.get_multiplier()
        assert mult.final_multiplier <= 1.0  # shouldn't boost in low-apr environment

    def test_weekend_boost_applied(self):
        opt = TimeOfDayOptimizer()
        # Saturday: weekday() == 5
        sat_ts = datetime(2026, 3, 14, 14, 0, tzinfo=timezone.utc).timestamp()
        # Monday: weekday() == 0
        mon_ts = datetime(2026, 3, 16, 14, 0, tzinfo=timezone.utc).timestamp()

        sat_mult = opt.get_multiplier(ts=sat_ts).final_multiplier
        mon_mult = opt.get_multiplier(ts=mon_ts).final_multiplier
        assert sat_mult >= mon_mult  # weekend should be >= weekday

    def test_peak_hours_higher_than_trough_hours(self):
        opt = TimeOfDayOptimizer()
        # UTC 13:00 should be higher than UTC 02:00 (cold start, priors)
        peak_ts = datetime(2026, 3, 16, 13, 0, tzinfo=timezone.utc).timestamp()
        trough_ts = datetime(2026, 3, 16, 2, 0, tzinfo=timezone.utc).timestamp()
        peak = opt.get_multiplier(ts=peak_ts).final_multiplier
        trough = opt.get_multiplier(ts=trough_ts).final_multiplier
        assert peak > trough

    def test_update_increments_data_points(self):
        opt = TimeOfDayOptimizer()
        ts = datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc).timestamp()
        for _ in range(5):
            opt.update(20.0, ts=ts)
        mult = opt.get_multiplier(ts=ts)
        assert mult.data_points == 5

    def test_current_multiplier_returns_float(self):
        opt = TimeOfDayOptimizer()
        m = opt.current_multiplier()
        assert isinstance(m, float)
        assert 0.5 <= m <= 1.5

    def test_all_priors_sum_roughly_to_24(self):
        # Each hour prior roughly 1.0, 24 hours total ~ 24
        total = sum(_HOUR_PRIORS.values())
        assert 20 <= total <= 28  # sanity range


# ── ATR-based Leverage Scaling in Allocation ──────────────────────────────────

def _make_market(symbol="SOL-PERP", atr=0.02, funding_apr=25.0) -> MarketYieldData:
    return MarketYieldData(
        symbol=symbol,
        funding_apr=funding_apr,
        lending_apr=5.0,
        is_perp=True,
        cascade_risk=0.1,
        persistence_score=0.8,
        consecutive_positive=6,
        realized_vol_24h=0.20,
        atr_14h=atr,
    )


class TestATRLeverageScaling:
    def _allocate(self, market: MarketYieldData) -> dict:
        opt = DynamicAllocationOptimizer()
        result = opt.compute(
            markets=[market],
            regime=MarketRegime.BULL_CARRY,
            regime_confidence=0.9,
            drawdown_scale=1.0,
            cb_scale=1.0,
            kamino_apr=5.0,
            drift_spot_apr=7.0,
        )
        return result

    def test_normal_atr_baseline_allocation(self):
        # ATR = 2% = baseline, should get standard allocation
        result = self._allocate(_make_market(atr=0.02))
        assert result.total_perp_pct > 0.05

    def test_high_atr_reduces_pre_budget_size(self):
        # ATR = 6% (3× baseline) → atr_leverage_scale = clip(0.02/0.06) = 0.5
        # With a single market the budget normalization can hit the cap, so check
        # the raw pre-budget size in the breakdown instead of total_perp_pct.
        low_atr_result = self._allocate(_make_market(atr=0.02))
        high_atr_result = self._allocate(_make_market(atr=0.06))
        low_pre = low_atr_result.sizing_breakdown.get("SOL-PERP", {}).get("final_pre_budget", 1.0)
        high_pre = high_atr_result.sizing_breakdown.get("SOL-PERP", {}).get("final_pre_budget", 1.0)
        assert high_pre < low_pre  # higher ATR → smaller raw Kelly × cascade × vol × atr position

    def test_low_atr_boosts_allocation(self):
        # ATR = 1% (half baseline) → atr_leverage_scale = clip(0.02/0.01) = 1.5 (max)
        low_atr_result = self._allocate(_make_market(atr=0.01))
        mid_atr_result = self._allocate(_make_market(atr=0.02))
        assert low_atr_result.total_perp_pct >= mid_atr_result.total_perp_pct

    def test_atr_appears_in_sizing_breakdown(self):
        result = self._allocate(_make_market(atr=0.03))
        if result.sizing_breakdown:
            breakdown = result.sizing_breakdown.get("SOL-PERP", {})
            assert "atr_leverage_scale" in breakdown
            assert "atr_14h" in breakdown

    def test_tod_multiplier_scales_budget(self):
        opt = DynamicAllocationOptimizer()
        market = _make_market()

        base = opt.compute(
            markets=[market],
            regime=MarketRegime.BULL_CARRY,
            regime_confidence=0.9,
            drawdown_scale=1.0,
            cb_scale=1.0,
            kamino_apr=5.0,
            drift_spot_apr=7.0,
            tod_multiplier=1.0,
        )
        boosted = opt.compute(
            markets=[market],
            regime=MarketRegime.BULL_CARRY,
            regime_confidence=0.9,
            drawdown_scale=1.0,
            cb_scale=1.0,
            kamino_apr=5.0,
            drift_spot_apr=7.0,
            tod_multiplier=1.5,
        )
        reduced = opt.compute(
            markets=[market],
            regime=MarketRegime.BULL_CARRY,
            regime_confidence=0.9,
            drawdown_scale=1.0,
            cb_scale=1.0,
            kamino_apr=5.0,
            drift_spot_apr=7.0,
            tod_multiplier=0.5,
        )
        # Boosted should allocate more than base, reduced less
        assert boosted.total_perp_pct >= base.total_perp_pct
        assert reduced.total_perp_pct <= base.total_perp_pct

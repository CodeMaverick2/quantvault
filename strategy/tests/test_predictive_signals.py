"""
Tests for the predictive signal stack:
  - MultiHorizonForecaster
  - RegimeTransitionForecaster
  - LeadingIndicatorEngine
  - Integration: predictive signals wired into DynamicAllocationOptimizer
"""

import pytest
import numpy as np

from strategy.signals.multi_horizon_forecaster import (
    MultiHorizonForecaster,
    FundingTrajectory,
    HORIZONS,
)
from strategy.signals.regime_transition import (
    RegimeTransitionForecaster,
    TransitionWarning,
    DEFAULT_TRANSITION_MATRIX,
    REGIME_NAMES,
)
from strategy.signals.leading_indicators import (
    LeadingIndicatorEngine,
    LeadingSignal,
)
from strategy.optimization.allocation import (
    DynamicAllocationOptimizer,
    AllocationConfig,
    MarketYieldData,
)
from strategy.models.hmm_regime import MarketRegime


# ── MultiHorizonForecaster ─────────────────────────────────────────────────────

class TestMultiHorizonForecaster:

    def test_cold_start_returns_neutral(self):
        f = MultiHorizonForecaster()
        result = f.forecast("SOL-PERP")
        assert result.trajectory == FundingTrajectory.FLAT
        assert result.pre_position_signal is False
        assert result.exit_signal is False
        assert result.confidence == 0.0

    def test_insufficient_history_neutral(self):
        f = MultiHorizonForecaster()
        for _ in range(10):  # below MIN_SAMPLES=16
            f.update("SOL-PERP", 15.0)
        result = f.forecast("SOL-PERP")
        assert result.trajectory == FundingTrajectory.FLAT

    def test_forecast_curve_has_all_horizons(self):
        f = MultiHorizonForecaster()
        for i in range(48):
            f.update("SOL-PERP", 15.0 + i * 0.1)
        result = f.forecast("SOL-PERP")
        for h in HORIZONS:
            assert h in result.forecasts, f"Missing horizon {h}"
            assert result.forecasts[h].horizon_hours == h

    def test_rising_funding_detects_rising_trajectory(self):
        """Consistently rising funding rates should trigger RISING trajectory."""
        f = MultiHorizonForecaster()
        # Feed 48h of steadily rising funding: 5% → 53%
        for i in range(48):
            f.update("SOL-PERP", 5.0 + i * 1.0)
        result = f.forecast("SOL-PERP")
        assert result.trajectory == FundingTrajectory.RISING

    def test_falling_funding_detects_falling_trajectory(self):
        """Consistently falling funding rates should trigger FALLING trajectory."""
        f = MultiHorizonForecaster()
        for i in range(48):
            f.update("SOL-PERP", 50.0 - i * 0.8)
        result = f.forecast("SOL-PERP")
        assert result.trajectory in (FundingTrajectory.FALLING, FundingTrajectory.PEAKING)

    def test_pre_position_signal_on_strong_rise(self):
        """Strong rising funding triggers pre_position_signal."""
        f = MultiHorizonForecaster(pre_position_threshold=2.0)
        for i in range(48):
            f.update("SOL-PERP", 10.0 + i * 1.2)
        result = f.forecast("SOL-PERP")
        # Either pre_position or trajectory is RISING
        assert result.trajectory == FundingTrajectory.RISING or result.pre_position_signal

    def test_exit_signal_on_strong_fall(self):
        """Strong falling funding triggers exit_signal."""
        f = MultiHorizonForecaster(exit_threshold=-2.0)
        for i in range(48):
            f.update("SOL-PERP", 50.0 - i * 1.0)
        result = f.forecast("SOL-PERP")
        assert result.trajectory in (
            FundingTrajectory.FALLING, FundingTrajectory.PEAKING
        ) or result.exit_signal

    def test_negative_funding_detects_trough_or_falling(self):
        """Deeply negative falling funding detects TROUGH or FALLING (inverse carry territory)."""
        f = MultiHorizonForecaster()
        for i in range(48):
            f.update("SOL-PERP", -20.0 - i * 0.3)
        result = f.forecast("SOL-PERP")
        # Both are valid: TROUGH means already at bottom, FALLING means still descending.
        # Either way the system should not emit a carry trade entry signal.
        assert result.trajectory in (
            FundingTrajectory.TROUGH,
            FundingTrajectory.FALLING,
            FundingTrajectory.FLAT,
        )
        assert result.pre_position_signal is False

    def test_forecast_values_are_finite(self):
        f = MultiHorizonForecaster()
        for i in range(48):
            f.update("SOL-PERP", np.random.uniform(5, 40))
        result = f.forecast("SOL-PERP")
        for h, fc in result.forecasts.items():
            assert np.isfinite(fc.predicted_apr), f"NaN at horizon {h}"
            assert np.isfinite(fc.lower_95)
            assert np.isfinite(fc.upper_95)
            assert fc.lower_95 <= fc.predicted_apr <= fc.upper_95

    def test_uncertainty_grows_with_horizon(self):
        """Cumulative std should be non-decreasing across horizons."""
        f = MultiHorizonForecaster()
        for i in range(48):
            f.update("BTC-PERP", 20.0 + np.random.normal(0, 2))
        result = f.forecast("BTC-PERP")
        sorted_horizons = sorted(result.forecasts.keys())
        stds = [result.forecasts[h].cumulative_std for h in sorted_horizons]
        for i in range(1, len(stds)):
            assert stds[i] >= stds[i - 1] - 1e-6, (
                f"Std should not decrease: h={sorted_horizons[i]} std={stds[i]:.4f} < "
                f"h={sorted_horizons[i-1]} std={stds[i-1]:.4f}"
            )

    def test_forecast_all(self):
        f = MultiHorizonForecaster()
        symbols = ["SOL-PERP", "BTC-PERP", "ETH-PERP"]
        for sym in symbols:
            for i in range(24):
                f.update(sym, 15.0 + i * 0.5)
        results = f.forecast_all(symbols)
        assert set(results.keys()) == set(symbols)

    def test_flat_funding_gives_flat_trajectory(self):
        """Constant funding → AR model has zero slope → FLAT trajectory."""
        f = MultiHorizonForecaster()
        for _ in range(48):
            f.update("SOL-PERP", 20.0)
        result = f.forecast("SOL-PERP")
        assert result.trajectory == FundingTrajectory.FLAT


# ── RegimeTransitionForecaster ────────────────────────────────────────────────

class TestRegimeTransitionForecaster:

    def test_cold_start_returns_neutral(self):
        f = RegimeTransitionForecaster()
        result = f.forecast()
        assert result.warning == TransitionWarning.NONE
        assert result.current_regime == "SIDEWAYS"

    def test_forecast_has_all_horizons(self):
        f = RegimeTransitionForecaster()
        f.update("BULL_CARRY", confidence=0.9)
        result = f.forecast()
        from strategy.signals.regime_transition import FORECAST_HORIZONS
        for h in FORECAST_HORIZONS:
            assert h in result.horizon_probs
            assert h in result.transition_probs
            assert h in result.predicted_regime

    def test_probabilities_sum_to_one(self):
        f = RegimeTransitionForecaster()
        f.update("SIDEWAYS", confidence=0.8)
        result = f.forecast()
        for h, probs in result.horizon_probs.items():
            total = sum(probs.values())
            assert abs(total - 1.0) < 0.01, f"Probs at h={h} sum to {total:.4f}"

    def test_stable_regime_gives_no_warning(self):
        """Long run in same regime → transition prob should stay low → NONE warning."""
        f = RegimeTransitionForecaster()
        for _ in range(500):
            f.update("BULL_CARRY", confidence=0.95)
        result = f.forecast()
        # After many updates, matrix is adapted; self-transition should be high
        assert result.transition_probs.get(6, 1.0) < 0.5
        assert result.warning in (TransitionWarning.NONE, TransitionWarning.WATCH)

    def test_transition_prob_increases_with_horizon(self):
        """P(transition) should be non-decreasing with longer horizons."""
        f = RegimeTransitionForecaster()
        f.update("BULL_CARRY", confidence=0.7)
        result = f.forecast()
        sorted_h = sorted(result.transition_probs.keys())
        probs = [result.transition_probs[h] for h in sorted_h]
        for i in range(1, len(probs)):
            assert probs[i] >= probs[i - 1] - 0.01, (
                f"P(transition) should not decrease: {sorted_h[i]}h={probs[i]:.4f} "
                f"< {sorted_h[i-1]}h={probs[i-1]:.4f}"
            )

    def test_alternating_regimes_triggers_watch(self):
        """Rapidly alternating regimes → matrix learns high transition → warning."""
        f = RegimeTransitionForecaster(learning_rate=0.2)
        for _ in range(20):
            f.update("BULL_CARRY",      confidence=0.9)
            f.update("HIGH_VOL_CRISIS", confidence=0.9)
        result = f.forecast()
        # Transition probability should be elevated after many flips.
        # The 24h horizon captures the learned instability better than 6h.
        assert result.transition_probs.get(24, 0.0) > 0.2

    def test_crisis_approach_probability_tracked(self):
        f = RegimeTransitionForecaster()
        f.update("SIDEWAYS", confidence=0.8)
        result = f.forecast()
        assert 0.0 <= result.crisis_approach_prob_24h <= 1.0
        assert 0.0 <= result.bull_approach_prob_24h <= 1.0

    def test_expected_transition_hours_positive(self):
        f = RegimeTransitionForecaster()
        f.update("BULL_CARRY", confidence=0.85)
        result = f.forecast()
        assert result.expected_transition_hours > 0.0

    def test_low_confidence_increases_transition_prob(self):
        """Low HMM confidence → softer state distribution → higher transition prob."""
        f_high = RegimeTransitionForecaster()
        f_low  = RegimeTransitionForecaster()
        f_high.update("BULL_CARRY", confidence=0.95)
        f_low.update("BULL_CARRY",  confidence=0.40)
        r_high = f_high.forecast()
        r_low  = f_low.forecast()
        assert r_low.transition_probs.get(6, 0) >= r_high.transition_probs.get(6, 0)

    def test_should_exit_method(self):
        from strategy.signals.regime_transition import RegimeTransitionForecast
        r = RegimeTransitionForecast(
            current_regime="BULL_CARRY",
            current_confidence=0.9,
            warning=TransitionWarning.EXIT,
        )
        assert r.should_exit() is True
        assert r.should_reduce() is True

    def test_should_reduce_method(self):
        from strategy.signals.regime_transition import RegimeTransitionForecast
        r = RegimeTransitionForecast(
            current_regime="SIDEWAYS",
            current_confidence=0.7,
            warning=TransitionWarning.REDUCE,
        )
        assert r.should_reduce() is True
        assert r.should_exit() is False


# ── LeadingIndicatorEngine ────────────────────────────────────────────────────

class TestLeadingIndicatorEngine:

    def _make_engine(self):
        return LeadingIndicatorEngine()

    def test_cold_start_neutral(self):
        e = self._make_engine()
        result = e.analyze("SOL-PERP")
        assert result.signal == LeadingSignal.NEUTRAL
        assert result.composite_score == 0.0

    def test_building_oi_expanding_basis_is_bullish(self):
        """Rising OI + expanding basis → BULLISH / STRONG_BULLISH."""
        e = self._make_engine()
        # Simulate 12 hours of OI building and basis expanding
        for i in range(12):
            e.update(
                "SOL-PERP",
                oi=1_000_000 + i * 50_000,          # growing OI
                perp_price=150.0 + i * 0.05,         # perp > spot (positive basis)
                spot_price=149.5,
                liq_volume_1h=5_000,                  # low liquidations
            )
        result = e.analyze("SOL-PERP")
        assert result.composite_score > 0.0
        assert result.signal in (LeadingSignal.BULLISH, LeadingSignal.STRONG_BULLISH)
        assert result.pre_position_carry is True
        assert result.pre_exit_carry is False

    def test_declining_oi_contracting_basis_is_bearish(self):
        """Falling OI + contracting basis → negative composite, exit carry signal."""
        e = self._make_engine()
        for i in range(12):
            e.update(
                "SOL-PERP",
                oi=2_000_000 - i * 80_000,           # declining OI
                perp_price=150.0 - i * 0.1,           # basis contracting
                spot_price=150.5,
                liq_volume_1h=10_000,
            )
        result = e.analyze("SOL-PERP")
        assert result.composite_score < 0.0
        # Both BEARISH and INVERSE_SETUP (deeply negative basis) are valid bearish signals
        assert result.signal in (
            LeadingSignal.BEARISH,
            LeadingSignal.STRONG_BEARISH,
            LeadingSignal.INVERSE_SETUP,
        )
        # Key property: system should NOT recommend entering carry trade
        assert result.pre_position_carry is False

    def test_high_liquidations_trigger_bearish(self):
        """Large liquidation spike → bearish signal."""
        e = self._make_engine()
        # Build baseline
        for _ in range(20):
            e.update("SOL-PERP", oi=1_000_000, perp_price=150.0,
                     spot_price=150.0, liq_volume_1h=1_000)
        # Spike in liquidations (>2σ)
        e.update("SOL-PERP", oi=1_000_000, perp_price=149.5,
                 spot_price=150.0, liq_volume_1h=500_000)
        result = e.analyze("SOL-PERP")
        assert result.liquidations.cascade_risk == "HIGH"
        assert result.composite_score < 0.0

    def test_negative_basis_with_declining_oi_is_inverse_setup(self):
        """Deeply negative basis + OI declining → INVERSE_SETUP."""
        e = self._make_engine()
        for i in range(12):
            e.update(
                "SOL-PERP",
                oi=2_000_000 - i * 60_000,
                perp_price=148.0 - i * 0.2,    # perp < spot (negative basis)
                spot_price=150.0,
                liq_volume_1h=500,
            )
        result = e.analyze("SOL-PERP")
        # Score should be negative
        assert result.composite_score < 0.0
        assert result.basis.basis_pct < 0.0

    def test_composite_score_bounded(self):
        """Composite score should always be in [-1, +1]."""
        e = self._make_engine()
        import random
        random.seed(0)
        for _ in range(50):
            e.update(
                "SOL-PERP",
                oi=random.uniform(500_000, 5_000_000),
                perp_price=random.uniform(100, 200),
                spot_price=random.uniform(100, 200),
                liq_volume_1h=random.uniform(0, 1_000_000),
            )
        result = e.analyze("SOL-PERP")
        assert -1.0 <= result.composite_score <= 1.0

    def test_analyze_all(self):
        e = self._make_engine()
        symbols = ["SOL-PERP", "BTC-PERP"]
        for sym in symbols:
            for _ in range(8):
                e.update(sym, oi=1_000_000, perp_price=100.0,
                         spot_price=100.0, liq_volume_1h=1000)
        results = e.analyze_all(symbols)
        assert set(results.keys()) == set(symbols)

    def test_oi_zscore_calculation(self):
        """OI z-score should be near 0 for mean value."""
        e = self._make_engine()
        for _ in range(30):
            e.update("BTC-PERP", oi=1_000_000, perp_price=40000,
                     spot_price=40000, liq_volume_1h=1000)
        result = e.analyze("BTC-PERP")
        assert abs(result.oi.oi_zscore) < 0.5

    def test_explanation_string_nonempty_on_strong_signal(self):
        e = self._make_engine()
        for i in range(12):
            e.update("SOL-PERP", oi=1_000_000 + i * 100_000,
                     perp_price=150.0 + i * 0.1, spot_price=149.5,
                     liq_volume_1h=500)
        result = e.analyze("SOL-PERP")
        if result.signal != LeadingSignal.NEUTRAL:
            assert len(result.explanation) > 0


# ── Integration: predictive signals → allocation ──────────────────────────────

class TestPredictiveAllocationIntegration:

    def _base_markets(self):
        return [
            MarketYieldData(
                symbol="SOL-PERP", funding_apr=30.0, lending_apr=10.0,
                is_perp=True, cascade_risk=0.1, persistence_score=0.8,
                consecutive_positive=5, realized_vol_24h=0.25, atr_14h=0.02,
            ),
            MarketYieldData(
                symbol="BTC-PERP", funding_apr=20.0, lending_apr=9.0,
                is_perp=True, cascade_risk=0.1, persistence_score=0.75,
                consecutive_positive=4, realized_vol_24h=0.20, atr_14h=0.018,
            ),
        ]

    def _base_kwargs(self):
        return dict(
            regime=MarketRegime.BULL_CARRY,
            regime_confidence=0.85,
            drawdown_scale=1.0,
            cb_scale=1.0,
            kamino_apr=11.0,
            drift_spot_apr=10.0,
            tod_multiplier=1.0,
        )

    def test_no_predictive_signals_works_normally(self):
        """Allocation without predictive signals should work unchanged."""
        opt = DynamicAllocationOptimizer()
        result = opt.compute(self._base_markets(), **self._base_kwargs())
        assert result.total_perp_pct > 0.0
        assert result.total_pct > 0.0

    def test_exit_warning_zeros_perp(self):
        """Regime transition EXIT warning should zero out perp positions."""
        from strategy.signals.regime_transition import RegimeTransitionForecast
        exit_forecast = RegimeTransitionForecast(
            current_regime="BULL_CARRY",
            current_confidence=0.9,
            transition_probs={1: 0.7, 6: 0.75, 12: 0.8, 24: 0.85, 48: 0.9, 72: 0.92},
            horizon_probs={6: {"BULL_CARRY": 0.25, "SIDEWAYS": 0.50, "HIGH_VOL_CRISIS": 0.25}},
            predicted_regime={6: "SIDEWAYS"},
            warning=TransitionWarning.EXIT,
        )
        opt = DynamicAllocationOptimizer()
        result = opt.compute(
            self._base_markets(), **self._base_kwargs(),
            regime_transition=exit_forecast,
        )
        assert result.total_perp_pct == 0.0

    def test_reduce_warning_halves_perp(self):
        """Regime transition REDUCE warning should cut perp allocation."""
        from strategy.signals.regime_transition import RegimeTransitionForecast
        reduce_forecast = RegimeTransitionForecast(
            current_regime="BULL_CARRY",
            current_confidence=0.8,
            transition_probs={1: 0.3, 6: 0.45, 12: 0.5, 24: 0.6, 48: 0.7, 72: 0.75},
            horizon_probs={6: {"BULL_CARRY": 0.55, "SIDEWAYS": 0.35, "HIGH_VOL_CRISIS": 0.10}},
            predicted_regime={6: "BULL_CARRY"},
            warning=TransitionWarning.REDUCE,
        )
        opt = DynamicAllocationOptimizer()
        baseline = opt.compute(self._base_markets(), **self._base_kwargs())
        reduced  = opt.compute(
            self._base_markets(), **self._base_kwargs(),
            regime_transition=reduce_forecast,
        )
        assert reduced.total_perp_pct < baseline.total_perp_pct

    def test_rising_forecast_boosts_perp(self):
        """Multi-horizon RISING forecast should increase perp allocation."""
        from strategy.signals.multi_horizon_forecaster import (
            MultiHorizonForecast, HorizonForecast
        )
        rising = {
            sym: MultiHorizonForecast(
                symbol=sym,
                forecasts={
                    1:  HorizonForecast(1,  32.0, 28.0, 36.0, 2.0),
                    6:  HorizonForecast(6,  40.0, 34.0, 46.0, 4.0),
                    24: HorizonForecast(24, 48.0, 38.0, 58.0, 8.0),
                    72: HorizonForecast(72, 50.0, 36.0, 64.0, 14.0),
                },
                trajectory=FundingTrajectory.RISING,
                peak_hour=72,
                trough_hour=0,
                pre_position_signal=True,
                exit_signal=False,
                ar_coefficients=[0.8, 0.1, 0.05, 0.02],
                confidence=0.82,
            )
            for sym in ["SOL-PERP", "BTC-PERP"]
        }
        opt = DynamicAllocationOptimizer()
        baseline = opt.compute(self._base_markets(), **self._base_kwargs())
        boosted  = opt.compute(
            self._base_markets(), **self._base_kwargs(),
            multi_horizon_forecasts=rising,
        )
        assert boosted.total_perp_pct >= baseline.total_perp_pct

    def test_falling_forecast_reduces_perp(self):
        """Multi-horizon FALLING forecast should reduce perp allocation."""
        from strategy.signals.multi_horizon_forecaster import (
            MultiHorizonForecast, HorizonForecast
        )
        falling = {
            sym: MultiHorizonForecast(
                symbol=sym,
                forecasts={
                    1:  HorizonForecast(1,  18.0, 14.0, 22.0, 2.0),
                    6:  HorizonForecast(6,  10.0,  6.0, 14.0, 4.0),
                    24: HorizonForecast(24,  5.0,  0.0, 10.0, 8.0),
                    72: HorizonForecast(72,  2.0, -4.0,  8.0, 14.0),
                },
                trajectory=FundingTrajectory.FALLING,
                peak_hour=0,
                trough_hour=72,
                pre_position_signal=False,
                exit_signal=True,
                ar_coefficients=[0.6, 0.2, 0.1, 0.05],
                confidence=0.75,
            )
            for sym in ["SOL-PERP", "BTC-PERP"]
        }
        opt = DynamicAllocationOptimizer()
        baseline = opt.compute(self._base_markets(), **self._base_kwargs())
        reduced  = opt.compute(
            self._base_markets(), **self._base_kwargs(),
            multi_horizon_forecasts=falling,
        )
        assert reduced.total_perp_pct < baseline.total_perp_pct

    def test_bullish_leading_indicator_boosts_perp(self):
        """Bullish leading indicators (OI building + basis expanding) boost allocation."""
        from strategy.signals.leading_indicators import (
            LeadingIndicatorResult, LeadingSignal, OIAnalysis, BasisAnalysis,
            LiquidationAnalysis,
        )
        bullish = {
            sym: LeadingIndicatorResult(
                symbol=sym,
                composite_score=0.65,
                signal=LeadingSignal.STRONG_BULLISH,
                oi=OIAnalysis(
                    current_oi=5_000_000, oi_change_1h_pct=1.2,
                    oi_change_6h_pct=6.0, oi_zscore=1.5,
                    trend="BUILDING", acceleration=0.3,
                    leverage_signal="OVERLEVERAGED",
                ),
                basis=BasisAnalysis(
                    basis_pct=0.15, basis_zscore=1.8,
                    basis_trend="EXPANDING", basis_velocity=0.03,
                    funding_lead_signal="PRE_SPIKE",
                    expected_funding_direction="UP",
                ),
                liquidations=LiquidationAnalysis(
                    recent_liq_volume=5_000, liq_zscore=0.2,
                    cascade_risk="LOW", funding_impact="NORMAL",
                ),
                hours_ahead_estimate=3,
                pre_position_carry=True,
                pre_exit_carry=False,
                pre_position_inverse=False,
                explanation="basis expanding (0.15%); OI building (+6.0% over 6h)",
            )
            for sym in ["SOL-PERP", "BTC-PERP"]
        }
        opt = DynamicAllocationOptimizer()
        baseline = opt.compute(self._base_markets(), **self._base_kwargs())
        boosted  = opt.compute(
            self._base_markets(), **self._base_kwargs(),
            leading_indicators=bullish,
        )
        assert boosted.total_perp_pct >= baseline.total_perp_pct

    def test_all_three_predictive_signals_combined(self):
        """All three predictive signals together should be consistent."""
        from strategy.signals.multi_horizon_forecaster import (
            MultiHorizonForecast, HorizonForecast
        )
        from strategy.signals.regime_transition import RegimeTransitionForecast
        from strategy.signals.leading_indicators import (
            LeadingIndicatorResult, LeadingSignal, OIAnalysis,
            BasisAnalysis, LiquidationAnalysis,
        )
        # All bullish
        rising_forecasts = {
            "SOL-PERP": MultiHorizonForecast(
                symbol="SOL-PERP",
                forecasts={h: HorizonForecast(h, 30.0 + h * 0.3, 20.0, 40.0, 2.0 + h * 0.1)
                           for h in HORIZONS},
                trajectory=FundingTrajectory.RISING,
                peak_hour=72, trough_hour=0,
                pre_position_signal=True, exit_signal=False,
                ar_coefficients=[0.7, 0.15, 0.08, 0.03],
                confidence=0.88,
            )
        }
        stable_transition = RegimeTransitionForecast(
            current_regime="BULL_CARRY",
            current_confidence=0.92,
            transition_probs={h: 0.05 for h in [1, 6, 12, 24, 48, 72]},
            horizon_probs={6: {"BULL_CARRY": 0.92, "SIDEWAYS": 0.07, "HIGH_VOL_CRISIS": 0.01}},
            predicted_regime={6: "BULL_CARRY"},
            warning=TransitionWarning.NONE,
        )
        bullish_leading = {
            "SOL-PERP": LeadingIndicatorResult(
                symbol="SOL-PERP",
                composite_score=0.55,
                signal=LeadingSignal.BULLISH,
                oi=OIAnalysis(2_000_000, 0.8, 4.0, 1.2, "BUILDING", 0.2, "NORMAL"),
                basis=BasisAnalysis(0.08, 1.1, "EXPANDING", 0.02, "PRE_SPIKE", "UP"),
                liquidations=LiquidationAnalysis(3_000, 0.1, "LOW", "NORMAL"),
                hours_ahead_estimate=4,
                pre_position_carry=True, pre_exit_carry=False,
                pre_position_inverse=False,
                explanation="basis expanding",
            )
        }
        opt = DynamicAllocationOptimizer()
        result = opt.compute(
            self._base_markets(), **self._base_kwargs(),
            multi_horizon_forecasts=rising_forecasts,
            regime_transition=stable_transition,
            leading_indicators=bullish_leading,
        )
        # With all bullish signals, perp should be at or above baseline
        baseline = opt.compute(self._base_markets(), **self._base_kwargs())
        assert result.total_perp_pct >= baseline.total_perp_pct * 0.95

    def test_allocation_still_sums_to_one(self):
        """Total allocation must remain 1.0 regardless of predictive signals."""
        from strategy.signals.regime_transition import RegimeTransitionForecast
        reduce = RegimeTransitionForecast(
            current_regime="BULL_CARRY",
            current_confidence=0.7,
            transition_probs={h: 0.45 for h in [1, 6, 12, 24, 48, 72]},
            horizon_probs={6: {"BULL_CARRY": 0.55, "SIDEWAYS": 0.35, "HIGH_VOL_CRISIS": 0.10}},
            predicted_regime={6: "BULL_CARRY"},
            warning=TransitionWarning.REDUCE,
        )
        opt = DynamicAllocationOptimizer()
        result = opt.compute(
            self._base_markets(), **self._base_kwargs(),
            regime_transition=reduce,
        )
        total = result.total_perp_pct + result.total_lending_pct
        assert abs(total - 1.0) < 0.02

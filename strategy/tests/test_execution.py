"""
Tests for production execution layer:
  - FeeModel (realistic Drift fees + slippage)
  - FundingTimingOptimizer (hold vs. exit per hour)
  - MarginHealthMonitor (liquidation risk management)
  - SmartRebalanceEngine (cost-aware position management)
"""

import pytest
import time
from datetime import datetime, timezone

from strategy.execution.fee_model import (
    ExecutionCostModel, DriftFeeConfig, SlippageModel, OrderType,
)
from strategy.execution.funding_timing import FundingTimingOptimizer
from strategy.execution.margin_health import (
    MarginHealthMonitor, HealthStatus,
)
from strategy.execution.rebalance_engine import (
    SmartRebalanceEngine, RebalanceUrgency,
)


# ── FeeModel ──────────────────────────────────────────────────────────────────

class TestFeeModel:

    def test_taker_round_trip_is_0_2_pct(self):
        cfg = DriftFeeConfig(taker_fee_bps=10.0, maker_rebate_bps=2.0)
        assert abs(cfg.round_trip_cost_pct(OrderType.MARKET) - 0.002) < 1e-9

    def test_maker_round_trip_is_negative(self):
        """Maker-only fills should earn a rebate (negative cost)."""
        cfg = DriftFeeConfig(taker_fee_bps=10.0, maker_rebate_bps=2.0)
        assert cfg.round_trip_cost_pct(OrderType.POST) < 0.0

    def test_ioc_is_taker(self):
        cfg = DriftFeeConfig(taker_fee_bps=10.0, maker_rebate_bps=2.0)
        assert cfg.round_trip_cost_pct(OrderType.IOC) == cfg.round_trip_cost_pct(OrderType.MARKET)

    def test_tier_discount_reduces_fee(self):
        base = DriftFeeConfig(taker_fee_bps=10.0, tier_discount_pct=0.0)
        disc = DriftFeeConfig(taker_fee_bps=10.0, tier_discount_pct=0.20)
        assert disc.taker_fee_pct < base.taker_fee_pct

    def test_slippage_increases_with_order_size(self):
        slippage = SlippageModel()
        s_small  = slippage.estimate_slippage_bps(5_000)
        s_medium = slippage.estimate_slippage_bps(100_000)
        s_large  = slippage.estimate_slippage_bps(1_000_000)
        assert s_small < s_medium < s_large

    def test_slippage_at_zero_is_minimal(self):
        s = SlippageModel()
        assert s.estimate_slippage_bps(0) < 2.0

    def test_cost_estimate_sums_correctly(self):
        model = ExecutionCostModel(nav=100_000)
        est = model.estimate_cost(0.10, OrderType.MARKET)
        assert abs(est.total_cost_pct - (est.fees_pct + est.slippage_pct)) < 1e-9
        assert abs(est.total_cost_usd - est.total_cost_pct * 10_000) < 0.01

    def test_should_not_rebalance_tiny_drift(self):
        model = ExecutionCostModel(nav=100_000)
        do_it, cost, gain = model.should_rebalance(
            current_pct=0.20, target_pct=0.20,  # no change
            current_apr=25.0, target_apr=25.0,
            hours_held=10.0,
        )
        assert do_it is False

    def test_should_rebalance_large_apr_improvement(self):
        """Large APR improvement + meaningful position change should justify rebalancing."""
        model = ExecutionCostModel(nav=100_000)
        # Increasing from 15% to 25% allocation while APR improves from 10% to 40%
        do_it, cost, gain = model.should_rebalance(
            current_pct=0.15, target_pct=0.25,
            current_apr=10.0, target_apr=40.0,
            hours_held=48.0,
        )
        assert do_it is True
        assert gain > 0

    def test_breakeven_hours_positive(self):
        model = ExecutionCostModel(nav=100_000)
        be_hours = model.breakeven_hours(
            position_change_pct=0.10,
            apr_improvement=20.0,
        )
        assert be_hours > 0
        assert be_hours < 8760   # should break even within a year

    def test_breakeven_hours_infinite_for_zero_improvement(self):
        model = ExecutionCostModel(nav=100_000)
        be = model.breakeven_hours(0.10, 0.0)
        assert be == float("inf")

    def test_twap_splits_large_order(self):
        model = ExecutionCostModel(nav=1_000_000)
        schedule = model.twap_schedule(
            total_size_usd=500_000,
            total_hours=4.0,
            max_single_order_usd=100_000,
        )
        assert len(schedule) == 5  # 500k / 100k = 5 slices
        total = sum(s["size_usd"] for s in schedule)
        assert abs(total - 500_000) < 1.0

    def test_twap_no_split_small_order(self):
        model = ExecutionCostModel(nav=100_000)
        schedule = model.twap_schedule(
            total_size_usd=50_000,
            total_hours=4.0,
            max_single_order_usd=100_000,
        )
        assert len(schedule) == 1


# ── FundingTimingOptimizer ────────────────────────────────────────────────────

class TestFundingTimingOptimizer:

    def test_cold_start_uses_live_funding(self):
        """With no history, decision should use live funding rate."""
        opt = FundingTimingOptimizer(nav=100_000, hold_threshold_apr=5.0)
        now = datetime(2025, 3, 15, 14, 0, 0, tzinfo=timezone.utc)
        decision = opt.decide(now, current_funding_apr=30.0)
        assert decision.should_hold is True
        assert decision.predicted_apr == 30.0

    def test_low_funding_recommends_exit(self):
        """Predicted APR below threshold → exit."""
        opt = FundingTimingOptimizer(
            nav=100_000, hold_threshold_apr=5.0, lending_apr=10.0
        )
        # Build history: very low funding for this hour
        for day in range(30):
            ts = datetime(2025, 2, day % 28 + 1, 3, 0, 0, tzinfo=timezone.utc)
            opt.record_funding("SOL-PERP", ts, apr=1.0)
        now = datetime(2025, 3, 15, 3, 0, 0, tzinfo=timezone.utc)
        decision = opt.decide(now, symbols=["SOL-PERP"], current_funding_apr=1.5)
        assert decision.should_exit is True

    def test_high_funding_recommends_hold(self):
        """Live funding well above threshold → hold regardless."""
        opt = FundingTimingOptimizer(nav=100_000, hold_threshold_apr=5.0)
        now = datetime(2025, 3, 15, 14, 0, 0, tzinfo=timezone.utc)
        decision = opt.decide(now, current_funding_apr=50.0)
        # Live override: 50% >> 5% × 2 = 10%
        assert decision.should_hold is True

    def test_record_and_retrieve_history(self):
        opt = FundingTimingOptimizer()
        now = datetime(2025, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
        for _ in range(20):
            opt.record_funding("SOL-PERP", now, apr=25.0)
        stats = opt.get_hour_stats("SOL-PERP", 12, now.weekday())
        assert stats.n_observations == 20
        assert abs(stats.mean_apr - 25.0) < 0.1

    def test_positive_rate_tracks_correctly(self):
        opt = FundingTimingOptimizer()
        now = datetime(2025, 3, 15, 8, 0, 0, tzinfo=timezone.utc)
        for i in range(20):
            apr = 15.0 if i % 2 == 0 else -5.0   # 50% positive
            opt.record_funding("SOL-PERP", now, apr=apr)
        stats = opt.get_hour_stats("SOL-PERP", 8, now.weekday())
        assert abs(stats.positive_rate - 0.5) < 0.1

    def test_funding_income_greater_than_lending_when_high_apr(self):
        """When funding is high, funding_income > lending_income."""
        opt = FundingTimingOptimizer(
            nav=100_000, perp_allocation=0.45, lending_apr=10.0
        )
        now = datetime(2025, 3, 15, 14, 0, 0, tzinfo=timezone.utc)
        decision = opt.decide(now, current_funding_apr=40.0)
        assert decision.expected_funding_income > decision.expected_lending_income

    def test_decision_has_valid_fields(self):
        opt = FundingTimingOptimizer(nav=100_000)
        now = datetime(2025, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
        d = opt.decide(now, current_funding_apr=15.0)
        assert 0 <= d.hour_of_day <= 23
        assert 0 <= d.day_of_week <= 6
        assert isinstance(d.should_hold, bool)
        assert isinstance(d.should_exit, bool)
        assert d.should_hold != d.should_exit
        assert len(d.reason) > 0

    def test_worst_hours_returns_n_entries(self):
        opt = FundingTimingOptimizer()
        for h in range(24):
            for d in range(7):
                ts = datetime(2025, 3, 1 + d, h, 0, 0, tzinfo=timezone.utc)
                opt.record_funding("SOL-PERP", ts, apr=float(h))
        worst = opt.worst_hours("SOL-PERP", n=5)
        assert len(worst) == 5


# ── MarginHealthMonitor ───────────────────────────────────────────────────────

class TestMarginHealthMonitor:

    def test_healthy_position(self):
        monitor = MarginHealthMonitor(nav=100_000)
        health = monitor.update_position(
            symbol="SOL-PERP",
            notional_usd=50_000,
            collateral_usd=100_000,   # very well collateralized
            unrealized_pnl=0.0,
            mark_price=150.0,
            entry_price=150.0,
        )
        assert health.status == HealthStatus.HEALTHY
        assert health.health_ratio > 2.0

    def test_overleveraged_position_triggers_danger(self):
        """High leverage with adverse PnL should trigger DANGER.
        health = (collateral + upnl) / (notional × 5%)
        = (100k + (-88k)) / (200k × 0.05) = 12k / 10k = 1.2 → DANGER
        """
        monitor = MarginHealthMonitor(nav=100_000)
        health = monitor.update_position(
            symbol="SOL-PERP",
            notional_usd=200_000,
            collateral_usd=100_000,
            unrealized_pnl=-88_000,  # net collateral = 12k, req margin = 10k → health 1.2
            mark_price=194.0,
            entry_price=150.0,
        )
        assert health.status in (HealthStatus.DANGER, HealthStatus.CRITICAL)
        assert health.health_ratio < 1.3

    def test_critical_position_triggers_emergency(self):
        """Position near liquidation threshold → CRITICAL.
        health = (100k + (-95.5k)) / (100k × 0.05) = 4.5k / 5k = 0.9 → CRITICAL
        """
        monitor = MarginHealthMonitor(nav=100_000)
        health = monitor.update_position(
            symbol="SOL-PERP",
            notional_usd=100_000,
            collateral_usd=100_000,
            unrealized_pnl=-95_500,  # nearly wiped: health = 4.5k/5k = 0.9
            mark_price=245.5,
            entry_price=150.0,
        )
        assert health.status == HealthStatus.CRITICAL
        assert health.recommended_action.startswith("EMERGENCY")

    def test_portfolio_health_aggregates(self):
        monitor = MarginHealthMonitor(nav=500_000)
        for sym, notional, upnl in [
            ("SOL-PERP", 50_000, 1_000),
            ("BTC-PERP", 40_000, -500),
            ("ETH-PERP", 30_000, 2_000),
        ]:
            monitor.update_position(
                symbol=sym, notional_usd=notional,
                collateral_usd=500_000 / 3, unrealized_pnl=upnl,
                mark_price=100.0, entry_price=100.0,
            )
        ph = monitor.portfolio_health()
        assert ph.total_notional == pytest.approx(120_000, rel=0.01)
        assert ph.status in HealthStatus.__members__.values()

    def test_deleverage_scale_healthy(self):
        monitor = MarginHealthMonitor(nav=100_000)
        monitor.update_position(
            "SOL-PERP", 20_000, 100_000, 0.0, 150.0, 150.0
        )
        assert monitor.deleverage_scale() == 1.0

    def test_deleverage_scale_danger(self):
        # health = (100k - 88k) / (200k × 0.05) = 12k/10k = 1.2 → DANGER → scale 0.5
        monitor = MarginHealthMonitor(nav=100_000)
        monitor.update_position(
            "SOL-PERP", 200_000, 100_000, -88_000, 194.0, 150.0
        )
        assert monitor.deleverage_scale() <= 0.5

    def test_max_adverse_move_positive(self):
        monitor = MarginHealthMonitor(nav=100_000)
        h = monitor.update_position(
            "SOL-PERP", 50_000, 100_000, 0.0, 150.0, 150.0
        )
        assert h.max_adverse_move_pct > 0

    def test_compute_safe_notional(self):
        """Safe notional should give health ≥ target_health."""
        monitor = MarginHealthMonitor(nav=500_000)
        safe_notional = monitor.compute_safe_notional(
            collateral_usd=100_000, target_health=2.5
        )
        # Check: collateral / (safe_notional × 5%) ≈ 2.5
        implied_health = 100_000 / (safe_notional * 0.05)
        assert abs(implied_health - 2.5) < 0.01

    def test_health_trend_stable(self):
        """No history → STABLE trend."""
        monitor = MarginHealthMonitor(nav=100_000)
        assert monitor.health_trend() == "STABLE"

    def test_no_positions_gives_healthy(self):
        monitor = MarginHealthMonitor(nav=100_000)
        ph = monitor.portfolio_health()
        assert ph.status == HealthStatus.HEALTHY
        assert ph.total_notional == 0.0


# ── SmartRebalanceEngine ──────────────────────────────────────────────────────

class TestSmartRebalanceEngine:

    def test_no_change_returns_all_skips(self):
        """Identical current and target → skip everything."""
        engine = SmartRebalanceEngine(nav=100_000)
        plan = engine.plan(
            current_allocations={"SOL-PERP": 0.20},
            target_allocations={"SOL-PERP": 0.20},
            current_aprs={"SOL-PERP": 25.0},
            target_aprs={"SOL-PERP": 25.0},
        )
        executable = [i for i in plan.instructions if i.action != "SKIP"]
        assert len(executable) == 0

    def test_tiny_drift_is_skipped(self):
        """Drift below min_drift_pct → SKIP."""
        engine = SmartRebalanceEngine(nav=100_000, min_drift_pct=0.02)
        plan = engine.plan(
            current_allocations={"SOL-PERP": 0.200},
            target_allocations={"SOL-PERP": 0.210},  # 1% drift < 2% min
            current_aprs={"SOL-PERP": 25.0},
            target_aprs={"SOL-PERP": 25.0},
        )
        skipped = [i for i in plan.instructions if i.action == "SKIP"]
        assert len(skipped) > 0

    def test_large_drift_always_executes(self):
        """Drift above force_drift_pct → always execute."""
        engine = SmartRebalanceEngine(nav=100_000, force_drift_pct=0.05)
        plan = engine.plan(
            current_allocations={"SOL-PERP": 0.10},
            target_allocations={"SOL-PERP": 0.20},  # 10% drift >> 5% threshold
            current_aprs={"SOL-PERP": 25.0},
            target_aprs={"SOL-PERP": 25.0},
            urgency=RebalanceUrgency.NORMAL,
        )
        executable = [i for i in plan.instructions if i.action != "SKIP"]
        assert len(executable) > 0

    def test_emergency_overrides_cost_check(self):
        """EMERGENCY urgency → execute regardless of cost."""
        engine = SmartRebalanceEngine(nav=100_000)
        plan = engine.plan(
            current_allocations={"SOL-PERP": 0.20},
            target_allocations={"SOL-PERP": 0.21},   # 1% drift, would normally skip
            current_aprs={"SOL-PERP": 25.0},
            target_aprs={"SOL-PERP": 25.0},
            urgency=RebalanceUrgency.EMERGENCY,
        )
        executable = [i for i in plan.instructions if i.action != "SKIP"]
        assert len(executable) > 0

    def test_high_apr_improvement_justifies_rebalance(self):
        """Large APR gain should make rebalance worthwhile."""
        engine = SmartRebalanceEngine(nav=500_000, hours_per_cycle=24.0)
        plan = engine.plan(
            current_allocations={"SOL-PERP": 0.20},
            target_allocations={"SOL-PERP": 0.30},   # 10% drift
            current_aprs={"SOL-PERP": 5.0},
            target_aprs={"SOL-PERP": 45.0},          # 40% APR improvement
        )
        assert plan.worthwhile is True

    def test_emergency_uses_market_orders(self):
        engine = SmartRebalanceEngine(nav=100_000)
        plan = engine.plan(
            current_allocations={"SOL-PERP": 0.20},
            target_allocations={"SOL-PERP": 0.00},
            current_aprs={"SOL-PERP": 0.0},
            target_aprs={"SOL-PERP": 0.0},
            urgency=RebalanceUrgency.EMERGENCY,
        )
        executable = [i for i in plan.instructions if i.action != "SKIP"]
        assert all(i.order_type == OrderType.MARKET for i in executable)

    def test_passive_uses_post_only_for_small_orders(self):
        engine = SmartRebalanceEngine(nav=100_000)
        plan = engine.plan(
            current_allocations={"SOL-PERP": 0.20},
            target_allocations={"SOL-PERP": 0.25},  # 5% drift, $5k order
            current_aprs={"SOL-PERP": 25.0},
            target_aprs={"SOL-PERP": 35.0},
            urgency=RebalanceUrgency.PASSIVE,
        )
        executable = [i for i in plan.instructions if i.action != "SKIP"]
        if executable:
            assert all(i.order_type in (OrderType.POST, OrderType.IOC) for i in executable)

    def test_fee_efficiency_stats_populated(self):
        engine = SmartRebalanceEngine(nav=100_000)
        for _ in range(5):
            engine.plan(
                current_allocations={"SOL-PERP": 0.20},
                target_allocations={"SOL-PERP": 0.22},
                current_aprs={"SOL-PERP": 25.0},
                target_aprs={"SOL-PERP": 25.0},
            )
        stats = engine.fee_efficiency_stats()
        assert "total_checks" in stats
        assert stats["total_checks"] == 5

    def test_netting_savings_non_negative(self):
        engine = SmartRebalanceEngine(nav=500_000)
        plan = engine.plan(
            current_allocations={"SOL-PERP": 0.30, "BTC-PERP": 0.10},
            target_allocations={"SOL-PERP": 0.10, "BTC-PERP": 0.30},
            current_aprs={"SOL-PERP": 30.0, "BTC-PERP": 20.0},
            target_aprs={"SOL-PERP": 30.0, "BTC-PERP": 20.0},
        )
        assert plan.netting_savings_usd >= 0

    def test_new_position_produces_executable_instruction(self):
        """Opening a new position from 0% → 15% should produce an executable instruction."""
        engine = SmartRebalanceEngine(nav=100_000, force_drift_pct=0.04)
        plan = engine.plan(
            current_allocations={"ETH-PERP": 0.00},
            target_allocations={"ETH-PERP": 0.15},
            current_aprs={"ETH-PERP": 0.0},
            target_aprs={"ETH-PERP": 20.0},
            urgency=RebalanceUrgency.URGENT,
        )
        executable = [i for i in plan.instructions if i.action != "SKIP"]
        assert len(executable) > 0
        assert executable[0].size_usd == pytest.approx(15_000, rel=0.01)

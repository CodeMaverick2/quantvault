"""Tests for dynamic allocation optimizer."""

import pytest

from strategy.models.hmm_regime import MarketRegime
from strategy.optimization.allocation import (
    AllocationConfig,
    AllocationResult,
    DynamicAllocationOptimizer,
    MarketYieldData,
)


def make_markets(
    sol_apr: float = 20.0,
    btc_apr: float = 15.0,
    eth_apr: float = 12.0,
    cascade_risk: float = 0.2,
    persistence_score: float = 0.8,
    consecutive_positive: int = 6,
) -> list[MarketYieldData]:
    return [
        MarketYieldData("SOL-PERP", sol_apr, 6.0, True, cascade_risk,
                        persistence_score=persistence_score,
                        consecutive_positive=consecutive_positive),
        MarketYieldData("BTC-PERP", btc_apr, 6.0, True, cascade_risk,
                        persistence_score=persistence_score,
                        consecutive_positive=consecutive_positive),
        MarketYieldData("ETH-PERP", eth_apr, 6.0, True, cascade_risk,
                        persistence_score=persistence_score,
                        consecutive_positive=consecutive_positive),
    ]


@pytest.fixture
def optimizer():
    return DynamicAllocationOptimizer(
        AllocationConfig(
            min_lending_pct=0.10,
            max_perp_pct=0.60,
            max_single_perp_pct=0.25,
            target_funding_apr_threshold=10.0,
        )
    )


class TestDynamicAllocationOptimizer:
    def test_total_allocation_is_one(self, optimizer):
        result = optimizer.compute(
            markets=make_markets(),
            regime=MarketRegime.BULL_CARRY,
            regime_confidence=0.9,
            drawdown_scale=1.0,
            cb_scale=1.0,
            kamino_apr=5.0,
            drift_spot_apr=7.0,
        )
        total = (
            result.kamino_lending_pct
            + result.drift_spot_lending_pct
            + sum(result.perp_allocations.values())
        )
        assert abs(total - 1.0) < 0.02, f"Total={total:.4f}"

    def test_crisis_regime_zero_perp(self, optimizer):
        result = optimizer.compute(
            markets=make_markets(),
            regime=MarketRegime.HIGH_VOL_CRISIS,
            regime_confidence=0.9,
            drawdown_scale=1.0,
            cb_scale=1.0,
            kamino_apr=5.0,
            drift_spot_apr=7.0,
        )
        assert result.total_perp_pct == pytest.approx(0.0, abs=0.01)

    def test_cb_scale_zero_exits_perps(self, optimizer):
        result = optimizer.compute(
            markets=make_markets(),
            regime=MarketRegime.BULL_CARRY,
            regime_confidence=0.9,
            drawdown_scale=1.0,
            cb_scale=0.0,  # circuit breaker fired
            kamino_apr=5.0,
            drift_spot_apr=7.0,
        )
        assert result.total_perp_pct == pytest.approx(0.0, abs=0.01)

    def test_min_lending_respected(self, optimizer):
        result = optimizer.compute(
            markets=make_markets(),
            regime=MarketRegime.BULL_CARRY,
            regime_confidence=0.9,
            drawdown_scale=1.0,
            cb_scale=1.0,
            kamino_apr=5.0,
            drift_spot_apr=7.0,
        )
        assert result.total_lending_pct >= 0.10 - 0.01

    def test_single_perp_cap(self, optimizer):
        result = optimizer.compute(
            markets=make_markets(sol_apr=100.0),  # extreme funding
            regime=MarketRegime.BULL_CARRY,
            regime_confidence=1.0,
            drawdown_scale=1.0,
            cb_scale=1.0,
            kamino_apr=5.0,
            drift_spot_apr=7.0,
        )
        for sym, alloc in result.perp_allocations.items():
            assert alloc <= 0.25 + 0.01, f"{sym} allocation {alloc:.3f} exceeds 0.25 cap"

    def test_below_threshold_funding_excluded(self, optimizer):
        low_markets = make_markets(sol_apr=5.0, btc_apr=3.0, eth_apr=2.0)  # all below 10%
        result = optimizer.compute(
            markets=low_markets,
            regime=MarketRegime.BULL_CARRY,
            regime_confidence=0.9,
            drawdown_scale=1.0,
            cb_scale=1.0,
            kamino_apr=5.0,
            drift_spot_apr=7.0,
        )
        assert result.total_perp_pct == pytest.approx(0.0, abs=0.01)

    def test_high_cascade_risk_excludes_market(self, optimizer):
        risky_markets = make_markets(cascade_risk=0.9)  # all markets too risky
        result = optimizer.compute(
            markets=risky_markets,
            regime=MarketRegime.BULL_CARRY,
            regime_confidence=0.9,
            drawdown_scale=1.0,
            cb_scale=1.0,
            kamino_apr=5.0,
            drift_spot_apr=7.0,
        )
        assert result.total_perp_pct == pytest.approx(0.0, abs=0.01)

    def test_expected_apr_is_positive(self, optimizer):
        result = optimizer.compute(
            markets=make_markets(),
            regime=MarketRegime.BULL_CARRY,
            regime_confidence=0.9,
            drawdown_scale=1.0,
            cb_scale=1.0,
            kamino_apr=5.0,
            drift_spot_apr=7.0,
        )
        assert result.expected_blended_apr > 0.0

    def test_blended_apr_accuracy(self, optimizer):
        """Expected blended APR must exactly match the weighted sum of component APRs."""
        kamino_apr = 5.0
        drift_apr = 7.0
        result = optimizer.compute(
            markets=make_markets(sol_apr=20.0, btc_apr=15.0, eth_apr=12.0),
            regime=MarketRegime.BULL_CARRY,
            regime_confidence=0.9,
            drawdown_scale=1.0,
            cb_scale=1.0,
            kamino_apr=kamino_apr,
            drift_spot_apr=drift_apr,
        )
        # Manually recompute expected blended APR
        perp_apr_map = {"SOL-PERP": 20.0, "BTC-PERP": 15.0, "ETH-PERP": 12.0}
        expected = (
            result.kamino_lending_pct * kamino_apr
            + result.drift_spot_lending_pct * drift_apr
            + sum(alloc * perp_apr_map.get(sym, 0.0) for sym, alloc in result.perp_allocations.items())
        )
        assert result.expected_blended_apr == pytest.approx(expected, rel=1e-6)

    def test_min_lending_pct_enforced_even_with_large_perp_budget(self, optimizer):
        """When perp allocations would exceed 1 - min_lending_pct, lending must still be >= min."""
        result = optimizer.compute(
            markets=make_markets(sol_apr=100.0, btc_apr=100.0, eth_apr=100.0),
            regime=MarketRegime.BULL_CARRY,
            regime_confidence=1.0,
            drawdown_scale=1.0,
            cb_scale=1.0,
            kamino_apr=5.0,
            drift_spot_apr=7.0,
        )
        assert result.total_lending_pct >= 0.10 - 0.005
        total = result.total_perp_pct + result.total_lending_pct
        assert total <= 1.0 + 0.005

    def test_bull_regime_has_more_perp_than_sideways(self, optimizer):
        bull = optimizer.compute(
            markets=make_markets(),
            regime=MarketRegime.BULL_CARRY,
            regime_confidence=0.9,
            drawdown_scale=1.0,
            cb_scale=1.0,
            kamino_apr=5.0,
            drift_spot_apr=7.0,
        )
        sideways = optimizer.compute(
            markets=make_markets(),
            regime=MarketRegime.SIDEWAYS,
            regime_confidence=0.9,
            drawdown_scale=1.0,
            cb_scale=1.0,
            kamino_apr=5.0,
            drift_spot_apr=7.0,
        )
        assert bull.total_perp_pct > sideways.total_perp_pct

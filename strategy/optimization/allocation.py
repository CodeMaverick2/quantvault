"""
Dynamic yield optimizer: computes target allocation percentages
across lending protocols and delta-neutral perp positions.

Uses regime signal, funding rates, and risk metrics to compute
the optimal portfolio split every 10 minutes.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..models.hmm_regime import MarketRegime
from ..risk.position_limits import kelly_position_size

logger = logging.getLogger(__name__)


@dataclass
class MarketYieldData:
    """Real-time yield data for a single market/protocol."""
    symbol: str
    funding_apr: float             # current annualized funding rate
    lending_apr: float             # lending protocol APY
    is_perp: bool
    cascade_risk: float = 0.0     # 0–1 risk score for this market


@dataclass
class AllocationResult:
    """Target allocation across all strategies."""
    # Lending allocations (% of vault NAV)
    kamino_lending_pct: float = 0.10
    drift_spot_lending_pct: float = 0.10

    # Delta-neutral perp allocations (% per market)
    perp_allocations: dict[str, float] = field(default_factory=dict)

    # Stat arb allocations
    stat_arb_allocations: dict[str, float] = field(default_factory=dict)

    # Metadata
    total_perp_pct: float = 0.0
    total_lending_pct: float = 0.0
    total_pct: float = 0.0
    regime: str = "UNKNOWN"
    position_scale: float = 1.0
    expected_blended_apr: float = 0.0

    def validate(self) -> bool:
        total = (
            self.kamino_lending_pct
            + self.drift_spot_lending_pct
            + sum(self.perp_allocations.values())
            + sum(self.stat_arb_allocations.values())
        )
        return abs(total - 1.0) < 0.01


@dataclass
class AllocationConfig:
    min_lending_pct: float = 0.10
    max_perp_pct: float = 0.60
    max_single_perp_pct: float = 0.25
    max_stat_arb_pct: float = 0.15
    target_funding_apr_threshold: float = 10.0  # % APR minimum to enter perp


class DynamicAllocationOptimizer:
    """
    Computes target portfolio allocations using:

    1. Regime signal (HMM) → scales overall perp exposure
    2. Funding rate ranking → allocates more to highest-yielding markets
    3. Kelly criterion → sizes individual market allocations
    4. Cascade risk → zero allocation to high-risk markets
    5. Drawdown scale → overall portfolio scale from risk manager

    The optimizer is deterministic and idempotent given the same inputs,
    making it safe to call repeatedly without state accumulation.
    """

    def __init__(self, config: AllocationConfig | None = None):
        self.config = config or AllocationConfig()

    def compute(
        self,
        markets: list[MarketYieldData],
        regime: MarketRegime,
        regime_confidence: float,
        drawdown_scale: float,
        cb_scale: float,
        kamino_apr: float,
        drift_spot_apr: float,
    ) -> AllocationResult:
        """
        Main allocation computation.

        Args:
            markets: List of perp markets with their current yield data
            regime: Current HMM-classified regime
            regime_confidence: Posterior probability of the regime [0, 1]
            drawdown_scale: Scale factor from DrawdownController [0, 1]
            cb_scale: Scale factor from CircuitBreaker [0, 1]
            kamino_apr: Current Kamino USDC lending APY
            drift_spot_apr: Current Drift spot USDC lending APY

        Returns:
            AllocationResult with normalized percentage allocations
        """
        # Combined external scale
        external_scale = drawdown_scale * cb_scale
        regime_scale = regime.position_scale() * min(1.0, regime_confidence + 0.3)
        effective_scale = external_scale * regime_scale

        result = AllocationResult(regime=regime.name, position_scale=effective_scale)

        # --- Phase 1: Determine perp allocations ---
        perp_budget = self.config.max_perp_pct * effective_scale

        eligible_markets = [
            m for m in markets
            if m.is_perp
            and m.funding_apr >= self.config.target_funding_apr_threshold
            and m.cascade_risk < 0.50   # exclude high-risk markets
        ]

        perp_allocs: dict[str, float] = {}
        if eligible_markets and perp_budget > 0.01:
            perp_allocs = self._allocate_perps(eligible_markets, perp_budget)

        result.perp_allocations = perp_allocs
        result.total_perp_pct = sum(perp_allocs.values())

        # --- Phase 2: Lending fills the remainder ---
        remaining = 1.0 - result.total_perp_pct

        # Allocate remaining to lending, preferring higher APY
        total_lending = max(remaining, self.config.min_lending_pct)
        if drift_spot_apr >= kamino_apr:
            drift_share = 0.60
            kamino_share = 0.40
        else:
            drift_share = 0.40
            kamino_share = 0.60

        result.drift_spot_lending_pct = total_lending * drift_share
        result.kamino_lending_pct = total_lending * kamino_share
        result.total_lending_pct = total_lending

        # --- Phase 3: Normalize to 100% ---
        total = result.total_perp_pct + result.total_lending_pct
        if abs(total - 1.0) > 0.005:
            # Adjust lending to fill remaining
            adjustment = 1.0 - result.total_perp_pct
            result.drift_spot_lending_pct = adjustment * drift_share
            result.kamino_lending_pct = adjustment * kamino_share
            result.total_lending_pct = adjustment

        result.total_pct = (
            result.total_perp_pct + result.total_lending_pct
        )

        # --- Expected blended APR ---
        blended = (
            result.kamino_lending_pct * kamino_apr
            + result.drift_spot_lending_pct * drift_spot_apr
            + sum(
                alloc * next(
                    (m.funding_apr for m in eligible_markets if m.symbol == sym), 0.0
                )
                for sym, alloc in perp_allocs.items()
            )
        )
        result.expected_blended_apr = blended

        logger.info(
            "Allocation computed: regime=%s scale=%.2f perp=%.1f%% lending=%.1f%% E[APR]=%.1f%%",
            regime.name,
            effective_scale,
            result.total_perp_pct * 100,
            result.total_lending_pct * 100,
            result.expected_blended_apr,
        )

        return result

    def _allocate_perps(
        self,
        markets: list[MarketYieldData],
        budget: float,
    ) -> dict[str, float]:
        """
        Allocate perp budget across eligible markets using Kelly criterion.
        """
        if not markets:
            return {}

        # Compute Kelly sizes for each market
        kelly_sizes: dict[str, float] = {}
        for m in markets:
            # Use funding APR as expected return, assume ~20% annualized vol for sizing
            period_return = m.funding_apr / (24 * 365.25)  # per-hour return
            period_variance = (0.20 / np.sqrt(24 * 365.25)) ** 2
            kelly = kelly_position_size(
                expected_return=period_return,
                variance=period_variance,
                fraction=0.25,
                max_pct=self.config.max_single_perp_pct,
            )
            kelly_sizes[m.symbol] = kelly

        total_kelly = sum(kelly_sizes.values())
        if total_kelly <= 0:
            return {}

        # Scale Kelly sizes to fit within budget
        allocs: dict[str, float] = {}
        for sym, size in kelly_sizes.items():
            normalized = (size / total_kelly) * budget
            allocs[sym] = min(normalized, self.config.max_single_perp_pct)

        # Re-normalize after capping
        total_alloc = sum(allocs.values())
        if total_alloc > budget:
            scale = budget / total_alloc
            allocs = {sym: v * scale for sym, v in allocs.items()}

        return {sym: v for sym, v in allocs.items() if v > 0.005}

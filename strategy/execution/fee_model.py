"""
Realistic fee model for Drift Protocol v2 perpetuals.

This is one of the most impactful production improvements — the naive backtest
uses 0.001% per rebalance, but real execution costs are ~7x higher.

Drift Protocol v2 Fee Schedule — VOLUME-TIERED (verified from docs, 2025):

  | Tier | 30-day Volume   | Taker Fee  | Maker Rebate |
  |------|----------------|------------|--------------|
  |  1   | ≤ $2M          | 0.0350%    | -0.0025%     |
  |  2   | $2M – $10M     | 0.0300%    | -0.0025%     |
  |  3   | $10M – $50M    | 0.0275%    | -0.0030%     |
  |  4   | $50M – $200M   | 0.0250%    | -0.0030%     |
  | VIP  | > $200M        | 0.0200%    | -0.0035%     |

  DRIFT token staking multipliers (stack on volume tier):
    1,000  DRIFT (Kickstarter): -5% taker, +5% maker rebate
    100,000 DRIFT (Master):     -30% taker, +30% maker rebate
    250,000 DRIFT (Champion):   -40% taker, +40% maker rebate

  High-Leverage mode: taker fees 2x the lowest tier (~0.0400%)

  Round-trip taker (Tier 1): 0.0700% (open + close) — NOT 0.20% as assumed
  Round-trip maker (Tier 1): -0.0050% (earn rebates both sides via Post-Only orders)

KEY INSIGHT: every unnecessary market order costs ~0.07% round-trip.
At 10% APR funding, that's equivalent to ~2.5 DAYS of funding income lost.
Use Post-Only limit orders for non-urgent rebalances to turn cost → rebate.

Slippage model (Drift DLOB):
  Drift uses a Decentralised Limit Order Book (DLOB) filled by keeper bots.
  Market impact depends on order size relative to available liquidity.
  For a $10M vault:
    < $10k order:   1-3 bps slippage (negligible)
    $10k-$100k:     3-8 bps slippage
    $100k-$1M:      8-20 bps slippage
    > $1M:          20-50 bps slippage (should TWAP)

Rebalance cost-benefit:
  Only rebalance when expected APR improvement × hours_until_next_check
  exceeds the round-trip execution cost.

  Break-even formula:
    min_apr_improvement = round_trip_cost / (hours_between_rebalances / 8760)

  Example: round-trip 0.20%, check every 10h:
    min_apr_improvement = 0.20% / (10/8760) = 175% APR
  → Never rebalance unless APR changes by 175% in 10h?

  That's too conservative. Real approach: threshold on POSITION CHANGE SIZE.
  Only transact if the position change × APR_difference > execution_cost.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class OrderType(str, Enum):
    MARKET  = "MARKET"   # taker — fills immediately, pays full taker fee
    LIMIT   = "LIMIT"    # maker — earns rebate IF it rests on book and fills
    IOC     = "IOC"      # immediate-or-cancel limit — often taker in practice
    POST    = "POST"     # post-only limit — guaranteed maker, rejects if would cross


@dataclass
class DriftFeeConfig:
    """
    Drift Protocol v2 fee schedule.
    Default = Tier 1 (≤$2M/month volume). Most new vaults start here.
    """
    taker_fee_bps:    float = 3.5   # 3.5 bps = 0.035% (Tier 1 confirmed)
    maker_rebate_bps: float = 0.25  # 0.25 bps = 0.0025% rebate (Tier 1 confirmed)
    # Volume tier discount — set to 0 for Tier 1, higher for established vaults
    tier_discount_pct: float = 0.0  # 0% = Tier 1

    @property
    def taker_fee_pct(self) -> float:
        return self.taker_fee_bps / 10_000 * (1.0 - self.tier_discount_pct)

    @property
    def maker_rebate_pct(self) -> float:
        return self.maker_rebate_bps / 10_000

    def round_trip_cost_pct(self, order_type: OrderType = OrderType.MARKET) -> float:
        """Full round-trip cost (open + close) as fraction of position size."""
        if order_type in (OrderType.LIMIT, OrderType.POST):
            # Earn rebate both sides
            return -(self.maker_rebate_pct * 2)  # negative = net income
        elif order_type == OrderType.IOC:
            # Usually fills as taker
            return self.taker_fee_pct * 2
        else:
            return self.taker_fee_pct * 2


@dataclass
class SlippageModel:
    """
    Empirical slippage model for Drift DLOB.
    Maps order size (USD) → expected slippage (bps).
    Based on Drift liquidity depth data from 2024-2025.
    """
    # (order_size_usd, slippage_bps) breakpoints
    # Linear interpolation between breakpoints
    breakpoints: list[tuple[float, float]] = None

    def __post_init__(self):
        if self.breakpoints is None:
            self.breakpoints = [
                (0,          0.5),   # <$1k:      ~0.5 bps
                (10_000,     2.0),   # $10k:       2 bps
                (50_000,     5.0),   # $50k:       5 bps
                (100_000,    8.0),   # $100k:      8 bps
                (500_000,   18.0),   # $500k:     18 bps
                (1_000_000, 30.0),   # $1M:       30 bps
                (5_000_000, 60.0),   # $5M:       60 bps
            ]

    def estimate_slippage_bps(self, order_size_usd: float) -> float:
        """Linear interpolation of slippage from order size."""
        sizes  = [b[0] for b in self.breakpoints]
        slips  = [b[1] for b in self.breakpoints]
        return float(np.interp(order_size_usd, sizes, slips))

    def estimate_slippage_pct(self, order_size_usd: float) -> float:
        return self.estimate_slippage_bps(order_size_usd) / 10_000


@dataclass
class ExecutionCostEstimate:
    order_size_usd:   float
    order_type:       OrderType
    fees_pct:         float     # taker fee or maker rebate
    slippage_pct:     float     # market impact estimate
    total_cost_pct:   float     # fees + slippage (negative = net income from rebate)
    total_cost_usd:   float
    is_profitable:    bool      # True if executing improves expected APR enough


class ExecutionCostModel:
    """
    Full execution cost model combining Drift fees + slippage.

    Usage:
        model = ExecutionCostModel(nav=1_000_000)

        # Should we rebalance? (increase SOL-PERP from 15% to 20%)
        decision = model.should_rebalance(
            current_pct=0.15,
            target_pct=0.20,
            current_apr=25.0,
            target_apr=25.0,     # same APR, just sizing
            hours_held=10.0,
        )
    """

    def __init__(
        self,
        nav:          float = 100_000.0,
        fee_config:   DriftFeeConfig = None,
        slippage:     SlippageModel  = None,
        default_order_type: OrderType = OrderType.MARKET,
    ):
        self.nav  = nav
        self.fees = fee_config or DriftFeeConfig()
        self.slip = slippage  or SlippageModel()
        self.default_order_type = default_order_type

    def estimate_cost(
        self,
        position_change_pct: float,   # change in allocation as fraction of NAV
        order_type: OrderType = None,
    ) -> ExecutionCostEstimate:
        """Estimate cost of executing a position change."""
        ot = order_type or self.default_order_type
        order_size_usd = abs(position_change_pct) * self.nav

        # Fee component
        fee_pct = self.fees.round_trip_cost_pct(ot)  # open + close

        # Slippage: applies to the size being transacted
        slip_pct = self.slip.estimate_slippage_pct(order_size_usd)

        total_cost_pct = fee_pct + slip_pct   # as fraction of order size
        total_cost_usd = order_size_usd * total_cost_pct

        return ExecutionCostEstimate(
            order_size_usd=order_size_usd,
            order_type=ot,
            fees_pct=fee_pct,
            slippage_pct=slip_pct,
            total_cost_pct=total_cost_pct,
            total_cost_usd=total_cost_usd,
            is_profitable=False,  # filled by should_rebalance
        )

    def should_rebalance(
        self,
        current_pct: float,          # current allocation as fraction of NAV
        target_pct: float,           # target allocation
        current_apr: float,          # current position APR (%)
        target_apr: float,           # target position APR (%)
        hours_held: float = 10.0,    # how many hours we'll hold the new position
        order_type: OrderType = None,
    ) -> tuple[bool, ExecutionCostEstimate, float]:
        """
        Decide whether to rebalance based on cost-benefit analysis.

        Returns (should_rebalance, cost_estimate, apr_benefit_pct).

        The decision rule:
          Execute if: expected APR gain from rebalancing > execution cost
          i.e.: (target_apr - current_apr) × position_pct × (hours_held/8760)
                > round_trip_cost × |position_change_pct|
        """
        position_change = abs(target_pct - current_pct)
        if position_change < 0.001:   # < 0.1% change — not worth it
            cost = self.estimate_cost(0.0, order_type)
            return False, cost, 0.0

        cost = self.estimate_cost(position_change, order_type)

        # Expected APR gain from executing
        # If we're SIZING UP into a higher APR market: gain on new size
        # If we're ROTATING to a different market: gain from spread
        apr_diff = target_apr - current_apr
        avg_alloc = (current_pct + target_pct) / 2
        expected_gain_pct = (
            apr_diff / 100.0           # APR as fraction
            * avg_alloc                 # weighted by position size
            * hours_held / 8_760.0     # pro-rated for holding period
            * 100.0                    # back to percentage
        )

        # Cost as percentage of NAV
        cost_as_nav_pct = cost.total_cost_usd / self.nav * 100.0

        profitable = expected_gain_pct > cost_as_nav_pct
        cost.is_profitable = profitable

        logger.debug(
            "Rebalance check: change=%.1f%% NAV, apr_diff=%.1f%%, "
            "gain=%.4f%% NAV, cost=%.4f%% NAV → %s",
            position_change * 100, apr_diff, expected_gain_pct, cost_as_nav_pct,
            "EXECUTE" if profitable else "SKIP",
        )

        return profitable, cost, expected_gain_pct

    def breakeven_hours(
        self,
        position_change_pct: float,
        apr_improvement: float,
        order_type: OrderType = None,
    ) -> float:
        """
        Minimum hours to hold a rebalanced position to break even on costs.

        breakeven_hours = cost_pct / (apr_improvement × position_pct / 8760)
        """
        if apr_improvement <= 0 or position_change_pct <= 0:
            return float("inf")

        cost = self.estimate_cost(position_change_pct, order_type)
        cost_pct = cost.total_cost_usd / self.nav * 100.0

        hourly_gain = apr_improvement / 100.0 * position_change_pct / 8_760.0 * 100.0
        if hourly_gain <= 0:
            return float("inf")

        return cost_pct / hourly_gain

    def optimal_order_type(
        self,
        urgency: str = "normal",    # "urgent", "normal", "passive"
        book_depth_ratio: float = 1.0,
    ) -> OrderType:
        """
        Select order type based on urgency and market conditions.

        urgent  → MARKET (e.g., circuit breaker triggered, exit immediately)
        normal  → IOC limit at mid+1bps (fills as taker most of the time, slightly cheaper)
        passive → POST_ONLY limit (earn rebate, but may not fill — use for non-urgent sizing)
        """
        if urgency == "urgent":
            return OrderType.MARKET
        if urgency == "passive" and book_depth_ratio > 0.7:
            return OrderType.POST
        return OrderType.IOC

    def twap_schedule(
        self,
        total_size_usd: float,
        total_hours: float,
        max_single_order_usd: float = 100_000.0,
    ) -> list[dict]:
        """
        Generate TWAP (time-weighted average price) schedule for large orders.
        Splits into equal-sized slices to minimize market impact.

        Rule: single order > $100k should be TWAP'd to reduce slippage.
        """
        if total_size_usd <= max_single_order_usd:
            return [{"hour_offset": 0.0, "size_usd": total_size_usd}]

        n_slices = max(2, int(np.ceil(total_size_usd / max_single_order_usd)))
        slice_size = total_size_usd / n_slices
        interval = total_hours / n_slices

        return [
            {"hour_offset": i * interval, "size_usd": slice_size}
            for i in range(n_slices)
        ]

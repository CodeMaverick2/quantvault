"""Position limit validation and Kelly criterion-based sizing."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PositionLimits:
    max_leverage: float = 2.0
    max_single_market_pct: float = 0.30
    max_total_perp_pct: float = 0.60
    min_lending_pct: float = 0.10
    min_health_buffer: float = 0.30      # margin cushion above liquidation
    max_concentration_pct: float = 0.40  # no single position > 40% notional


@dataclass
class PositionCheckResult:
    is_valid: bool
    adjusted_size: float
    reason: str
    warnings: list[str]


class PositionValidator:
    """Validates and adjusts position sizes against configured limits."""

    def __init__(self, limits: PositionLimits | None = None):
        self.limits = limits or PositionLimits()

    def validate_perp_allocation(
        self,
        requested_pct: float,
        nav_usd: float,
        current_perp_pct: float,
        market_symbol: str,
    ) -> PositionCheckResult:
        """Validate a requested perp allocation change."""
        warnings: list[str] = []

        # Single market cap
        if requested_pct > self.limits.max_single_market_pct:
            adjusted = self.limits.max_single_market_pct
            warnings.append(
                f"{market_symbol}: reduced from {requested_pct:.1%} to {adjusted:.1%} "
                f"(single-market cap {self.limits.max_single_market_pct:.1%})"
            )
            requested_pct = adjusted

        # Total perp cap
        new_total = current_perp_pct + requested_pct
        if new_total > self.limits.max_total_perp_pct:
            excess = new_total - self.limits.max_total_perp_pct
            requested_pct = max(0.0, requested_pct - excess)
            warnings.append(
                f"Total perp allocation capped at {self.limits.max_total_perp_pct:.1%}"
            )

        return PositionCheckResult(
            is_valid=requested_pct > 0,
            adjusted_size=requested_pct,
            reason="OK" if not warnings else "; ".join(warnings),
            warnings=warnings,
        )

    def validate_leverage(
        self,
        total_collateral_usd: float,
        total_notional_usd: float,
    ) -> PositionCheckResult:
        """Validate that leverage is within limits."""
        if total_collateral_usd <= 0:
            return PositionCheckResult(False, 0.0, "Zero collateral", [])

        actual_leverage = total_notional_usd / total_collateral_usd
        if actual_leverage <= self.limits.max_leverage:
            return PositionCheckResult(True, total_notional_usd, "OK", [])

        max_notional = total_collateral_usd * self.limits.max_leverage
        return PositionCheckResult(
            is_valid=False,
            adjusted_size=max_notional,
            reason=f"Leverage {actual_leverage:.2f}x exceeds limit {self.limits.max_leverage:.2f}x",
            warnings=[f"Reduce notional from ${total_notional_usd:,.0f} to ${max_notional:,.0f}"],
        )

    def validate_health_rate(
        self,
        maintenance_margin_usd: float,
        actual_margin_usd: float,
    ) -> PositionCheckResult:
        """Ensure health rate maintains min_health_buffer above 1.0."""
        if maintenance_margin_usd <= 0:
            return PositionCheckResult(True, actual_margin_usd, "No maintenance margin", [])

        health_rate = actual_margin_usd / maintenance_margin_usd
        min_health = 1.0 + self.limits.min_health_buffer  # e.g. 1.30

        if health_rate >= min_health:
            return PositionCheckResult(True, actual_margin_usd, f"Health rate: {health_rate:.2f}", [])

        return PositionCheckResult(
            is_valid=False,
            adjusted_size=0.0,
            reason=f"Health rate {health_rate:.2f} below minimum {min_health:.2f}",
            warnings=["Reduce position to restore health rate"],
        )


def kelly_position_size(
    expected_return: float,
    variance: float,
    fraction: float = 0.25,    # fractional Kelly (0.25 = quarter Kelly)
    max_pct: float = 0.60,
) -> float:
    """
    Compute Kelly criterion position size as a fraction of portfolio.

    Full Kelly = μ / σ² (for log-normally distributed returns, where μ is the
    expected log-return and σ² is its variance — equivalent to the continuous-time
    Kelly formula assuming geometric Brownian motion for perp funding returns).
    We use fractional Kelly (0.25 = quarter Kelly) for more conservative sizing,
    reducing risk of ruin in the fat-tailed crypto environment.

    Args:
        expected_return: Expected return per period (e.g. funding APR / periods_per_year)
        variance: Variance of return per period
        fraction: Kelly fraction (0.25 = quarter Kelly, recommended for crypto)
        max_pct: Maximum allocation cap

    Returns:
        Portfolio fraction [0, max_pct]
    """
    # Guard against near-zero or negative variance to avoid numerical instability
    if variance < 1e-8:
        return 0.0

    # Full Kelly = mu/sigma^2, using fractional Kelly for crypto risk management
    full_kelly = expected_return / variance
    fractional = full_kelly * fraction
    return float(np.clip(fractional, 0.0, max_pct))

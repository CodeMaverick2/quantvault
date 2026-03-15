"""
Dynamic yield optimizer: computes target allocation percentages
across lending protocols and delta-neutral perp positions.

Uses regime signal, funding rates, and risk metrics to compute
the optimal portfolio split every 10 minutes.

Key improvements over naive delta-neutral:
  1. Dual-timeframe regime agreement gate (fast + slow HMM must agree)
  2. Kelly × (1 - cascade_score) multiplicative sizing
  3. Funding persistence filter (min 3 consecutive positive hours)
  4. Volatility-adjusted position scaling (vol-targeting)
  5. Basis-momentum confirmation for entry quality
  6. ATR-responsive leverage scaling (inversely proportional to realized vol)
  7. Time-of-day multiplier (concentrate positions during high-yield UTC windows)
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
    """Real-time yield + risk data for a single market/protocol."""
    symbol: str
    funding_apr: float             # current annualized funding rate (%)
    lending_apr: float             # lending protocol APY (%)
    is_perp: bool
    cascade_risk: float = 0.0      # 0–1 cascade risk score
    persistence_score: float = 1.0  # 0–1 funding persistence quality
    realized_vol_24h: float = 0.20  # 24h realized vol (annualized, fraction)
    consecutive_positive: int = 0   # consecutive hours of positive funding
    atr_14h: float = 0.02          # 14-period ATR as fraction of price (default 2%)


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

    # Sizing attribution for transparency
    sizing_breakdown: dict[str, dict] = field(default_factory=dict)

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
    min_persistence_score: float = 0.55          # gate: minimum entry quality
    min_consecutive_positive: int = 3            # gate: minimum consecutive positive hours
    target_vol: float = 0.15                     # target portfolio vol for vol-targeting
    cascade_risk_entry_gate: float = 0.50        # max cascade risk to allow entry


class DynamicAllocationOptimizer:
    """
    Computes target portfolio allocations using a 7-layer signal stack:

    Layer 1: Regime gate (HMM) — sets max perp budget via position_scale
    Layer 2: Funding quality gate — requires persistence + APR threshold
    Layer 3: Kelly sizing — positions sized by risk-adjusted expected return
    Layer 4: Cascade risk multiplier — Kelly × (1 - cascade_score)
    Layer 5: Vol targeting — scale down when realized vol exceeds target
    Layer 6: ATR-responsive leverage — halve leverage when ATR doubles vs baseline
    Layer 7: Time-of-day multiplier — concentrate during high-yield UTC windows

    The result is a deterministic, idempotent allocation that degrades
    gracefully as market conditions deteriorate.

    Example sizing for SOL-PERP with:
      - funding APR = 20%, vol = 20% annualized, cascade = 0.2, atr = 3%, ToD = 1.1
      - kelly_fraction(0.25x) = 0.38
      - cascade_adj = 0.38 × (1 - 0.2) = 0.30
      - vol_adj = min(1.0, 0.15/0.20) = 0.75 → 0.30 × 0.75 = 0.23
      - atr_adj = clip(0.02/0.03, 0.5, 1.5) = 0.67 → 0.23 × 0.67 = 0.15
      - ToD × perp_budget = 0.60 × 1.1 = 0.66 budget cap
      - Final = 0.15 (15% of NAV, inside budget)
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
        # Dual-timeframe HMM agreement (optional — defaults to single HMM)
        fast_regime: MarketRegime | None = None,
        fast_confidence: float = 1.0,
        # Time-of-day multiplier from TimeOfDayOptimizer [0.5, 1.5]
        tod_multiplier: float = 1.0,
    ) -> AllocationResult:
        """
        Main allocation computation.

        Args:
            markets: List of perp markets with current yield + risk data
            regime: Slow HMM regime (weekly timescale)
            regime_confidence: Slow HMM posterior probability
            drawdown_scale: Scale factor from DrawdownController [0,1]
            cb_scale: Scale factor from CircuitBreaker [0,1]
            kamino_apr: Current Kamino USDC lending APY (%)
            drift_spot_apr: Current Drift spot USDC lending APY (%)
            fast_regime: Fast HMM regime (4h timescale, optional)
            fast_confidence: Fast HMM confidence

        Returns:
            AllocationResult with normalized percentage allocations
        """
        # ── Layer 1: Regime gate ──────────────────────────────────────────────
        external_scale = drawdown_scale * cb_scale
        regime_scale = regime.position_scale() * min(1.0, regime_confidence + 0.3)

        # Dual-timeframe consensus: if fast and slow HMMs disagree, reduce scale
        if fast_regime is not None:
            if fast_regime == regime:
                # Agreement bonus: slight boost to confidence
                dual_agreement = min(1.0, (regime_confidence + fast_confidence) / 2 + 0.1)
            elif fast_regime == MarketRegime.HIGH_VOL_CRISIS or regime == MarketRegime.HIGH_VOL_CRISIS:
                # Either fast OR slow flags crisis → immediate caution
                dual_agreement = 0.0
            else:
                # Soft disagreement (BULL vs SIDEWAYS) — use conservative scale
                dual_agreement = min(regime_confidence, fast_confidence) * 0.5
            regime_scale = regime.position_scale() * dual_agreement

        effective_scale = external_scale * regime_scale

        result = AllocationResult(regime=regime.name, position_scale=effective_scale)

        # ── Layer 2: Funding quality gate ────────────────────────────────────
        eligible_markets = [
            m for m in markets
            if m.is_perp
            and m.funding_apr >= self.config.target_funding_apr_threshold
            and m.cascade_risk < self.config.cascade_risk_entry_gate
            and m.persistence_score >= self.config.min_persistence_score
            and m.consecutive_positive >= self.config.min_consecutive_positive
        ]

        # ── Layer 3+4+5+6+7: Kelly × cascade × vol × ATR × ToD ──────────────
        # ToD multiplier concentrates size during historically rich UTC windows
        # Clipped separately so regime_scale and external_scale remain unaffected
        tod_scale = float(np.clip(tod_multiplier, 0.5, 1.5))
        perp_budget = self.config.max_perp_pct * effective_scale * tod_scale
        perp_allocs: dict[str, float] = {}
        sizing_breakdown: dict[str, dict] = {}

        if eligible_markets and perp_budget > 0.01:
            perp_allocs, sizing_breakdown = self._allocate_perps(
                eligible_markets, perp_budget
            )

        result.perp_allocations = perp_allocs
        result.total_perp_pct = sum(perp_allocs.values())
        result.sizing_breakdown = sizing_breakdown

        # ── Lending fills remainder with min_lending floor ────────────────────
        if drift_spot_apr >= kamino_apr:
            drift_share = 0.60
            kamino_share = 0.40
        else:
            drift_share = 0.40
            kamino_share = 0.60

        lending_from_remainder = 1.0 - result.total_perp_pct
        total_lending = max(lending_from_remainder, self.config.min_lending_pct)

        # If min_lending forces total > 1.0, scale down perps proportionally
        if result.total_perp_pct + total_lending > 1.0:
            total_lending = self.config.min_lending_pct
            perp_budget_capped = 1.0 - total_lending
            if result.total_perp_pct > perp_budget_capped:
                scale = perp_budget_capped / result.total_perp_pct if result.total_perp_pct > 0 else 0.0
                perp_allocs = {sym: v * scale for sym, v in perp_allocs.items()}
                result.perp_allocations = perp_allocs
                result.total_perp_pct = sum(perp_allocs.values())

        result.drift_spot_lending_pct = total_lending * drift_share
        result.kamino_lending_pct = total_lending * kamino_share
        result.total_lending_pct = total_lending

        # Final normalization
        total = result.total_perp_pct + result.total_lending_pct
        if abs(total - 1.0) > 0.005:
            adjustment = 1.0 - result.total_perp_pct
            result.drift_spot_lending_pct = adjustment * drift_share
            result.kamino_lending_pct = adjustment * kamino_share
            result.total_lending_pct = adjustment

        result.total_pct = result.total_perp_pct + result.total_lending_pct

        # ── Expected blended APR ──────────────────────────────────────────────
        funding_apr_map: dict[str, float] = {m.symbol: m.funding_apr for m in eligible_markets}
        blended = (
            result.kamino_lending_pct * kamino_apr
            + result.drift_spot_lending_pct * drift_spot_apr
            + sum(
                alloc * funding_apr_map.get(sym, 0.0)
                for sym, alloc in perp_allocs.items()
            )
        )
        result.expected_blended_apr = blended

        logger.info(
            "Allocation: regime=%s(fast=%s) scale=%.2f perp=%.1f%% lending=%.1f%% "
            "E[APR]=%.1f%% eligible=%d/%d",
            regime.name,
            fast_regime.name if fast_regime else "N/A",
            effective_scale,
            result.total_perp_pct * 100,
            result.total_lending_pct * 100,
            result.expected_blended_apr,
            len(eligible_markets),
            len(markets),
        )

        return result

    def _allocate_perps(
        self,
        markets: list[MarketYieldData],
        budget: float,
    ) -> tuple[dict[str, float], dict[str, dict]]:
        """
        Allocate perp budget using Kelly × (1 - cascade) × vol_adjustment.

        Returns (allocations, sizing_breakdown).
        """
        kelly_cascade_sizes: dict[str, float] = {}
        breakdown: dict[str, dict] = {}

        for m in markets:
            # Kelly sizing: use funding APR as expected return per-hour
            # and market vol for variance estimate
            period_return = m.funding_apr / 100.0 / (24 * 365.25)  # per-hour fractional
            annualized_vol = max(m.realized_vol_24h, 0.10)          # minimum 10% vol floor
            period_variance = (annualized_vol / np.sqrt(24 * 365.25)) ** 2

            kelly_raw = kelly_position_size(
                expected_return=period_return,
                variance=period_variance,
                fraction=0.25,   # 25% fractional Kelly
                max_pct=self.config.max_single_perp_pct,
            )

            # Layer 4: Cascade risk multiplier — reduce exposure in risky conditions
            # High cascade score dramatically reduces position
            cascade_adj = kelly_raw * (1.0 - m.cascade_risk)

            # Layer 5: Vol targeting — scale down when realized vol exceeds target
            if m.realized_vol_24h > 0:
                vol_scale = min(1.0, self.config.target_vol / m.realized_vol_24h)
            else:
                vol_scale = 1.0

            # Layer 6: ATR-responsive leverage — reduce leverage in high-ATR regimes
            # Scale inversely: if ATR doubles, leverage halves (capped 0.5x – 1.5x)
            # BASE_ATR = 2% represents a "normal" market; higher ATR → smaller position
            BASE_ATR = 0.02
            atr = max(m.atr_14h, 0.005)  # floor at 0.5% to avoid div by zero
            atr_leverage_scale = float(np.clip(BASE_ATR / atr, 0.5, 1.5))

            final_size = cascade_adj * vol_scale * atr_leverage_scale

            kelly_cascade_sizes[m.symbol] = max(final_size, 0.0)
            breakdown[m.symbol] = {
                "kelly_raw": round(kelly_raw, 4),
                "cascade_adj": round(cascade_adj, 4),
                "vol_scale": round(vol_scale, 4),
                "atr_leverage_scale": round(atr_leverage_scale, 4),
                "final_pre_budget": round(final_size, 4),
                "funding_apr": m.funding_apr,
                "cascade_risk": m.cascade_risk,
                "persistence_score": m.persistence_score,
                "realized_vol_24h": m.realized_vol_24h,
                "atr_14h": m.atr_14h,
            }

        total_kelly = sum(kelly_cascade_sizes.values())
        if total_kelly <= 0:
            return {}, breakdown

        # Scale to fit within budget
        allocs: dict[str, float] = {}
        for sym, size in kelly_cascade_sizes.items():
            normalized = (size / total_kelly) * budget
            allocs[sym] = min(normalized, self.config.max_single_perp_pct)
            if sym in breakdown:
                breakdown[sym]["final_alloc"] = round(allocs[sym], 4)

        # Re-normalize after capping
        total_alloc = sum(allocs.values())
        if total_alloc > budget:
            scale = budget / total_alloc
            allocs = {sym: v * scale for sym, v in allocs.items()}

        return {sym: v for sym, v in allocs.items() if v > 0.005}, breakdown

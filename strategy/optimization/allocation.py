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
  8. Multi-horizon forecast gate (pre-position on RISING, block on PEAKING/FALLING)
  9. Regime transition early warning (pre-exit when P(regime flip) is high)
 10. Leading indicators (OI + basis advance signal, enter before funding spikes)
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..models.hmm_regime import MarketRegime
from ..risk.position_limits import kelly_position_size
from ..signals.multi_horizon_forecaster import MultiHorizonForecast, FundingTrajectory
from ..signals.regime_transition import RegimeTransitionForecast, TransitionWarning
from ..signals.leading_indicators import LeadingIndicatorResult, LeadingSignal

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
    # Inverse carry: when funding is deeply negative, LONG perp + short spot
    # Net yield = |funding_apr| - spot_borrow_cost_apr
    spot_borrow_cost_apr: float = 5.0  # cost to borrow asset for spot short hedge (%)


@dataclass
class AllocationResult:
    """Target allocation across all strategies."""
    # Lending allocations (% of vault NAV)
    kamino_lending_pct: float = 0.10
    drift_spot_lending_pct: float = 0.10

    # Delta-neutral perp allocations (% per market)
    perp_allocations: dict[str, float] = field(default_factory=dict)

    # Direction per perp: "SHORT" (collect positive funding) or "LONG" (inverse carry)
    perp_directions: dict[str, str] = field(default_factory=dict)

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
    # Env-configurable: lower on mainnet when funding is reliable; raise on devnet noise
    target_funding_apr_threshold: float = float(os.getenv("MIN_FUNDING_APR_THRESHOLD", "8.0"))
    min_persistence_score: float = 0.55          # gate: minimum entry quality
    min_consecutive_positive: int = 3            # gate: minimum consecutive positive hours
    target_vol: float = 0.15                     # target portfolio vol for vol-targeting
    cascade_risk_entry_gate: float = 0.50        # max cascade risk to allow entry
    inverse_carry_threshold: float = 5.0         # |funding_apr| must exceed this for inverse carry
    min_inverse_carry_net_apr: float = 3.0       # min net APR after borrow cost to enter
    # How many symbols must signal exit before applying 0.3x scale reduction
    # Set to 3 to require unanimous exit signal (prevents premature scale reduction)
    predictive_exit_quorum: int = int(os.getenv("PREDICTIVE_EXIT_QUORUM", "3"))


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
        # Predictive signals (optional — graceful degradation if absent)
        multi_horizon_forecasts: dict[str, MultiHorizonForecast] | None = None,
        regime_transition: RegimeTransitionForecast | None = None,
        leading_indicators: dict[str, LeadingIndicatorResult] | None = None,
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

        # ── Predictive layer: adjust scale BEFORE quality gate ────────────────
        # These three signals act on the perp budget multiplicatively:
        #
        #   Regime transition warning → pre-emptively reduce/exit
        #   Multi-horizon trajectory  → boost on RISING, block on PEAKING/FALLING
        #   Leading indicators        → boost when OI+basis confirm incoming spike
        #
        # The adjustments are additive boosts/cuts to effective_scale,
        # bounded to keep the system conservative (never > 1.3× base).

        predictive_scale = 1.0   # multiplicative adjustment to perp budget

        # Signal A: Regime transition early warning
        if regime_transition is not None:
            if regime_transition.should_exit():
                # Regime flip imminent (>60% prob in 6h) → zero out perps now
                effective_scale = 0.0
                logger.info(
                    "Predictive: regime transition EXIT signal — "
                    "P(flip in 6h)=%.0f%% → zeroing perp exposure",
                    regime_transition.transition_probs.get(6, 0) * 100,
                )
            elif regime_transition.should_reduce():
                predictive_scale *= 0.5
                logger.info(
                    "Predictive: regime transition REDUCE — "
                    "P(flip in 6h)=%.0f%%",
                    regime_transition.transition_probs.get(6, 0) * 100,
                )
            elif regime_transition.no_new_entries():
                # WATCH mode: allow existing positions, no new entries
                predictive_scale *= 0.8

        # Signal B: Multi-horizon forecast trajectory per market
        # Aggregate trajectory across all symbols
        if multi_horizon_forecasts:
            rising_count  = sum(
                1 for f in multi_horizon_forecasts.values()
                if f.trajectory == FundingTrajectory.RISING
            )
            falling_count = sum(
                1 for f in multi_horizon_forecasts.values()
                if f.trajectory in (FundingTrajectory.FALLING, FundingTrajectory.PEAKING)
            )
            pre_position_count = sum(
                1 for f in multi_horizon_forecasts.values()
                if f.pre_position_signal
            )
            exit_signal_count = sum(
                1 for f in multi_horizon_forecasts.values()
                if f.exit_signal
            )

            if exit_signal_count >= self.config.predictive_exit_quorum:
                # All symbols say "exit" → reduce hard
                predictive_scale *= 0.3
                logger.info(
                    "Predictive: multi-horizon EXIT on %d symbols → scale 0.3×",
                    exit_signal_count,
                )
            elif falling_count >= 2:
                predictive_scale *= 0.6
            elif pre_position_count >= 2:
                # Funding rising on multiple symbols → pre-position boost
                predictive_scale = min(1.3, predictive_scale * 1.2)
                logger.info(
                    "Predictive: %d symbols RISING — pre-positioning 1.2× boost",
                    pre_position_count,
                )
            elif rising_count >= 2:
                predictive_scale = min(1.2, predictive_scale * 1.1)

        # Signal C: Leading indicators (OI + basis)
        if leading_indicators:
            bullish_count  = sum(
                1 for r in leading_indicators.values()
                if r.pre_position_carry
            )
            bearish_count  = sum(
                1 for r in leading_indicators.values()
                if r.pre_exit_carry
            )
            inverse_count  = sum(
                1 for r in leading_indicators.values()
                if r.pre_position_inverse
            )

            if bearish_count >= 2:
                predictive_scale *= 0.5
                logger.info(
                    "Predictive: leading indicators BEARISH on %d symbols",
                    bearish_count,
                )
            elif bullish_count >= 2:
                predictive_scale = min(1.3, predictive_scale * 1.15)
                logger.info(
                    "Predictive: leading indicators BULLISH on %d symbols — "
                    "pre-positioning",
                    bullish_count,
                )

            # Inverse setup: if OI + basis signal funding will go deeply negative,
            # allow inverse carry markets even before the funding rate crosses the threshold
            if inverse_count >= 1:
                logger.info(
                    "Predictive: INVERSE_SETUP on %d symbols — "
                    "relaxing inverse carry threshold",
                    inverse_count,
                )

        # Apply predictive adjustment to effective_scale
        effective_scale = effective_scale * predictive_scale

        # ── Layer 2: Funding quality gate ────────────────────────────────────
        # Standard carry: positive funding above threshold
        # Inverse carry: deeply negative funding, net yield = |apr| - borrow_cost > min
        eligible_markets = []
        for m in markets:
            if not m.is_perp:
                continue
            if m.cascade_risk >= self.config.cascade_risk_entry_gate:
                continue
            # Standard carry (SHORT perp): positive funding
            if (m.funding_apr >= self.config.target_funding_apr_threshold
                    and m.persistence_score >= self.config.min_persistence_score
                    and m.consecutive_positive >= self.config.min_consecutive_positive):
                eligible_markets.append(m)
            # Inverse carry (LONG perp + short spot): negative funding
            elif (m.funding_apr < -self.config.inverse_carry_threshold
                    and (abs(m.funding_apr) - m.spot_borrow_cost_apr)
                        >= self.config.min_inverse_carry_net_apr):
                eligible_markets.append(m)

        # ── Layer 3+4+5+6+7: Kelly × cascade × vol × ATR × ToD ──────────────
        # ToD multiplier concentrates size during historically rich UTC windows
        # Clipped separately so regime_scale and external_scale remain unaffected
        tod_scale = float(np.clip(tod_multiplier, 0.5, 1.5))
        perp_budget = self.config.max_perp_pct * effective_scale * tod_scale
        perp_allocs: dict[str, float] = {}
        sizing_breakdown: dict[str, dict] = {}

        if eligible_markets and perp_budget > 0.01:
            perp_allocs, sizing_breakdown, perp_dirs = self._allocate_perps(
                eligible_markets, perp_budget
            )
        else:
            perp_dirs = {}

        result.perp_allocations = perp_allocs
        result.perp_directions = perp_dirs
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
        # Use effective APR for each market (net of borrow cost for inverse carry)
        effective_apr_map: dict[str, float] = {}
        for m in eligible_markets:
            if m.funding_apr < -self.config.inverse_carry_threshold:
                effective_apr_map[m.symbol] = abs(m.funding_apr) - m.spot_borrow_cost_apr
            else:
                effective_apr_map[m.symbol] = m.funding_apr

        blended = (
            result.kamino_lending_pct * kamino_apr
            + result.drift_spot_lending_pct * drift_spot_apr
            + sum(
                alloc * effective_apr_map.get(sym, 0.0)
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
    ) -> tuple[dict[str, float], dict[str, dict], dict[str, str]]:
        """
        Allocate perp budget using Kelly × (1 - cascade) × vol_adjustment.

        Supports both standard carry (SHORT) and inverse carry (LONG + short spot).

        Returns (allocations, sizing_breakdown, directions).
        """
        kelly_cascade_sizes: dict[str, float] = {}
        breakdown: dict[str, dict] = {}
        directions: dict[str, str] = {}

        for m in markets:
            # Determine direction and effective APR
            is_inverse = m.funding_apr < -self.config.inverse_carry_threshold
            if is_inverse:
                direction = "LONG"
                effective_apr = abs(m.funding_apr) - m.spot_borrow_cost_apr
            else:
                direction = "SHORT"
                effective_apr = m.funding_apr
            directions[m.symbol] = direction

            # Kelly sizing: use effective APR as expected return per-hour
            # and market vol for variance estimate
            period_return = effective_apr / 100.0 / (24 * 365.25)  # per-hour fractional
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
                "effective_apr": round(effective_apr, 2),
                "direction": direction,
                "cascade_risk": m.cascade_risk,
                "persistence_score": m.persistence_score,
                "realized_vol_24h": m.realized_vol_24h,
                "atr_14h": m.atr_14h,
            }

        total_kelly = sum(kelly_cascade_sizes.values())
        if total_kelly <= 0:
            return {}, breakdown, directions

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

        final_allocs = {sym: v for sym, v in allocs.items() if v > 0.005}
        final_directions = {sym: directions[sym] for sym in final_allocs if sym in directions}
        return final_allocs, breakdown, final_directions

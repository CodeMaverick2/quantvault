"""Real-time cascade risk scoring using order book and market microstructure signals."""

import logging
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CascadeRiskInput:
    """All signals needed to compute cascade risk score."""

    # Order book imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol) — range [-1, 1]
    ob_imbalance: float = 0.0

    # Funding rate as a percentile [0, 1] of recent 30-day distribution
    funding_percentile: float = 0.5

    # Open interest as percentile [0, 1] of recent 30-day distribution
    oi_percentile: float = 0.5

    # Book depth ratio: current / 24h average (< 1 = thin book)
    book_depth_ratio: float = 1.0

    # Basis percentage: (perp - spot) / spot
    basis_pct: float = 0.0

    # Recent liquidation volume (USD) as percentile
    liquidation_percentile: float = 0.0

    # Rate of change of OI (positive = growing leverage)
    oi_change_1h: float = 0.0

    # Price return over last hour (negative = selling pressure)
    price_return_1h: float = 0.0


@dataclass
class CascadeRiskResult:
    score: float                         # 0.0 (no risk) — 1.0 (extreme risk)
    triggered: bool                      # True if score >= threshold
    components: dict[str, float] = field(default_factory=dict)
    dominant_signal: str = ""
    recommendation: str = ""


class CascadeRiskScorer:
    """
    Computes a composite cascade risk score from market microstructure inputs.

    Scores above 0.70 trigger the circuit breaker (emergency position reduction).

    Based on anatomy of the October 2025 liquidation cascade:
    - OBI flipped from +0.057 to -0.22 (60s)
    - Book depth dropped 98% ($103M → $0.17M)
    - Funding rate at 99th percentile before the cascade
    - OI at multi-month highs
    """

    def __init__(
        self,
        trigger_threshold: float = 0.70,
        weights: dict[str, float] | None = None,
    ):
        self.trigger_threshold = trigger_threshold
        self.weights = weights or {
            "obi": 0.25,
            "funding": 0.25,
            "oi": 0.20,
            "book_depth": 0.15,
            "liquidation": 0.10,
            "basis": 0.05,
        }
        # Validate weights sum to ~1
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning("Cascade risk weights sum to %.3f (expected 1.0)", total)

        # Rolling history for percentile calculations
        self._funding_history: list[float] = []
        self._oi_history: list[float] = []
        self._liquidation_history: list[float] = []
        self._depth_history: list[float] = []

    def score(self, inp: CascadeRiskInput) -> CascadeRiskResult:
        """Compute cascade risk score from inputs."""
        components: dict[str, float] = {}

        # OBI component: negative OBI (ask pressure) = selling pressure
        # Normalized so that -1.0 OBI → 1.0 risk, +1.0 OBI → 0.0 risk
        components["obi"] = max(0.0, (0.0 - inp.ob_imbalance) / 2.0 + 0.5)

        # Funding rate: extreme positive funding = overcrowded longs = cascade risk
        components["funding"] = inp.funding_percentile

        # OI: high OI = maximum leverage = amplified cascade potential
        components["oi"] = inp.oi_percentile

        # Book depth: thin book = cascades propagate further
        # depth_ratio < 0.30 → max risk; depth_ratio = 1.0 → no depth risk
        components["book_depth"] = max(0.0, 1.0 - inp.book_depth_ratio)

        # Liquidation volume: elevated recent liquidations = cascade in progress
        components["liquidation"] = inp.liquidation_percentile

        # Basis blowout: extreme basis indicates stress / dislocated market
        components["basis"] = min(1.0, abs(inp.basis_pct) / 0.03)  # 3% basis = max

        # Weighted composite
        composite = sum(
            self.weights.get(k, 0.0) * v for k, v in components.items()
        )
        composite = float(np.clip(composite, 0.0, 1.0))

        # Interaction multiplier: OBI + OI + funding all extreme simultaneously = amplified
        extreme_count = sum([
            components["obi"] > 0.75,
            components["funding"] > 0.85,
            components["oi"] > 0.85,
            components["book_depth"] > 0.70,
        ])
        if extreme_count >= 3:
            composite = min(1.0, composite * 1.25)  # 25% amplification

        triggered = composite >= self.trigger_threshold
        dominant = max(components, key=lambda k: self.weights.get(k, 0) * components[k])

        recommendation = self._get_recommendation(composite, components)

        return CascadeRiskResult(
            score=composite,
            triggered=triggered,
            components=components,
            dominant_signal=dominant,
            recommendation=recommendation,
        )

    def update_history(
        self,
        funding_rate: float,
        open_interest: float,
        liquidation_volume: float,
        book_depth: float,
        max_history: int = 720,  # 30 days at hourly
    ) -> None:
        """Update rolling history for percentile calculations."""
        self._funding_history.append(funding_rate)
        self._oi_history.append(open_interest)
        self._liquidation_history.append(liquidation_volume)
        self._depth_history.append(book_depth)

        if len(self._funding_history) > max_history:
            self._funding_history = self._funding_history[-max_history:]
            self._oi_history = self._oi_history[-max_history:]
            self._liquidation_history = self._liquidation_history[-max_history:]
            self._depth_history = self._depth_history[-max_history:]

    def compute_percentile(self, value: float, history: list[float]) -> float:
        """Compute where `value` falls in the historical distribution."""
        if len(history) < 10:
            return 0.5  # insufficient history → neutral
        below = sum(1 for x in history if x <= value)
        return below / len(history)

    def build_input_from_market(
        self,
        ob_imbalance: float,
        funding_rate: float,
        open_interest: float,
        book_depth: float,
        liquidation_volume: float = 0.0,
        basis_pct: float = 0.0,
        oi_change_1h: float = 0.0,
        price_return_1h: float = 0.0,
    ) -> CascadeRiskInput:
        """
        Build a CascadeRiskInput using rolling history for percentile normalization.
        Call update_history() first with recent data.
        """
        funding_pct = self.compute_percentile(funding_rate, self._funding_history)
        oi_pct = self.compute_percentile(open_interest, self._oi_history)
        liq_pct = self.compute_percentile(liquidation_volume, self._liquidation_history)

        avg_depth = mean(self._depth_history) if self._depth_history else book_depth
        depth_ratio = book_depth / (avg_depth + 1e-9)

        return CascadeRiskInput(
            ob_imbalance=ob_imbalance,
            funding_percentile=funding_pct,
            oi_percentile=oi_pct,
            book_depth_ratio=depth_ratio,
            basis_pct=basis_pct,
            liquidation_percentile=liq_pct,
            oi_change_1h=oi_change_1h,
            price_return_1h=price_return_1h,
        )

    @staticmethod
    def _get_recommendation(score: float, components: dict[str, float]) -> str:
        if score >= 0.85:
            return "EMERGENCY_EXIT: Close all perp positions immediately. Flee to lending-only."
        if score >= 0.70:
            return "CIRCUIT_BREAKER: Reduce perp exposure by 75%. Increase lending allocation."
        if score >= 0.50:
            return "ELEVATED_RISK: Reduce perp exposure by 30%. Tighten hedge ratios."
        if score >= 0.30:
            return "MODERATE_RISK: Monitor closely. Keep current positions, no new entries."
        return "LOW_RISK: Normal operation."

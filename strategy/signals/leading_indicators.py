"""
Leading Indicators for Funding Rate and Regime Prediction.

These are the signals that move BEFORE funding rates change, giving the
system hours of advance warning to pre-position rather than react.

The two most reliable leading indicators in crypto perpetuals:

1. Open Interest (OI) Dynamics
   ─────────────────────────────
   OI measures total outstanding leveraged contracts.
   When OI builds rapidly, leveraged longs are accumulating → they will
   start paying more funding soon. OI accelerating = funding spike incoming.
   When OI collapses, deleveraging is occurring → funding will fall/flip.

   Timeline: OI changes typically lead funding by 2-8 hours.

2. Basis (Perp Premium)
   ─────────────────────
   Basis = (perp_price - spot_price) / spot_price × 100 (%)
   A widening basis means perp is trading at a premium to spot →
   arbitrageurs will short perp and buy spot → funding will rise to
   compensate longs. Basis is effectively a real-time forward signal
   for where funding is heading.

   Timeline: basis changes typically lead funding by 1-4 hours.

3. Liquidation Flow
   ─────────────────
   Large liquidation events cause cascading deleveraging → funding collapses.
   Tracking liquidation volume gives early warning of funding inversions.

Composite Leading Score:
   Combines OI, basis, and liquidation signals into a single [−1, +1] score:
     +1.0 = strong bullish carry signal → funding will rise → pre-position
     −1.0 = strong bearish signal → funding will fall/flip → pre-exit
      0.0 = neutral, no directional conviction
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Minimum history (hours) before signals are meaningful
MIN_OI_HISTORY    = 6
MIN_BASIS_HISTORY = 4


class LeadingSignal(str, Enum):
    STRONG_BULLISH  = "STRONG_BULLISH"   # pre-position for rising carry
    BULLISH         = "BULLISH"          # lean long carry, normal sizing
    NEUTRAL         = "NEUTRAL"          # no directional edge
    BEARISH         = "BEARISH"          # reduce carry exposure
    STRONG_BEARISH  = "STRONG_BEARISH"   # pre-exit, funding inversion likely
    INVERSE_SETUP   = "INVERSE_SETUP"    # setup for inverse carry (deeply negative soon)


@dataclass
class OIAnalysis:
    current_oi: float           # latest OI (USD)
    oi_change_1h_pct: float     # % change in OI over last hour
    oi_change_6h_pct: float     # % change in OI over last 6h
    oi_zscore: float            # current OI vs 30-day distribution
    trend: str                  # "BUILDING", "DECLINING", "FLAT"
    acceleration: float         # rate of change of the rate of change
    leverage_signal: str        # "OVERLEVERAGED", "NORMAL", "DELEVERAGING"


@dataclass
class BasisAnalysis:
    basis_pct: float            # (perp - spot) / spot × 100
    basis_zscore: float         # vs recent distribution
    basis_trend: str            # "EXPANDING", "CONTRACTING", "FLAT"
    basis_velocity: float       # hourly change in basis (% per hour)
    funding_lead_signal: str    # "PRE_SPIKE", "PRE_COLLAPSE", "NEUTRAL"
    expected_funding_direction: str  # "UP", "DOWN", "FLAT"


@dataclass
class LiquidationAnalysis:
    recent_liq_volume: float        # USD liquidated in last hour
    liq_zscore: float               # vs 30-day distribution
    cascade_risk: str               # "HIGH", "MEDIUM", "LOW"
    funding_impact: str             # "WILL_COLLAPSE", "ELEVATED", "NORMAL"


@dataclass
class LeadingIndicatorResult:
    symbol: str
    composite_score: float          # −1.0 to +1.0
    signal: LeadingSignal
    oi: OIAnalysis
    basis: BasisAnalysis
    liquidations: LiquidationAnalysis
    hours_ahead_estimate: int       # estimated hours before funding impact
    pre_position_carry: bool        # enter carry trade now (before rate rises)
    pre_exit_carry: bool            # exit carry now (before rate falls)
    pre_position_inverse: bool      # enter inverse carry (funding going very negative)
    explanation: str                # human-readable rationale


class LeadingIndicatorEngine:
    """
    Tracks OI, basis, and liquidation flow per symbol.
    Produces a composite leading score and actionable signal.

    Usage:
        engine = LeadingIndicatorEngine()

        # Feed data every hour
        engine.update("SOL-PERP",
            oi=5_000_000,
            perp_price=150.0,
            spot_price=149.5,
            liq_volume_1h=200_000,
        )

        result = engine.analyze("SOL-PERP")
        if result.pre_position_carry:
            # OI building + basis expanding → funding will rise → enter now
            ...
    """

    def __init__(
        self,
        oi_window:    int = 48,
        basis_window: int = 24,
        liq_window:   int = 72,
    ):
        self.oi_window    = oi_window
        self.basis_window = basis_window
        self.liq_window   = liq_window

        # Per-symbol rolling buffers
        self._oi:    dict[str, deque] = {}
        self._basis: dict[str, deque] = {}
        self._liq:   dict[str, deque] = {}

    def update(
        self,
        symbol: str,
        oi: float,
        perp_price: float,
        spot_price: float,
        liq_volume_1h: float = 0.0,
    ) -> None:
        """Record latest market data for a symbol."""
        if symbol not in self._oi:
            self._oi[symbol]    = deque(maxlen=self.oi_window)
            self._basis[symbol] = deque(maxlen=self.basis_window)
            self._liq[symbol]   = deque(maxlen=self.liq_window)

        self._oi[symbol].append(float(oi))

        basis = ((perp_price - spot_price) / spot_price * 100.0
                 if spot_price != 0 else 0.0)
        self._basis[symbol].append(float(basis))
        self._liq[symbol].append(float(liq_volume_1h))

    def analyze(self, symbol: str) -> LeadingIndicatorResult:
        oi_buf    = list(self._oi.get(symbol,    []))
        basis_buf = list(self._basis.get(symbol, []))
        liq_buf   = list(self._liq.get(symbol,   []))

        oi_analysis   = self._analyze_oi(oi_buf)
        basis_analysis = self._analyze_basis(basis_buf)
        liq_analysis  = self._analyze_liquidations(liq_buf)

        composite, signal, hours_ahead, explanation = self._composite(
            oi_analysis, basis_analysis, liq_analysis
        )

        pre_position_carry   = signal in (LeadingSignal.STRONG_BULLISH, LeadingSignal.BULLISH)
        pre_exit_carry       = signal in (LeadingSignal.STRONG_BEARISH, LeadingSignal.BEARISH)
        pre_position_inverse = signal == LeadingSignal.INVERSE_SETUP

        return LeadingIndicatorResult(
            symbol=symbol,
            composite_score=round(composite, 3),
            signal=signal,
            oi=oi_analysis,
            basis=basis_analysis,
            liquidations=liq_analysis,
            hours_ahead_estimate=hours_ahead,
            pre_position_carry=pre_position_carry,
            pre_exit_carry=pre_exit_carry,
            pre_position_inverse=pre_position_inverse,
            explanation=explanation,
        )

    def analyze_all(self, symbols: list[str]) -> dict[str, LeadingIndicatorResult]:
        return {sym: self.analyze(sym) for sym in symbols}

    # ── Internal analysis ─────────────────────────────────────────────────────

    def _analyze_oi(self, buf: list[float]) -> OIAnalysis:
        if len(buf) < MIN_OI_HISTORY:
            return OIAnalysis(
                current_oi=buf[-1] if buf else 0.0,
                oi_change_1h_pct=0.0,
                oi_change_6h_pct=0.0,
                oi_zscore=0.0,
                trend="FLAT",
                acceleration=0.0,
                leverage_signal="NORMAL",
            )

        arr = np.array(buf)
        current = float(arr[-1])

        # % changes
        ch_1h = float((arr[-1] - arr[-2]) / arr[-2] * 100.0) if arr[-2] != 0 else 0.0
        ch_6h = float((arr[-1] - arr[max(-7, -len(arr))]) / arr[max(-7, -len(arr))] * 100.0) if len(arr) >= 7 else ch_1h

        # Z-score vs window distribution
        mu  = float(arr.mean())
        std = float(arr.std()) + 1e-10
        zscore = (current - mu) / std

        # Trend classification
        # Use linear regression slope over recent 6 hours
        recent = arr[-min(7, len(arr)):]
        x = np.arange(len(recent), dtype=float)
        slope = float(np.polyfit(x, recent, 1)[0])
        slope_pct = slope / (mu + 1e-10) * 100.0

        if slope_pct > 0.3:
            trend = "BUILDING"
        elif slope_pct < -0.3:
            trend = "DECLINING"
        else:
            trend = "FLAT"

        # Acceleration: compare slope of first half vs second half
        if len(recent) >= 4:
            mid = len(recent) // 2
            slope_early = float(np.polyfit(np.arange(mid, dtype=float), recent[:mid], 1)[0])
            slope_late  = float(np.polyfit(np.arange(len(recent) - mid, dtype=float), recent[mid:], 1)[0])
            acceleration = float((slope_late - slope_early) / (abs(slope_early) + 1e-10))
        else:
            acceleration = 0.0

        # Leverage signal
        if zscore > 1.5 and trend == "BUILDING":
            leverage_signal = "OVERLEVERAGED"
        elif trend == "DECLINING" and ch_6h < -5.0:
            leverage_signal = "DELEVERAGING"
        else:
            leverage_signal = "NORMAL"

        return OIAnalysis(
            current_oi=round(current, 0),
            oi_change_1h_pct=round(ch_1h, 3),
            oi_change_6h_pct=round(ch_6h, 3),
            oi_zscore=round(zscore, 3),
            trend=trend,
            acceleration=round(acceleration, 3),
            leverage_signal=leverage_signal,
        )

    def _analyze_basis(self, buf: list[float]) -> BasisAnalysis:
        if len(buf) < MIN_BASIS_HISTORY:
            current = buf[-1] if buf else 0.0
            return BasisAnalysis(
                basis_pct=current,
                basis_zscore=0.0,
                basis_trend="FLAT",
                basis_velocity=0.0,
                funding_lead_signal="NEUTRAL",
                expected_funding_direction="FLAT",
            )

        arr = np.array(buf)
        current = float(arr[-1])
        mu      = float(arr.mean())
        std     = float(arr.std()) + 1e-10
        zscore  = (current - mu) / std

        # Velocity: change per hour over last 4 hours
        recent   = arr[-min(5, len(arr)):]
        velocity = float(np.polyfit(np.arange(len(recent), dtype=float), recent, 1)[0])

        # Trend
        if velocity > 0.02:
            trend = "EXPANDING"
        elif velocity < -0.02:
            trend = "CONTRACTING"
        else:
            trend = "FLAT"

        # Funding lead signal:
        # Expanding basis (perp > spot by growing margin) → longs paying more → funding rises
        # Contracting / negative basis → funding will fall
        if trend == "EXPANDING" and current > 0.05:
            lead = "PRE_SPIKE"
            direction = "UP"
        elif trend == "CONTRACTING" and current < -0.05:
            lead = "PRE_COLLAPSE"
            direction = "DOWN"
        elif current < -0.2 and trend == "CONTRACTING":
            lead = "PRE_INVERSION"
            direction = "DOWN"
        else:
            lead = "NEUTRAL"
            direction = "FLAT"

        return BasisAnalysis(
            basis_pct=round(current, 4),
            basis_zscore=round(zscore, 3),
            basis_trend=trend,
            basis_velocity=round(velocity, 5),
            funding_lead_signal=lead,
            expected_funding_direction=direction,
        )

    def _analyze_liquidations(self, buf: list[float]) -> LiquidationAnalysis:
        if not buf:
            return LiquidationAnalysis(
                recent_liq_volume=0.0,
                liq_zscore=0.0,
                cascade_risk="LOW",
                funding_impact="NORMAL",
            )

        arr = np.array(buf)
        current = float(arr[-1])
        mu      = float(arr.mean())
        std     = float(arr.std()) + 1e-10
        zscore  = (current - mu) / std

        if zscore > 2.0:
            cascade = "HIGH"
            impact  = "WILL_COLLAPSE"
        elif zscore > 1.0:
            cascade = "MEDIUM"
            impact  = "ELEVATED"
        else:
            cascade = "LOW"
            impact  = "NORMAL"

        return LiquidationAnalysis(
            recent_liq_volume=round(current, 0),
            liq_zscore=round(zscore, 3),
            cascade_risk=cascade,
            funding_impact=impact,
        )

    def _composite(
        self,
        oi: OIAnalysis,
        basis: BasisAnalysis,
        liq: LiquidationAnalysis,
    ) -> tuple[float, LeadingSignal, int, str]:
        """
        Combine OI, basis, and liquidation signals into a composite score.
        Returns (score, signal, hours_ahead_estimate, explanation).

        Weights:
          Basis:        40% (fastest leading indicator, 1-4h lead)
          OI trend:     35% (medium lead, 2-8h)
          Liquidations: 25% (coincident-to-lagging for collapse signals)
        """
        # Basis component [−1, +1]
        if basis.funding_lead_signal == "PRE_SPIKE":
            basis_score = min(1.0, 0.5 + basis.basis_zscore * 0.2)
        elif basis.funding_lead_signal in ("PRE_COLLAPSE", "PRE_INVERSION"):
            basis_score = max(-1.0, -0.5 + basis.basis_zscore * 0.2)
        else:
            basis_score = float(np.clip(basis.basis_zscore * 0.3, -0.5, 0.5))

        # OI component [−1, +1]
        if oi.trend == "BUILDING" and oi.leverage_signal == "OVERLEVERAGED":
            oi_score = 0.6 + min(0.4, oi.acceleration * 0.2)
        elif oi.trend == "BUILDING":
            oi_score = 0.3 + min(0.3, oi.oi_change_6h_pct * 0.02)
        elif oi.trend == "DECLINING" and oi.leverage_signal == "DELEVERAGING":
            oi_score = -0.7
        elif oi.trend == "DECLINING":
            oi_score = -0.3
        else:
            oi_score = 0.0

        # Liquidation component [−1, 0] (liquidations are always bearish for funding)
        if liq.cascade_risk == "HIGH":
            liq_score = -0.8
        elif liq.cascade_risk == "MEDIUM":
            liq_score = -0.4
        else:
            liq_score = 0.0

        # Weighted composite
        composite = (
            0.40 * basis_score
            + 0.35 * oi_score
            + 0.25 * liq_score
        )
        composite = float(np.clip(composite, -1.0, 1.0))

        # Classify signal
        if composite >= 0.5:
            signal = LeadingSignal.STRONG_BULLISH
        elif composite >= 0.2:
            signal = LeadingSignal.BULLISH
        elif composite <= -0.5 and basis.basis_pct < -0.15:
            signal = LeadingSignal.INVERSE_SETUP
        elif composite <= -0.5:
            signal = LeadingSignal.STRONG_BEARISH
        elif composite <= -0.2:
            signal = LeadingSignal.BEARISH
        else:
            signal = LeadingSignal.NEUTRAL

        # Hours-ahead estimate (basis is faster, OI is slower)
        if abs(basis_score) > 0.4:
            hours_ahead = 2
        elif abs(oi_score) > 0.4:
            hours_ahead = 5
        else:
            hours_ahead = 8

        # Human-readable explanation
        parts = []
        if abs(basis_score) > 0.3:
            parts.append(f"basis {basis.basis_trend.lower()} ({basis.basis_pct:+.2f}%)")
        if abs(oi_score) > 0.3:
            parts.append(f"OI {oi.trend.lower()} ({oi.oi_change_6h_pct:+.1f}% over 6h)")
        if liq_score < -0.3:
            parts.append(f"liquidation cascade risk {liq.cascade_risk.lower()}")
        explanation = "; ".join(parts) if parts else "no dominant leading signal"

        return composite, signal, hours_ahead, explanation

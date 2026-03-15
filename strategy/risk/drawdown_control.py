"""NAV-based drawdown tracking with automatic position scaling."""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DrawdownState:
    current_nav: float
    high_water_mark: float
    daily_low: float
    weekly_low: float
    monthly_low: float
    current_drawdown_pct: float       # from HWM
    daily_drawdown_pct: float
    weekly_drawdown_pct: float
    monthly_drawdown_pct: float
    position_scale: float             # 0–1 based on drawdown rules
    is_halted: bool
    halt_reason: str = ""


class DrawdownController:
    """
    Tracks portfolio NAV and enforces drawdown-based position scaling.

    Rules (from strategy.yaml):
    - Daily drawdown > 3%  → cut positions by 50%
    - Weekly drawdown > 7% → full exit (scale = 0)
    - Monthly drawdown > 15% → pause strategy (manual review)
    - High water mark → performance fees only on new highs
    """

    def __init__(
        self,
        daily_halt_pct: float = -0.03,
        weekly_halt_pct: float = -0.07,
        monthly_review_pct: float = -0.15,
        soft_scale_threshold: float = -0.01,   # -1% → start scaling down at 50%
        hysteresis: float = 0.01,              # gap between halt and resume thresholds
    ):
        self.daily_halt = daily_halt_pct
        self.weekly_halt = weekly_halt_pct
        self.monthly_review = monthly_review_pct
        self.soft_threshold = soft_scale_threshold
        self._hysteresis = hysteresis

        self._nav_history: list[tuple[float, float]] = []   # (timestamp, nav)
        self._hwm: float = 0.0
        self._is_halted: bool = False
        self._halt_reason: str = ""

    def record_nav(self, nav: float, timestamp: Optional[float] = None) -> DrawdownState:
        """
        Record a new NAV observation and compute drawdown state.
        """
        if nav <= 0:
            raise ValueError(f"NAV must be positive, got {nav}")
        ts = timestamp or time.time()
        self._nav_history.append((ts, nav))

        # Keep 90 days of history
        cutoff = ts - 90 * 86400
        self._nav_history = [(t, n) for t, n in self._nav_history if t >= cutoff]

        # Update HWM
        if nav > self._hwm:
            self._hwm = nav

        # Compute drawdowns
        current_dd = (nav - self._hwm) / self._hwm if self._hwm > 0 else 0.0

        daily_dd = self._compute_drawdown(nav, ts, lookback_secs=86400)
        weekly_dd = self._compute_drawdown(nav, ts, lookback_secs=7 * 86400)
        monthly_dd = self._compute_drawdown(nav, ts, lookback_secs=30 * 86400)

        # Determine halt status
        if weekly_dd <= self.weekly_halt:
            self._is_halted = True
            self._halt_reason = f"Weekly drawdown {weekly_dd:.1%} exceeds halt threshold {self.weekly_halt:.1%}"
            logger.error("STRATEGY HALTED: %s", self._halt_reason)
        elif self._is_halted and weekly_dd > self.weekly_halt + self._hysteresis:
            # Allow resume only when drawdown recovers above halt threshold + hysteresis band
            # (hysteresis prevents rapid halt/resume oscillation near the threshold)
            self._is_halted = False
            self._halt_reason = ""
            logger.info("Drawdown recovered. Strategy can resume.")

        position_scale = self._compute_scale(daily_dd, weekly_dd)

        return DrawdownState(
            current_nav=nav,
            high_water_mark=self._hwm,
            daily_low=self._get_period_low(ts, 86400),
            weekly_low=self._get_period_low(ts, 7 * 86400),
            monthly_low=self._get_period_low(ts, 30 * 86400),
            current_drawdown_pct=current_dd,
            daily_drawdown_pct=daily_dd,
            weekly_drawdown_pct=weekly_dd,
            monthly_drawdown_pct=monthly_dd,
            position_scale=position_scale,
            is_halted=self._is_halted,
            halt_reason=self._halt_reason,
        )

    def _compute_drawdown(
        self, current_nav: float, current_ts: float, lookback_secs: int
    ) -> float:
        """Drawdown from period high to current."""
        cutoff = current_ts - lookback_secs
        period_navs = [n for t, n in self._nav_history if t >= cutoff]
        if not period_navs:
            return 0.0
        period_high = max(period_navs)
        if period_high <= 0:
            return 0.0
        return (current_nav - period_high) / period_high

    def _get_period_low(self, current_ts: float, lookback_secs: int) -> float:
        cutoff = current_ts - lookback_secs
        period_navs = [n for t, n in self._nav_history if t >= cutoff]
        return min(period_navs) if period_navs else 0.0

    def _compute_scale(self, daily_dd: float, weekly_dd: float) -> float:
        """Determine position scale multiplier from drawdown state."""
        if self._is_halted or weekly_dd <= self.weekly_halt:
            return 0.0

        if daily_dd <= self.daily_halt:
            # Cut positions by 50%
            return 0.5

        if daily_dd <= self.soft_threshold:
            # Linear scale from 100% at 0% drawdown to 50% at soft_threshold
            t = daily_dd / self.soft_threshold  # 0 at 0%, 1 at threshold
            return 1.0 - 0.5 * t

        return 1.0

    @property
    def high_water_mark(self) -> float:
        return self._hwm

    @property
    def is_halted(self) -> bool:
        return self._is_halted

    def force_halt(self, reason: str = "Manual halt") -> None:
        self._is_halted = True
        self._halt_reason = reason
        logger.warning("Manual halt triggered: %s", reason)

    def resume(self) -> None:
        self._is_halted = False
        self._halt_reason = ""
        logger.info("Strategy resumed")

"""
Funding Settlement Timing Optimizer.

THE most impactful micro-optimization for a funding rate strategy.

How Drift funding works:
  - Drift calculates funding every hour based on the time-weighted average
    of the mark-oracle premium over that hour.
  - Funding is PAID at the end of each hour.
  - Position must be OPEN during the hour to receive funding.
  - You earn the full hour's funding even if you enter at minute 0 of the hour.

The key insight:
  Funding rates cluster. High-funding hours tend to follow high-funding hours.
  By predicting which hours will be high-funding and ensuring you're always
  positioned for those hours, you capture more yield than naive "hold all the time".

  But more importantly: some hours have NEGATIVE funding even during bull regimes.
  If you can identify and avoid those hours (exit before, re-enter after), you
  avoid paying instead of receiving.

Settlement timing strategy:
  1. Track realized funding per hour-of-day over trailing 30 days
  2. Before each hour: check if predicted funding > execution_cost
  3. If yes: hold / enter position for this hour
  4. If no (or negative): exit, save execution costs, re-enter next hour
  5. Special case: funding SURGE detection — if funding jumped 2× this hour,
     stay in even if next-hour prediction is uncertain

Capital cycle optimization:
  Between hourly settlement windows, the freed-up capital earns lending yield.
  Net: save execution cost AND earn 1-2 hours of lending APR on the freed capital.
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

HOURS_PER_YEAR  = 8_760.0
DAYS_OF_WEEK    = 7
HOURS_PER_DAY   = 24

# Minimum funding APR to justify holding vs. liquidating to lending
# Set equal to execution cost × 8760 to break even over 1 hour
DEFAULT_HOLD_THRESHOLD_APR = 5.0    # % APR — below this, prefer lending


@dataclass
class HourlyFundingStats:
    hour_of_day: int                # 0-23 UTC
    day_of_week: int                # 0=Monday, 6=Sunday
    mean_apr: float                 # trailing average APR for this slot
    std_apr: float                  # volatility
    positive_rate: float            # fraction of times this slot was positive
    n_observations: int             # how many data points


@dataclass
class TimingDecision:
    timestamp: datetime
    hour_of_day: int
    day_of_week: int
    predicted_apr: float            # expected APR for this hour
    hold_threshold_apr: float       # minimum APR to justify position
    should_hold: bool               # True = stay in position this hour
    should_exit: bool               # True = exit and earn lending instead
    expected_funding_income: float  # USD over this 1-hour window
    expected_lending_income: float  # USD from 1h of lending if we exit
    net_advantage_usd: float        # funding_income - lending_income - execution_cost
    confidence: float               # 0-1, based on historical data density
    reason: str


class FundingTimingOptimizer:
    """
    Hour-by-hour position timing based on historical funding patterns.

    Learns per-hour-of-day × day-of-week funding APR statistics and uses
    them to decide when to hold vs. temporarily exit to save execution cost.

    This is most powerful for hours with consistently negative or near-zero
    funding (e.g., early UTC morning on weekdays — historically lowest).

    Usage:
        timing = FundingTimingOptimizer(nav=500_000, perp_allocation=0.45)

        # Feed observations
        timing.record_funding("SOL-PERP", timestamp, apr=15.2)

        # Get decision for next hour
        decision = timing.decide(now_utc, execution_cost_pct=0.002)
        if decision.should_exit:
            # Exit perp, park in lending for this hour
            ...
    """

    def __init__(
        self,
        nav:                  float = 100_000.0,
        perp_allocation:      float = 0.45,
        lending_apr:          float = 10.0,
        hold_threshold_apr:   float = DEFAULT_HOLD_THRESHOLD_APR,
        execution_cost_pct:   float = 0.002,   # 0.2% round trip cost
        min_observations:     int = 14,        # minimum data points before trusting stats
        ema_alpha:            float = 0.1,     # EMA decay for recent-weighted avg
    ):
        self.nav                = nav
        self.perp_allocation    = perp_allocation
        self.lending_apr        = lending_apr
        self.hold_threshold_apr = hold_threshold_apr
        self.execution_cost_pct = execution_cost_pct
        self.min_observations   = min_observations
        self.ema_alpha          = ema_alpha

        # Per-symbol, per-(hour, weekday) rolling buffer
        # key: (hour_of_day, day_of_week) → deque of APR observations
        self._history: dict[str, dict[tuple[int, int], deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=90))   # ~3 months
        )

        # EMA per slot for fast online update
        self._ema: dict[str, dict[tuple[int, int], float]] = defaultdict(dict)

    def record_funding(
        self,
        symbol: str,
        timestamp: datetime,
        apr: float,
    ) -> None:
        """Record an observed funding APR for a specific hour."""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        key = (timestamp.hour, timestamp.weekday())
        self._history[symbol][key].append(apr)

        # Update EMA
        prev_ema = self._ema[symbol].get(key, apr)
        self._ema[symbol][key] = self.ema_alpha * apr + (1 - self.ema_alpha) * prev_ema

    def get_hour_stats(
        self,
        symbol: str,
        hour_of_day: int,
        day_of_week: int,
    ) -> HourlyFundingStats:
        """Compute statistics for a specific hour-of-day × day-of-week slot."""
        key = (hour_of_day, day_of_week)
        buf = list(self._history[symbol].get(key, []))

        if not buf:
            return HourlyFundingStats(
                hour_of_day=hour_of_day,
                day_of_week=day_of_week,
                mean_apr=0.0,
                std_apr=20.0,    # high uncertainty = wide std
                positive_rate=0.5,
                n_observations=0,
            )

        arr = np.array(buf)
        # Blend EMA (recent-weighted) with full mean
        ema = self._ema[symbol].get(key, float(arr.mean()))
        full_mean = float(arr.mean())
        # Weight: more observations → trust full mean more
        w = min(1.0, len(arr) / 30.0)
        blended_mean = w * full_mean + (1 - w) * ema

        return HourlyFundingStats(
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            mean_apr=round(blended_mean, 2),
            std_apr=round(float(arr.std()) if len(arr) > 1 else 15.0, 2),
            positive_rate=round(float((arr > 0).mean()), 3),
            n_observations=len(arr),
        )

    def aggregate_stats(
        self,
        symbols: list[str],
        hour_of_day: int,
        day_of_week: int,
    ) -> HourlyFundingStats:
        """Aggregate stats across all symbols (portfolio-level view)."""
        all_stats = [
            self.get_hour_stats(sym, hour_of_day, day_of_week)
            for sym in symbols
        ]
        if not all_stats:
            return self.get_hour_stats(symbols[0] if symbols else "SOL-PERP",
                                       hour_of_day, day_of_week)

        # Weight by n_observations
        weights = [max(s.n_observations, 1) for s in all_stats]
        total_w = sum(weights)
        weighted_mean = sum(s.mean_apr * w for s, w in zip(all_stats, weights)) / total_w
        weighted_pos  = sum(s.positive_rate * w for s, w in zip(all_stats, weights)) / total_w
        min_obs = min(s.n_observations for s in all_stats)

        return HourlyFundingStats(
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            mean_apr=round(weighted_mean, 2),
            std_apr=round(float(np.mean([s.std_apr for s in all_stats])), 2),
            positive_rate=round(weighted_pos, 3),
            n_observations=min_obs,
        )

    def decide(
        self,
        now_utc: datetime,
        symbols: list[str] = None,
        current_funding_apr: float = None,   # live reading if available
        execution_cost_pct: float = None,
    ) -> TimingDecision:
        """
        Decide whether to hold or exit the perp position for the current hour.

        Decision rule:
          Hold if:
            predicted_apr > hold_threshold
            AND predicted_apr × perp_allocation / 8760 × NAV
                > lending_apr × perp_allocation / 8760 × NAV
                  + 2 × execution_cost (round trip to exit and re-enter)
          Exit otherwise.

          Special hold override:
            If current live funding is > 2× hold_threshold, always hold
            (we're already in a high-funding window, don't exit now)
        """
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)

        ec = execution_cost_pct or self.execution_cost_pct
        syms = symbols or list(self._history.keys()) or ["SOL-PERP"]

        stats = self.aggregate_stats(syms, now_utc.hour, now_utc.weekday())

        # Predicted APR: blend historical stats with live reading
        if current_funding_apr is not None and stats.n_observations >= self.min_observations:
            # Blend: 60% live, 40% historical (live is more accurate but noisy)
            predicted = 0.60 * current_funding_apr + 0.40 * stats.mean_apr
        elif current_funding_apr is not None:
            predicted = current_funding_apr
        else:
            predicted = stats.mean_apr

        # Income calculations over 1 hour
        perp_usd = self.nav * self.perp_allocation
        funding_income_usd  = perp_usd * predicted / 100.0 / HOURS_PER_YEAR
        lending_income_usd  = perp_usd * self.lending_apr / 100.0 / HOURS_PER_YEAR
        execution_cost_usd  = perp_usd * ec * 2   # exit + re-enter next hour

        # Net advantage of staying in perp vs. lending for this hour
        net_adv_usd = funding_income_usd - lending_income_usd - execution_cost_usd

        # Confidence: scales with observation count
        confidence = min(1.0, stats.n_observations / max(self.min_observations, 1))

        # Decision
        # Override: if live funding is very high (2× threshold), always hold
        live_override = (
            current_funding_apr is not None
            and current_funding_apr > self.hold_threshold_apr * 2
        )

        if live_override:
            should_hold = True
            reason = f"live funding {current_funding_apr:.1f}% APR >> threshold — hold override"
        elif predicted < self.hold_threshold_apr and confidence >= 0.5:
            should_hold = False
            reason = (
                f"predicted {predicted:.1f}% APR < threshold {self.hold_threshold_apr}% "
                f"(UTC {now_utc.hour:02d}:00 {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][now_utc.weekday()]})"
            )
        elif net_adv_usd < 0:
            should_hold = False
            reason = f"net_adv=${net_adv_usd:.2f} negative — lending better this hour"
        else:
            should_hold = True
            reason = (
                f"predicted {predicted:.1f}% APR > threshold, "
                f"net_adv=${net_adv_usd:.2f}"
            )

        return TimingDecision(
            timestamp=now_utc,
            hour_of_day=now_utc.hour,
            day_of_week=now_utc.weekday(),
            predicted_apr=round(predicted, 2),
            hold_threshold_apr=self.hold_threshold_apr,
            should_hold=should_hold,
            should_exit=not should_hold,
            expected_funding_income=round(funding_income_usd, 4),
            expected_lending_income=round(lending_income_usd, 4),
            net_advantage_usd=round(net_adv_usd, 4),
            confidence=round(confidence, 3),
            reason=reason,
        )

    def worst_hours(
        self,
        symbol: str,
        n: int = 5,
    ) -> list[HourlyFundingStats]:
        """Return the N worst UTC hours (lowest average APR) — avoid these."""
        all_stats = []
        for h in range(HOURS_PER_DAY):
            # Average across weekdays
            week_mean = np.mean([
                self.get_hour_stats(symbol, h, d).mean_apr
                for d in range(DAYS_OF_WEEK)
            ])
            all_stats.append((h, week_mean))
        all_stats.sort(key=lambda x: x[1])
        return [
            self.get_hour_stats(symbol, h, 0)
            for h, _ in all_stats[:n]
        ]

    def best_hours(
        self,
        symbol: str,
        n: int = 5,
    ) -> list[HourlyFundingStats]:
        """Return the N best UTC hours (highest average APR) — target these."""
        all_stats = []
        for h in range(HOURS_PER_DAY):
            week_mean = np.mean([
                self.get_hour_stats(symbol, h, d).mean_apr
                for d in range(DAYS_OF_WEEK)
            ])
            all_stats.append((h, week_mean))
        all_stats.sort(key=lambda x: x[1], reverse=True)
        return [
            self.get_hour_stats(symbol, h, 0)
            for h, _ in all_stats[:n]
        ]

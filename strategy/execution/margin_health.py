"""
Drift Protocol Margin Health Monitor.

For a delta-neutral strategy, the biggest execution risk isn't market direction —
it's margin health. If the perp position moves against us (even temporarily),
we can get liquidated before the funding income covers the loss.

Drift liquidation mechanics (v2) — VERIFIED FROM DOCS:
  - Health % = 100% × (1 − maintenance_margin_requirement / total_collateral)
  - Liquidation threshold: Health % = 0% (NOT a ratio of 1.05)
  - Maintenance margin ratio: ~3% of notional (NOT 5%)
  - Asset weights: USDC 100%/100%, SOL 80%/90%, ETH/BTC 80%/90%
  - For a $100k short SOL-PERP:
      Required margin = $100k × 5% = $5,000
      If SOL moves +10% against our short: unrealized loss = $10,000
      Health drops from ~2.0 to ~1.5 (still safe)
      If SOL moves +40%: unrealized loss = $40k
      Health = ($60k collateral - $40k loss) / $5k req = 4.0 (still ok)
      But if using 3× leverage: health = ($33k - $40k) / $5k = NEGATIVE → LIQUIDATED

Key risk metric: LEVERAGE-ADJUSTED HEALTH
  At 1× effective leverage (fully collateralized), liquidation requires
  a 95%+ move against us — essentially impossible in delta-neutral.

  But if we're using cross-margin (perp + lending combined), the effective
  collateral is the lending balance, and a large rate spike can cause issues.

Auto-deleveraging thresholds (conservative):
  Health > 2.0:   Normal operation, full position
  Health 1.5-2.0: Watch mode, no new entries
  Health 1.3-1.5: Reduce 25%
  Health 1.1-1.3: Reduce 50%
  Health < 1.1:   Emergency exit all perps

Safe margin buffer:
  Always keep 20% of NAV as unallocated USDC reserve for margin top-ups.
  This buffer absorbs temporary adverse moves without triggering deleveraging.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    HEALTHY    = "HEALTHY"       # > 2.0 — full operation
    WATCH      = "WATCH"         # 1.5–2.0 — no new entries
    CAUTION    = "CAUTION"       # 1.3–1.5 — reduce 25%
    DANGER     = "DANGER"        # 1.1–1.3 — reduce 50%
    CRITICAL   = "CRITICAL"      # < 1.1 — emergency exit


@dataclass
class PositionHealth:
    symbol: str
    notional_usd: float          # position size in USD
    collateral_usd: float        # USDC collateral allocated to this position
    unrealized_pnl: float        # mark-to-market PnL (negative = loss)
    required_margin: float       # Drift maintenance margin (5% of notional)
    health_ratio: float          # (collateral + unrealized_pnl) / required_margin
    status: HealthStatus
    effective_leverage: float    # notional / collateral
    max_adverse_move_pct: float  # % move before liquidation
    recommended_action: str


@dataclass
class PortfolioHealth:
    timestamp: float
    positions: list[PositionHealth]
    total_collateral: float
    total_unrealized_pnl: float
    total_notional: float
    portfolio_health: float       # aggregate health
    status: HealthStatus
    margin_utilization: float    # used / total collateral
    free_collateral: float       # available for top-ups
    deleverage_recommendation: Optional[str]
    reserve_pct: float           # % of NAV unallocated (should stay > 20%)


class MarginHealthMonitor:
    """
    Monitors margin health and recommends deleveraging actions.

    For a delta-neutral strategy, the health monitor is the last line of
    defense before liquidation. It should be checked every block (~400ms on
    Solana), not every 10 minutes.

    Usage:
        monitor = MarginHealthMonitor(nav=500_000)

        # Update with live position data
        monitor.update_position(
            symbol="SOL-PERP",
            notional_usd=100_000,
            collateral_usd=110_000,
            unrealized_pnl=-5_000,
            mark_price=150.0,
            entry_price=145.0,
        )

        health = monitor.portfolio_health()
        if health.status == HealthStatus.DANGER:
            # Reduce positions immediately
            ...
    """

    # CONFIRMED from Drift v2 docs:
    # Health % = 100 × (1 − maintenance_margin / total_collateral)
    # Health = 0% → liquidation
    # Maintenance margin ratio ≈ 3% (not 5%)
    DRIFT_MAINTENANCE_MARGIN_PCT = 0.03   # 3% of notional (confirmed)
    DRIFT_LIQUIDATION_HEALTH_PCT = 0.0    # liquidated at 0% health

    # Thresholds are now in PERCENTAGE terms (0-100), not ratios
    # Recommended operational floors from research:
    #   > 50%: normal operation
    #   30-50%: watch — log, no new entries
    #   20-30%: caution — reduce 25%
    #   10-20%: danger — reduce 50%
    #   < 10%:  critical — emergency exit
    THRESHOLDS = {
        HealthStatus.HEALTHY:  50.0,   # > 50%
        HealthStatus.WATCH:    30.0,   # 30-50%
        HealthStatus.CAUTION:  20.0,   # 20-30%
        HealthStatus.DANGER:   10.0,   # 10-20%
        HealthStatus.CRITICAL:  0.0,   # < 10%
    }

    def __init__(
        self,
        nav: float = 100_000.0,
        maintenance_margin_pct: float = 0.03,  # 3% confirmed from Drift docs
        reserve_target_pct: float = 0.20,      # keep 20% NAV free
    ):
        self.nav = nav
        self.maintenance_margin_pct = maintenance_margin_pct
        self.reserve_target_pct = reserve_target_pct
        self._positions: dict[str, dict] = {}
        self._health_history: list[tuple[float, float]] = []  # (ts, health)

    def update_position(
        self,
        symbol: str,
        notional_usd: float,
        collateral_usd: float,
        unrealized_pnl: float,
        mark_price: float,
        entry_price: float,
    ) -> PositionHealth:
        """Update position data and compute health metrics."""
        self._positions[symbol] = {
            "notional_usd":    notional_usd,
            "collateral_usd":  collateral_usd,
            "unrealized_pnl":  unrealized_pnl,
            "mark_price":      mark_price,
            "entry_price":     entry_price,
            "updated_at":      time.time(),
        }
        return self._compute_position_health(symbol)

    def _compute_position_health(self, symbol: str) -> PositionHealth:
        p = self._positions[symbol]
        notional    = p["notional_usd"]
        collateral  = p["collateral_usd"]
        upnl        = p["unrealized_pnl"]
        mark        = p["mark_price"]
        entry       = p["entry_price"]

        required_margin = notional * self.maintenance_margin_pct
        net_collateral  = collateral + upnl

        # Drift formula: Health % = 100 × (1 - req_margin / total_collateral)
        # Health = 0% → liquidation; Health = 100% → no positions
        if net_collateral > 0:
            health = max(0.0, (1.0 - required_margin / net_collateral) * 100.0)
        else:
            health = 0.0

        effective_leverage = notional / collateral if collateral > 0 else 0.0

        # Maximum adverse move before liquidation (health reaches 0%):
        #   0 = 1 - req_margin / (net_collateral - loss)
        #   loss = net_collateral - required_margin
        #   move_pct = loss / notional
        loss_before_liq = max(0.0, net_collateral - required_margin)
        max_move = loss_before_liq / notional if notional > 0 else 0.0

        # Determine status (health is now in % terms: 0-100)
        status = HealthStatus.CRITICAL
        for h_status, threshold in sorted(self.THRESHOLDS.items(), key=lambda x: x[1], reverse=True):
            if health >= threshold:
                status = h_status
                break

        # Recommended action
        actions = {
            HealthStatus.HEALTHY:  "normal operation",
            HealthStatus.WATCH:    "no new entries, monitor closely",
            HealthStatus.CAUTION:  "reduce position 25%",
            HealthStatus.DANGER:   "reduce position 50% immediately",
            HealthStatus.CRITICAL: "EMERGENCY EXIT — close all positions now",
        }

        return PositionHealth(
            symbol=symbol,
            notional_usd=round(notional, 2),
            collateral_usd=round(collateral, 2),
            unrealized_pnl=round(upnl, 2),
            required_margin=round(required_margin, 2),
            health_ratio=round(health, 4),
            status=status,
            effective_leverage=round(effective_leverage, 3),
            max_adverse_move_pct=round(max_move * 100, 2),
            recommended_action=actions[status],
        )

    def portfolio_health(self) -> PortfolioHealth:
        """Compute aggregate portfolio health across all positions."""
        positions = [
            self._compute_position_health(sym)
            for sym in self._positions
        ]

        total_collateral = sum(p.collateral_usd for p in positions)
        total_upnl       = sum(p.unrealized_pnl for p in positions)
        total_notional   = sum(p.notional_usd for p in positions)
        total_req_margin = sum(p.required_margin for p in positions)

        portfolio_h = (
            (total_collateral + total_upnl) / total_req_margin
            if total_req_margin > 0 else 999.0
        )
        portfolio_h = max(0.0, portfolio_h)

        # Overall status = worst individual status
        if positions:
            statuses = [p.status for p in positions]
            severity = [HealthStatus.HEALTHY, HealthStatus.WATCH,
                        HealthStatus.CAUTION, HealthStatus.DANGER, HealthStatus.CRITICAL]
            overall_status = max(statuses, key=lambda s: severity.index(s))
        else:
            overall_status = HealthStatus.HEALTHY

        margin_util  = total_notional / self.nav if self.nav > 0 else 0.0
        free_collat  = max(0.0, self.nav - total_collateral)
        reserve_pct  = free_collat / self.nav if self.nav > 0 else 0.0

        # Deleverage recommendation
        deleverage = None
        if overall_status == HealthStatus.CAUTION:
            deleverage = "Reduce all perp positions by 25% — margin health approaching warning zone"
        elif overall_status == HealthStatus.DANGER:
            deleverage = "Reduce all perp positions by 50% immediately"
        elif overall_status == HealthStatus.CRITICAL:
            deleverage = "EMERGENCY EXIT — close all perp positions now, move collateral to lending"
        elif reserve_pct < self.reserve_target_pct:
            deleverage = f"Reserve buffer low ({reserve_pct:.1%} < {self.reserve_target_pct:.1%}) — add USDC or reduce perps"

        ts = time.time()
        self._health_history.append((ts, portfolio_h))
        if len(self._health_history) > 10_000:
            self._health_history = self._health_history[-5_000:]

        return PortfolioHealth(
            timestamp=ts,
            positions=positions,
            total_collateral=round(total_collateral, 2),
            total_unrealized_pnl=round(total_upnl, 2),
            total_notional=round(total_notional, 2),
            portfolio_health=round(portfolio_h, 4),
            status=overall_status,
            margin_utilization=round(margin_util, 4),
            free_collateral=round(free_collat, 2),
            deleverage_recommendation=deleverage,
            reserve_pct=round(reserve_pct, 4),
        )

    def deleverage_scale(self) -> float:
        """
        Returns the scale factor to apply to all perp positions.
        1.0 = normal, 0.75 = reduce 25%, 0.5 = reduce 50%, 0.0 = exit all.
        """
        health = self.portfolio_health()
        scales = {
            HealthStatus.HEALTHY:  1.0,
            HealthStatus.WATCH:    1.0,    # hold current, no new
            HealthStatus.CAUTION:  0.75,
            HealthStatus.DANGER:   0.50,
            HealthStatus.CRITICAL: 0.0,
        }
        return scales[health.status]

    def compute_safe_notional(
        self,
        collateral_usd: float,
        target_health_pct: float = 50.0,   # target Health% well above 0% liquidation
    ) -> float:
        """
        Compute the maximum safe notional for a given collateral amount.
        Ensures Health% >= target_health_pct.

        Health% = 100 × (1 - req_margin / collateral)
        target_health_pct/100 = 1 - (notional × maint_pct) / collateral
        → notional = collateral × (1 - target_health_pct/100) / maint_pct
        """
        buffer = 1.0 - target_health_pct / 100.0
        return collateral_usd * buffer / self.maintenance_margin_pct

    def health_trend(self, lookback_n: int = 10) -> str:
        """
        Detect if health is improving or deteriorating.
        Returns "IMPROVING", "STABLE", or "DETERIORATING".
        """
        if len(self._health_history) < lookback_n:
            return "STABLE"
        recent = [h for _, h in self._health_history[-lookback_n:]]
        slope = float(np.polyfit(range(len(recent)), recent, 1)[0])
        if slope > 0.01:
            return "IMPROVING"
        if slope < -0.01:
            return "DETERIORATING"
        return "STABLE"

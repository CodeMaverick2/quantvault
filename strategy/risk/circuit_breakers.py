"""
Multi-layer circuit breaker system for emergency position management.

Layer 1: Funding rate inversion detection
Layer 2: Basis blowout detection
Layer 3: Oracle deviation detection
Layer 4: Cascade risk score threshold
Layer 5: Venue liquidity collapse detection
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    TRIGGERED = "TRIGGERED"
    COOLING_DOWN = "COOLING_DOWN"


@dataclass
class CircuitBreakerEvent:
    trigger_name: str
    triggered_at: float     # unix timestamp
    value: float
    threshold: float
    action_taken: str
    resolved_at: Optional[float] = None

    @property
    def is_active(self) -> bool:
        return self.resolved_at is None

    @property
    def duration_secs(self) -> float:
        end = self.resolved_at or time.time()
        return end - self.triggered_at


@dataclass
class CircuitBreakerConfig:
    # Funding rate cost ceiling (annualized)
    max_negative_funding_apr: float = -0.45      # -45% APR = exit all
    # Perp/spot divergence
    max_basis_pct: float = 0.02                  # 2%
    # Oracle deviation vs. CEX reference
    max_oracle_deviation_pct: float = 0.005      # 0.5%
    # Cascade risk score
    cascade_risk_threshold: float = 0.70
    # Liquidity: current depth / 24h avg
    min_book_depth_ratio: float = 0.30
    # Cooldown after trigger fires
    cooldown_secs: int = 3600  # 1 hour


class CircuitBreaker:
    """
    Stateful circuit breaker that monitors multiple market conditions
    and triggers position reduction/exit when thresholds are breached.
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.config = config or CircuitBreakerConfig()
        self._state: CircuitBreakerState = CircuitBreakerState.NORMAL
        self._events: list[CircuitBreakerEvent] = []
        self._last_trigger_ts: float = 0.0

    @property
    def state(self) -> CircuitBreakerState:
        return self._state

    @property
    def is_triggered(self) -> bool:
        return self._state in (CircuitBreakerState.TRIGGERED, CircuitBreakerState.COOLING_DOWN)

    @property
    def active_events(self) -> list[CircuitBreakerEvent]:
        return [e for e in self._events if e.is_active]

    def check(
        self,
        funding_apr: float,
        basis_pct: float,
        oracle_deviation_pct: float,
        cascade_risk_score: float,
        book_depth_ratio: float,
    ) -> tuple[CircuitBreakerState, list[str]]:
        """
        Run all circuit breaker checks.

        Returns (new_state, list_of_triggered_checks).
        """
        now = time.time()

        # Check cooldown
        if self._state == CircuitBreakerState.COOLING_DOWN:
            if now - self._last_trigger_ts < self.config.cooldown_secs:
                remaining = self.config.cooldown_secs - (now - self._last_trigger_ts)
                logger.debug("Circuit breaker in cooldown: %.0f secs remaining", remaining)
                return self._state, []
            else:
                self._state = CircuitBreakerState.NORMAL
                logger.info("Circuit breaker cooldown expired. Resuming normal operation.")

        triggered_checks: list[str] = []

        # Check 1: Funding rate inversion
        if funding_apr < self.config.max_negative_funding_apr:
            triggered_checks.append(
                f"NEGATIVE_FUNDING: {funding_apr:.1%} APR < threshold {self.config.max_negative_funding_apr:.1%}"
            )
            self._log_event("NEGATIVE_FUNDING", funding_apr, self.config.max_negative_funding_apr)

        # Check 2: Basis blowout
        if abs(basis_pct) > self.config.max_basis_pct:
            triggered_checks.append(
                f"BASIS_BLOWOUT: {abs(basis_pct):.2%} > threshold {self.config.max_basis_pct:.2%}"
            )
            self._log_event("BASIS_BLOWOUT", abs(basis_pct), self.config.max_basis_pct)

        # Check 3: Oracle deviation
        if oracle_deviation_pct > self.config.max_oracle_deviation_pct:
            triggered_checks.append(
                f"ORACLE_DEVIATION: {oracle_deviation_pct:.3%} > threshold {self.config.max_oracle_deviation_pct:.3%}"
            )
            self._log_event("ORACLE_DEVIATION", oracle_deviation_pct, self.config.max_oracle_deviation_pct)

        # Check 4: Cascade risk
        if cascade_risk_score >= self.config.cascade_risk_threshold:
            triggered_checks.append(
                f"CASCADE_RISK: score={cascade_risk_score:.2f} >= threshold {self.config.cascade_risk_threshold:.2f}"
            )
            self._log_event("CASCADE_RISK", cascade_risk_score, self.config.cascade_risk_threshold)

        # Check 5: Liquidity collapse
        if book_depth_ratio < self.config.min_book_depth_ratio:
            triggered_checks.append(
                f"LIQUIDITY_COLLAPSE: depth_ratio={book_depth_ratio:.2f} < threshold {self.config.min_book_depth_ratio:.2f}"
            )
            self._log_event("LIQUIDITY_COLLAPSE", book_depth_ratio, self.config.min_book_depth_ratio)

        if triggered_checks:
            self._state = CircuitBreakerState.TRIGGERED
            self._last_trigger_ts = now
            for msg in triggered_checks:
                logger.warning("CIRCUIT BREAKER TRIGGERED: %s", msg)
        elif self._state == CircuitBreakerState.TRIGGERED:
            self._state = CircuitBreakerState.COOLING_DOWN
            logger.info("Circuit breaker conditions resolved. Entering cooldown.")

        return self._state, triggered_checks

    def resolve(self, event_name: str) -> None:
        """Mark a specific event as resolved."""
        now = time.time()
        for event in self._events:
            if event.trigger_name == event_name and event.is_active:
                event.resolved_at = now
                logger.info("Circuit breaker event resolved: %s", event_name)

    def force_reset(self) -> None:
        """Force reset all circuit breakers (use with caution)."""
        self._state = CircuitBreakerState.NORMAL
        self._last_trigger_ts = 0.0
        for event in self._events:
            if event.is_active:
                event.resolved_at = time.time()
        logger.warning("Circuit breakers force-reset")

    def get_position_multiplier(self) -> float:
        """
        Returns how much to scale positions given current CB state.
        1.0 = normal, 0.5 = halved, 0.0 = exit all.
        """
        if self._state == CircuitBreakerState.TRIGGERED:
            return 0.0
        if self._state == CircuitBreakerState.COOLING_DOWN:
            # Linearly scale back up during cooldown
            elapsed = time.time() - self._last_trigger_ts
            fraction = min(1.0, elapsed / self.config.cooldown_secs)
            return fraction * 0.5  # max 50% during cooldown ramp-up
        return 1.0  # NORMAL or WARNING

    def _log_event(self, name: str, value: float, threshold: float) -> None:
        # Only log new events (avoid duplicates in rapid-fire checks)
        existing = next(
            (e for e in self._events if e.trigger_name == name and e.is_active), None
        )
        if existing is None:
            self._events.append(
                CircuitBreakerEvent(
                    trigger_name=name,
                    triggered_at=time.time(),
                    value=value,
                    threshold=threshold,
                    action_taken="EMERGENCY_EXIT",
                )
            )

    def recent_events(self, n: int = 10) -> list[CircuitBreakerEvent]:
        return sorted(self._events, key=lambda e: e.triggered_at, reverse=True)[:n]

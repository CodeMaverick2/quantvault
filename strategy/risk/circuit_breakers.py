"""
Multi-layer circuit breaker system for emergency position management.

Layer 1: Funding rate inversion detection
Layer 2: Basis blowout detection
Layer 3: Oracle deviation detection
Layer 4: Cascade risk score threshold
Layer 5: Venue liquidity collapse detection
"""

import logging
import os
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
    trigger_count: int = 0

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
    # Devnet: set MAX_NEGATIVE_FUNDING_APR=-10.0 (−1000%) to bypass garbage devnet rates
    # Mainnet: keep at -0.45 (−45% APR)
    max_negative_funding_apr: float = float(os.getenv("MAX_NEGATIVE_FUNDING_APR", "-0.45"))
    # Perp/spot divergence
    max_basis_pct: float = 0.02                  # 2%
    # Oracle deviation vs. CEX reference
    max_oracle_deviation_pct: float = 0.005      # 0.5%
    # Cascade risk score
    cascade_risk_threshold: float = 0.70
    # Liquidity: current depth / 24h avg
    min_book_depth_ratio: float = 0.30
    # Cooldown after trigger fires — shorter on mainnet (real oracles recover faster)
    cooldown_secs: int = int(os.getenv("CB_COOLDOWN_SECS", "1800"))  # 30min default
    # Oracle manipulation defense: reject if oracle moves > N sigma in 1 slot
    # Based on Mango Markets exploit lessons — non-negotiable on Solana
    # Mainnet: 5.0 (legitimate moves rarely exceed 5σ); devnet: 8.0+ (noisy feeds)
    oracle_move_sigma_threshold: float = float(os.getenv("ORACLE_SIGMA_THRESHOLD", "5.0"))
    oracle_move_window: int = int(os.getenv("ORACLE_MOVE_WINDOW", "20"))


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
        # Oracle manipulation defense: rolling oracle price history per symbol
        from collections import deque
        self._oracle_history: dict[str, deque] = {}

    @property
    def state(self) -> CircuitBreakerState:
        return self._state

    @property
    def is_triggered(self) -> bool:
        return self._state in (CircuitBreakerState.TRIGGERED, CircuitBreakerState.COOLING_DOWN)

    @property
    def active_events(self) -> list[CircuitBreakerEvent]:
        return [e for e in self._events if e.is_active]

    def check_oracle_manipulation(self, symbol: str, oracle_price: float) -> bool:
        """
        Returns True if oracle price movement exceeds sigma threshold (potential manipulation).

        Defense against Mango Markets-style oracle manipulation on Solana.
        Drift uses Pyth oracles with on-chain price history — we track our own
        rolling window as a secondary validation layer.

        A legitimate oracle move of >3 sigma in a single update is extremely rare
        (p < 0.003 for normal distribution) and should trigger immediate halt.
        """
        from collections import deque
        import numpy as np

        if symbol not in self._oracle_history:
            self._oracle_history[symbol] = deque(maxlen=self.config.oracle_move_window)

        history = self._oracle_history[symbol]
        history.append(oracle_price)

        if len(history) < 3:
            return False  # insufficient history to detect manipulation

        prices = list(history)
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        if len(returns) < 2:
            return False

        import numpy as np
        returns_arr = np.array(returns)
        mean_ret = float(np.mean(returns_arr[:-1]))  # exclude last for comparison
        std_ret = float(np.std(returns_arr[:-1]))

        if std_ret < 1e-10:
            return False

        last_return = returns[-1]
        sigma_move = abs(last_return - mean_ret) / std_ret

        if sigma_move > self.config.oracle_move_sigma_threshold:
            logger.warning(
                "ORACLE_MANIPULATION_SUSPECTED: %s price moved %.2f sigma in 1 slot "
                "(price=%.4f, z=%.1f, threshold=%.1f)",
                symbol, sigma_move, oracle_price, sigma_move,
                self.config.oracle_move_sigma_threshold,
            )
            self._log_event(
                "ORACLE_MANIPULATION",
                sigma_move,
                self.config.oracle_move_sigma_threshold,
            )
            return True
        return False

    def check(
        self,
        funding_apr: float,
        basis_pct: float,
        oracle_deviation_pct: float,
        cascade_risk_score: float,
        book_depth_ratio: float,
        oracle_price: float = 0.0,
        symbol: str = "",
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

        # Check 6: Oracle manipulation (Mango Markets defense)
        if oracle_price > 0 and symbol:
            if self.check_oracle_manipulation(symbol, oracle_price):
                triggered_checks.append(
                    f"ORACLE_MANIPULATION: suspected price manipulation for {symbol}"
                )

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
            # Linearly ramp from 0 → 1.0 over full cooldown period
            elapsed = time.time() - self._last_trigger_ts
            fraction = min(1.0, elapsed / self.config.cooldown_secs)
            return fraction  # reaches full 1.0x at cooldown expiry
        return 1.0  # NORMAL or WARNING

    def _log_event(self, name: str, value: float, threshold: float) -> None:
        # If an active event already exists, increment its count instead of creating a duplicate
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
                    trigger_count=1,
                )
            )
        else:
            existing.trigger_count += 1

    def recent_events(self, n: int = 10) -> list[CircuitBreakerEvent]:
        return sorted(self._events, key=lambda e: e.triggered_at, reverse=True)[:n]

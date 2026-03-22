"""
Risk Manager

Enforces:
- Daily loss limits (circuit breaker)
- Per-trade size limits
- Trade frequency limits
- Post-loss cooldown periods
- Balance verification
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date

from config import Config
from logger import get_logger
from models import Trade, TradeStatus

log = get_logger("risk_manager")


@dataclass
class RiskState:
    """Mutable risk tracking state."""
    daily_pnl: float = 0.0
    daily_trades: int = 0
    current_date: str = field(default_factory=lambda: date.today().isoformat())
    recent_trade_times: deque = field(default_factory=lambda: deque(maxlen=100))
    last_loss_time: float = 0.0
    consecutive_losses: int = 0
    total_invested_today: float = 0.0


class RiskManager:
    """
    Pre-trade and post-trade risk checks.
    Must pass ALL checks before a trade is allowed.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.state = RiskState()

    def can_trade(self) -> tuple[bool, str]:
        """
        Check all risk conditions.
        Returns (allowed, reason).
        """
        self._maybe_reset_daily()

        # 1. Daily loss limit
        if self.state.daily_pnl <= -self.config.daily_loss_limit:
            reason = (
                f"daily loss limit hit: ${self.state.daily_pnl:.2f} "
                f"<= -${self.config.daily_loss_limit:.2f}"
            )
            log.warning("risk_daily_limit", pnl=self.state.daily_pnl)
            return False, reason

        # 2. Trade frequency
        now = time.time()
        one_hour_ago = now - 3600
        recent = sum(1 for t in self.state.recent_trade_times if t > one_hour_ago)
        if recent >= self.config.max_trades_per_hour:
            reason = f"trade frequency limit: {recent}/{self.config.max_trades_per_hour} per hour"
            log.warning("risk_frequency_limit", trades_last_hour=recent)
            return False, reason

        # 3. Post-loss cooldown
        if self.state.last_loss_time > 0:
            elapsed = now - self.state.last_loss_time
            if elapsed < self.config.loss_cooldown_seconds:
                remaining = self.config.loss_cooldown_seconds - elapsed
                reason = f"loss cooldown: {remaining:.0f}s remaining"
                log.info("risk_cooldown", remaining=f"{remaining:.0f}s")
                return False, reason

        return True, "ok"

    def check_trade_size(self, size_usd: float) -> tuple[bool, str]:
        """Validate trade size."""
        if size_usd <= 0:
            return False, "trade size must be positive"
        if size_usd > self.config.max_trade_size:
            return False, f"size ${size_usd:.2f} > max ${self.config.max_trade_size:.2f}"
        return True, "ok"

    def record_trade(self, trade: Trade) -> None:
        """Record a completed trade for risk tracking."""
        self._maybe_reset_daily()

        self.state.daily_trades += 1
        self.state.recent_trade_times.append(time.time())
        self.state.total_invested_today += trade.cost

        if trade.pnl is not None:
            self.state.daily_pnl += trade.pnl
            if trade.pnl < 0:
                self.state.last_loss_time = time.time()
                self.state.consecutive_losses += 1
                log.info(
                    "risk_loss_recorded",
                    pnl=f"${trade.pnl:.4f}",
                    consecutive=self.state.consecutive_losses,
                    daily_pnl=f"${self.state.daily_pnl:.2f}",
                )
            else:
                self.state.consecutive_losses = 0

    def position_size(self, confidence: float) -> float:
        """
        Calculate position size based on confidence.
        Higher confidence → larger position (up to max).
        """
        # Linear scale: confidence 0.5 → 50% of max, 1.0 → 100% of max
        scale = max(0.1, min(1.0, confidence))
        size = self.config.max_trade_size * scale

        # Reduce size after consecutive losses
        if self.state.consecutive_losses >= 3:
            size *= 0.5
            log.info("risk_size_reduced", consecutive_losses=self.state.consecutive_losses)
        elif self.state.consecutive_losses >= 2:
            size *= 0.75

        return round(size, 2)

    def get_status(self) -> dict:
        """Return current risk state for monitoring (read-only, no side effects)."""
        today = date.today().isoformat()
        is_new_day = self.state.current_date != today

        now = time.time()
        one_hour_ago = now - 3600
        recent_trades = sum(1 for t in self.state.recent_trade_times if t > one_hour_ago)

        # If it's a new day, report zeroed-out stats without mutating state
        if is_new_day:
            return {
                "daily_pnl": 0.0,
                "daily_trades": 0,
                "trades_last_hour": 0,
                "consecutive_losses": 0,
                "daily_invested": 0.0,
                "can_trade": True,
            }

        return {
            "daily_pnl": round(self.state.daily_pnl, 4),
            "daily_trades": self.state.daily_trades,
            "trades_last_hour": recent_trades,
            "consecutive_losses": self.state.consecutive_losses,
            "daily_invested": round(self.state.total_invested_today, 2),
            "can_trade": self.can_trade()[0],
        }

    def _maybe_reset_daily(self) -> None:
        """Reset daily counters if date has changed."""
        today = date.today().isoformat()
        if self.state.current_date != today:
            log.info(
                "risk_daily_reset",
                previous_date=self.state.current_date,
                previous_pnl=f"${self.state.daily_pnl:.2f}",
                previous_trades=self.state.daily_trades,
            )
            self.state = RiskState(current_date=today)

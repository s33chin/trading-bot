"""
Data models used throughout the bot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Side(str, Enum):
    UP = "up"
    DOWN = "down"


class TradeAction(str, Enum):
    BUY_UP = "buy_up"
    BUY_DOWN = "buy_down"
    BUY_BOTH = "buy_both"  # arbitrage
    SKIP = "skip"


class TradeStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    FAILED = "failed"


class StrategySource(str, Enum):
    MOMENTUM = "momentum"
    FUSION = "fusion"
    ARBITRAGE = "arbitrage"


# ─────────────────────────────────────────────────────────────
# Market
# ─────────────────────────────────────────────────────────────
@dataclass
class Market:
    """A single Polymarket 15-min BTC Up/Down market."""
    condition_id: str
    question: str
    up_token_id: str
    down_token_id: str
    end_timestamp: float  # unix epoch
    start_timestamp: float  # unix epoch (end - 900 for 15-min)
    slug: str = ""
    neg_risk: bool = True  # BTC 15-min markets are neg_risk

    @property
    def duration_seconds(self) -> float:
        return self.end_timestamp - self.start_timestamp

    @property
    def seconds_remaining(self) -> float:
        return max(0, self.end_timestamp - time.time())

    @property
    def elapsed_ratio(self) -> float:
        """0.0 = just started, 1.0 = expired."""
        if self.duration_seconds <= 0:
            return 1.0
        elapsed = time.time() - self.start_timestamp
        return min(1.0, max(0.0, elapsed / self.duration_seconds))

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.end_timestamp

    @property
    def is_active(self) -> bool:
        return self.start_timestamp <= time.time() < self.end_timestamp


# ─────────────────────────────────────────────────────────────
# Order Book
# ─────────────────────────────────────────────────────────────
@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class OrderBook:
    """Simplified order book for one side (UP or DOWN)."""
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    def fill_price(self, size: float) -> Optional[float]:
        """
        Walk the ask book to find the average fill price for `size` shares.
        Returns None if insufficient liquidity.
        """
        remaining = size
        total_cost = 0.0
        for level in self.asks:
            take = min(remaining, level.size)
            total_cost += take * level.price
            remaining -= take
            if remaining <= 0:
                return total_cost / size
        return None  # not enough liquidity


# ─────────────────────────────────────────────────────────────
# BTC Price Data
# ─────────────────────────────────────────────────────────────
@dataclass
class BTCPrice:
    """A single BTC price observation."""
    price: float
    timestamp: float  # unix epoch
    source: str = "binance"
    volume: float = 0.0


@dataclass
class BTCWindow:
    """
    Price data within a 15-minute window.
    The window_open_price is the BTC price at the start of the Polymarket window.
    """
    window_open_price: Optional[float] = None
    window_open_time: Optional[float] = None
    current_price: Optional[float] = None
    current_time: Optional[float] = None
    prices: list[BTCPrice] = field(default_factory=list)

    @property
    def delta(self) -> Optional[float]:
        """Price change as a fraction (e.g., 0.001 = 0.1%)."""
        if self.window_open_price is not None and self.current_price is not None and self.window_open_price != 0:
            return (self.current_price - self.window_open_price) / self.window_open_price
        return None

    @property
    def delta_pct(self) -> Optional[float]:
        """Price change as a percentage."""
        d = self.delta
        return d * 100 if d is not None else None

    @property
    def direction(self) -> Optional[Side]:
        d = self.delta
        if d is None:
            return None
        return Side.UP if d >= 0 else Side.DOWN

    @property
    def volatility(self) -> float:
        """Standard deviation of price returns within the window."""
        if len(self.prices) < 3:
            return 0.0
        returns = []
        for i in range(1, len(self.prices)):
            if self.prices[i - 1].price > 0:
                r = (self.prices[i].price - self.prices[i - 1].price) / self.prices[i - 1].price
                returns.append(r)
        if not returns:
            return 0.0
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        return variance ** 0.5


# ─────────────────────────────────────────────────────────────
# Signals
# ─────────────────────────────────────────────────────────────
@dataclass
class Signal:
    """A trading signal from a single source."""
    name: str
    direction: Side
    confidence: float  # 0.0 to 1.0
    source: StrategySource
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Signal({self.name}: {self.direction.value} @ {self.confidence:.2%})"


@dataclass
class FusedSignal:
    """Combined signal from multiple sources."""
    action: TradeAction
    direction: Optional[Side]
    confidence: float
    signals: list[Signal] = field(default_factory=list)
    reason: str = ""

    def __str__(self) -> str:
        return f"FusedSignal({self.action.value}: {self.confidence:.2%} — {self.reason})"


# ─────────────────────────────────────────────────────────────
# Trades
# ─────────────────────────────────────────────────────────────
@dataclass
class Trade:
    """A completed or pending trade."""
    id: str
    market: Market
    action: TradeAction
    side: Side
    token_id: str
    price: float
    size: float  # USD amount
    shares: float
    status: TradeStatus = TradeStatus.PENDING
    strategy: StrategySource = StrategySource.MOMENTUM
    order_id: Optional[str] = None
    fill_price: Optional[float] = None
    pnl: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

    @property
    def cost(self) -> float:
        return self.price * self.shares

    @property
    def potential_profit(self) -> Optional[float]:
        if self.fill_price:
            return (1.0 - self.fill_price) * self.shares
        return (1.0 - self.price) * self.shares


# ─────────────────────────────────────────────────────────────
# Performance
# ─────────────────────────────────────────────────────────────
@dataclass
class DailyStats:
    """Aggregated daily performance."""
    date: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    total_invested: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades

    @property
    def roi(self) -> float:
        if self.total_invested == 0:
            return 0.0
        return self.total_pnl / self.total_invested

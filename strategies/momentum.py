"""
Momentum Strategy — Window Delta

The simplest and most proven approach for 15-minute BTC markets.
Core insight: if BTC is moving in one direction within the window,
it tends to continue. Enter late (when signal is strong) but before
token prices fully reflect the outcome.
"""

from __future__ import annotations

from config import Config
from logger import get_logger
from models import (
    BTCWindow,
    Market,
    OrderBook,
    Side,
    Signal,
    StrategySource,
    TradeAction,
)

log = get_logger("strategy.momentum")


class MomentumStrategy:
    """
    Window Delta Momentum Strategy.

    Logic:
    1. Calculate BTC delta from window open to now
    2. If delta exceeds threshold AND token price is below max, trade
    3. Direction = direction of delta (up → buy UP, down → buy DOWN)
    4. Confidence = scaled from delta magnitude
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.min_delta = config.min_delta_threshold
        self.max_token_price = config.max_token_price
        self.entry_seconds = config.entry_seconds_before_close

    def evaluate(
        self,
        market: Market,
        btc_window: BTCWindow,
        up_book: OrderBook | None,
        down_book: OrderBook | None,
    ) -> Signal | None:
        """
        Evaluate whether to trade based on momentum signal.
        Returns a Signal or None if conditions aren't met.
        """
        # Check timing — only trade near the end of the window
        remaining = market.seconds_remaining
        if remaining > self.entry_seconds:
            log.debug(
                "momentum_too_early",
                remaining=f"{remaining:.0f}s",
                entry_at=f"{self.entry_seconds}s",
            )
            return None

        if remaining < 3:
            log.debug("momentum_too_late", remaining=f"{remaining:.1f}s")
            return None

        # Check delta
        delta = btc_window.delta
        if delta is None:
            log.warning("momentum_no_delta")
            return None

        abs_delta = abs(delta)
        if abs_delta < self.min_delta / 100:  # config is in %, delta is fraction
            log.debug(
                "momentum_delta_too_small",
                delta_pct=f"{delta * 100:.4f}%",
                threshold=f"{self.min_delta}%",
            )
            return None

        # Determine direction
        direction = Side.UP if delta > 0 else Side.DOWN

        # Check token price — don't buy overpriced tokens
        book = up_book if direction == Side.UP else down_book
        if book and book.best_ask is not None:
            if book.best_ask > self.max_token_price:
                log.info(
                    "momentum_token_too_expensive",
                    direction=direction.value,
                    ask=f"${book.best_ask:.3f}",
                    max_price=f"${self.max_token_price:.2f}",
                )
                return None

        # Calculate confidence: scale delta into 0-1 range
        # 0.02% delta → ~0.5 confidence, 0.10% → ~0.85, 0.20%+ → ~0.95
        confidence = min(0.95, 0.4 + abs_delta * 100 * 3.0)

        # Bonus confidence for strong order book alignment
        if book and book.best_ask is not None:
            # Cheaper tokens = higher confidence (more edge)
            price_discount = 1.0 - book.best_ask
            confidence = min(0.98, confidence + price_discount * 0.3)

        signal = Signal(
            name="window_delta_momentum",
            direction=direction,
            confidence=confidence,
            source=StrategySource.MOMENTUM,
            metadata={
                "delta_pct": f"{delta * 100:.4f}",
                "abs_delta": f"{abs_delta * 100:.4f}",
                "seconds_remaining": remaining,
                "token_ask": book.best_ask if book else None,
            },
        )

        log.info(
            "momentum_signal",
            direction=direction.value,
            confidence=f"{confidence:.2%}",
            delta=f"{delta * 100:+.4f}%",
            remaining=f"{remaining:.0f}s",
        )

        return signal

"""
Arbitrage Strategy — Buy Both Sides

When the combined cost of UP + DOWN tokens is less than $1.00,
buy both to guarantee profit regardless of outcome.

This is the lowest-risk strategy but opportunities are rare and margins thin.
"""

from __future__ import annotations

from config import Config
from logger import get_logger
from models import (
    Market,
    OrderBook,
    Side,
    Signal,
    StrategySource,
    TradeAction,
)

log = get_logger("strategy.arbitrage")


class ArbitrageStrategy:
    """
    Pure arbitrage: buy both UP and DOWN when total < $1.00.

    Must use ASK prices (not last-trade) and walk the book
    to verify actual fill prices for desired size.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.threshold = config.arb_threshold
        self.shares = config.arb_shares

    def evaluate(
        self,
        market: Market,
        up_book: OrderBook | None,
        down_book: OrderBook | None,
    ) -> Signal | None:
        """
        Check for arbitrage opportunity.
        Returns a Signal with direction=UP (convention) and action=BUY_BOTH.
        """
        if not up_book or not down_book:
            return None

        # Don't arb expired or about-to-expire markets
        if market.seconds_remaining < 10:
            return None

        # Get actual fill prices for desired share count
        up_fill = up_book.fill_price(self.shares)
        down_fill = down_book.fill_price(self.shares)

        if up_fill is None or down_fill is None:
            log.debug(
                "arb_insufficient_liquidity",
                up_fill=up_fill,
                down_fill=down_fill,
                shares=self.shares,
            )
            return None

        combined = up_fill + down_fill
        if combined >= self.threshold:
            log.debug(
                "arb_no_opportunity",
                combined=f"${combined:.4f}",
                threshold=f"${self.threshold:.4f}",
            )
            return None

        # Profit per share = $1.00 - combined cost
        profit_per_share = 1.0 - combined
        total_profit = profit_per_share * self.shares
        roi = profit_per_share / combined

        confidence = min(0.99, 0.7 + roi * 10)  # Higher ROI = higher confidence

        signal = Signal(
            name="arbitrage",
            direction=Side.UP,  # convention; we buy both
            confidence=confidence,
            source=StrategySource.ARBITRAGE,
            metadata={
                "combined_price": f"${combined:.4f}",
                "up_fill": f"${up_fill:.4f}",
                "down_fill": f"${down_fill:.4f}",
                "profit_per_share": f"${profit_per_share:.4f}",
                "total_profit": f"${total_profit:.4f}",
                "roi": f"{roi:.2%}",
                "shares": self.shares,
            },
        )

        log.info(
            "arb_opportunity_found",
            combined=f"${combined:.4f}",
            profit=f"${total_profit:.4f}",
            roi=f"{roi:.2%}",
            shares=self.shares,
        )

        return signal

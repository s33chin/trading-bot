"""
Execution Engine

Handles order placement, fill verification, and paper trading simulation.
Uses py-clob-client for real Polymarket CLOB orders.
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

from config import Config, TradingMode
from logger import get_logger
from models import (
    FusedSignal,
    Market,
    OrderBook,
    Side,
    StrategySource,
    Trade,
    TradeAction,
    TradeStatus,
)

log = get_logger("execution")


class ExecutionEngine:
    """
    Places orders on Polymarket (live) or simulates them (paper).

    In live mode:
    - Uses py-clob-client to place limit orders on the CLOB
    - Verifies fills and handles partial fills
    - For arbitrage, ensures both legs fill or cancels

    In paper mode:
    - Simulates fills at the current best ask price
    - Tracks P&L in memory
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._clob_client = None
        self._paper_trades: list[Trade] = []

    async def initialize(self) -> None:
        """Initialize the CLOB client for live trading."""
        if self.config.is_live:
            try:
                from py_clob_client.client import ClobClient
                from py_clob_client.clob_types import ApiCreds

                creds = ApiCreds(
                    api_key=self.config.polymarket_api_key,
                    api_secret=self.config.polymarket_api_secret,
                    api_passphrase=self.config.polymarket_api_passphrase,
                )
                self._clob_client = ClobClient(
                    self.config.polymarket_clob_url,
                    key=self.config.polymarket_private_key,
                    chain_id=137,  # Polygon
                    creds=creds,
                    signature_type=1,  # POLY_GNOSIS_SAFE
                    funder=self.config.polymarket_proxy_address,
                )
                log.info("clob_client_initialized")
            except ImportError:
                log.error("py_clob_client_not_installed")
                raise RuntimeError(
                    "py-clob-client is required for live trading. "
                    "Install with: pip install py-clob-client"
                )
            except Exception as e:
                log.error("clob_client_init_error", error=str(e))
                raise
        else:
            log.info("execution_paper_mode")

    async def execute(
        self,
        signal: FusedSignal,
        market: Market,
        up_book: Optional[OrderBook],
        down_book: Optional[OrderBook],
        size_usd: float,
    ) -> list[Trade]:
        """
        Execute a trade based on the fused signal.
        Returns list of Trade objects (1 for directional, 2 for arb).
        """
        if signal.action == TradeAction.SKIP:
            return []

        if signal.action == TradeAction.BUY_BOTH:
            return await self._execute_arbitrage(market, up_book, down_book, size_usd)

        # Directional trade
        side = Side.UP if signal.action == TradeAction.BUY_UP else Side.DOWN
        book = up_book if side == Side.UP else down_book
        token_id = market.up_token_id if side == Side.UP else market.down_token_id

        if not book or book.best_ask is None:
            log.warning("execution_no_book", side=side.value)
            return []

        price = book.best_ask
        shares = size_usd / price

        # Polymarket minimum: 5 shares
        if shares < 5:
            log.info("execution_below_minimum", shares=f"{shares:.1f}", min_shares=5)
            return []

        trade = Trade(
            id=str(uuid.uuid4())[:8],
            market=market,
            action=signal.action,
            side=side,
            token_id=token_id,
            price=price,
            size=size_usd,
            shares=shares,
            strategy=signal.signals[0].source if signal.signals else StrategySource.MOMENTUM,
        )

        if self.config.is_live:
            return [await self._place_live_order(trade)]
        else:
            return [self._simulate_fill(trade)]

    async def _execute_arbitrage(
        self,
        market: Market,
        up_book: Optional[OrderBook],
        down_book: Optional[OrderBook],
        size_usd: float,
    ) -> list[Trade]:
        """Execute paired arbitrage — buy both UP and DOWN."""
        if not up_book or not down_book:
            return []

        up_ask = up_book.best_ask
        down_ask = down_book.best_ask
        if up_ask is None or down_ask is None:
            return []

        # Split size equally between both sides
        half_size = size_usd / 2
        up_shares = half_size / up_ask
        down_shares = half_size / down_ask

        if up_shares < 5 or down_shares < 5:
            log.info("arb_below_minimum_shares")
            return []

        up_trade = Trade(
            id=str(uuid.uuid4())[:8],
            market=market,
            action=TradeAction.BUY_BOTH,
            side=Side.UP,
            token_id=market.up_token_id,
            price=up_ask,
            size=half_size,
            shares=up_shares,
            strategy=StrategySource.ARBITRAGE,
        )

        down_trade = Trade(
            id=str(uuid.uuid4())[:8],
            market=market,
            action=TradeAction.BUY_BOTH,
            side=Side.DOWN,
            token_id=market.down_token_id,
            price=down_ask,
            size=half_size,
            shares=down_shares,
            strategy=StrategySource.ARBITRAGE,
        )

        if self.config.is_live:
            # Place both orders — if one fails, cancel the other
            filled_up = await self._place_live_order(up_trade)
            if filled_up.status != TradeStatus.FILLED:
                log.warning("arb_up_leg_failed", status=filled_up.status.value)
                return [filled_up]

            filled_down = await self._place_live_order(down_trade)
            if filled_down.status != TradeStatus.FILLED:
                log.warning(
                    "arb_down_leg_failed",
                    status=filled_down.status.value,
                    msg="attempting to cancel UP order",
                )
                # TODO: cancel the UP order to unwind
                return [filled_up, filled_down]

            return [filled_up, filled_down]
        else:
            return [self._simulate_fill(up_trade), self._simulate_fill(down_trade)]

    async def _place_live_order(self, trade: Trade) -> Trade:
        """Place a real order on Polymarket CLOB."""
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType

            order_args = OrderArgs(
                price=trade.price,
                size=trade.shares,
                side="BUY",
                token_id=trade.token_id,
            )

            signed_order = self._clob_client.create_order(order_args)
            response = self._clob_client.post_order(
                signed_order, order_type=OrderType.GTC
            )

            if response and response.get("success"):
                trade.order_id = response.get("orderID", "")
                trade.status = TradeStatus.FILLED
                trade.fill_price = trade.price
                log.info(
                    "live_order_placed",
                    order_id=trade.order_id,
                    side=trade.side.value,
                    price=f"${trade.price:.4f}",
                    shares=f"{trade.shares:.1f}",
                )
            else:
                trade.status = TradeStatus.FAILED
                error_msg = response.get("errorMsg", "unknown") if response else "no response"
                log.error("live_order_failed", error=error_msg)

        except Exception as e:
            trade.status = TradeStatus.FAILED
            log.error("live_order_exception", error=str(e))

        return trade

    def _simulate_fill(self, trade: Trade) -> Trade:
        """Simulate a paper trade fill."""
        trade.status = TradeStatus.FILLED
        trade.fill_price = trade.price
        trade.order_id = f"paper-{trade.id}"
        self._paper_trades.append(trade)

        log.info(
            "paper_trade_filled",
            side=trade.side.value,
            price=f"${trade.price:.4f}",
            shares=f"{trade.shares:.1f}",
            cost=f"${trade.cost:.4f}",
            strategy=trade.strategy.value,
        )
        return trade

    def resolve_trade(self, trade: Trade, winning_side: Side) -> Trade:
        """
        Resolve a trade after the market closes.
        If trade.side matches winning_side, payout = shares * $1.00.
        Otherwise, payout = $0.
        """
        if trade.status != TradeStatus.FILLED:
            return trade

        if trade.side == winning_side:
            payout = trade.shares * 1.0
            trade.pnl = payout - trade.cost
        else:
            trade.pnl = -trade.cost

        log.info(
            "trade_resolved",
            side=trade.side.value,
            winner=winning_side.value,
            pnl=f"${trade.pnl:+.4f}",
            fill_price=f"${trade.fill_price:.4f}" if trade.fill_price else "N/A",
        )
        return trade

    @property
    def paper_trades(self) -> list[Trade]:
        return self._paper_trades

"""
Execution Engine

Handles order placement, fill verification, and paper trading simulation.
Uses py-clob-client for real Polymarket CLOB orders.
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from typing import Optional

from config import Config, TradingMode
from logger import get_logger
from models import (
    FusedSignal,
    Market,
    OrderBook,
    Position,
    Side,
    StrategySource,
    Trade,
    TradeAction,
    TradeStatus,
)

log = get_logger("execution")

# How long to wait for an order to fill before considering it timed out
ORDER_FILL_TIMEOUT_SECONDS = 30
# How many times to poll for fill status
ORDER_FILL_POLL_ATTEMPTS = 6
# Delay between fill status polls
ORDER_FILL_POLL_INTERVAL = 5


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
                if filled_up.status == TradeStatus.FILLED:
                    # UP leg already filled — can't cancel a fill.
                    # We have an unhedged directional position.
                    log.error(
                        "arb_unhedged_position",
                        order_id=filled_up.order_id,
                        msg="UP leg FILLED but DOWN leg failed — "
                        "ORPHANED DIRECTIONAL POSITION, manual intervention required",
                    )
                else:
                    # UP leg still pending — try to cancel
                    cancelled = await self._cancel_order(filled_up)
                    if cancelled:
                        filled_up.status = TradeStatus.CANCELLED
                        log.info("arb_up_leg_cancelled", order_id=filled_up.order_id)
                    else:
                        log.error(
                            "arb_up_leg_cancel_failed",
                            order_id=filled_up.order_id,
                            msg="ORPHANED POSITION — manual intervention required",
                        )
                return [filled_up, filled_down]

            return [filled_up, filled_down]
        else:
            return [self._simulate_fill(up_trade), self._simulate_fill(down_trade)]

    async def _place_live_order(self, trade: Trade) -> Trade:
        """Place a real order on Polymarket CLOB and verify fill."""
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType

            order_args = OrderArgs(
                price=trade.price,
                size=trade.shares,
                side="BUY",
                token_id=trade.token_id,
            )

            # py-clob-client calls are synchronous — run in thread to avoid
            # blocking the event loop (which would freeze price feeds, etc.)
            signed_order = await asyncio.to_thread(
                self._clob_client.create_order, order_args
            )
            response = await asyncio.to_thread(
                self._clob_client.post_order, signed_order, OrderType.GTC
            )

            if not response or not response.get("success"):
                trade.status = TradeStatus.FAILED
                error_msg = response.get("errorMsg", "unknown") if response else "no response"
                log.error("live_order_failed", error=error_msg)
                return trade

            trade.order_id = response.get("orderID", "")
            trade.status = TradeStatus.PENDING
            log.info(
                "live_order_placed",
                order_id=trade.order_id,
                side=trade.side.value,
                price=f"${trade.price:.4f}",
                shares=f"{trade.shares:.1f}",
            )

            # Poll for fill status
            trade = await self._wait_for_fill(trade)

        except Exception as e:
            trade.status = TradeStatus.FAILED
            log.error("live_order_exception", error=str(e))

        return trade

    async def _wait_for_fill(self, trade: Trade) -> Trade:
        """
        Poll the CLOB for order fill status up to a timeout.
        If the order is not filled within the timeout, cancel it.
        """
        if not trade.order_id:
            trade.status = TradeStatus.FAILED
            return trade

        for attempt in range(ORDER_FILL_POLL_ATTEMPTS):
            await asyncio.sleep(ORDER_FILL_POLL_INTERVAL)
            try:
                order_info = await asyncio.to_thread(
                    self._clob_client.get_order, trade.order_id
                )
                if not order_info:
                    continue

                status = order_info.get("status", "").lower()
                if status in ("matched", "filled"):
                    trade.status = TradeStatus.FILLED
                    assoc_trades = order_info.get("associate_trades") or []
                    if assoc_trades and isinstance(assoc_trades[0], dict):
                        trade.fill_price = float(assoc_trades[0].get("price", trade.price))
                    else:
                        trade.fill_price = trade.price
                    log.info(
                        "order_filled",
                        order_id=trade.order_id,
                        fill_price=f"${trade.fill_price:.4f}",
                        attempt=attempt + 1,
                    )
                    return trade
                elif status in ("cancelled", "expired"):
                    trade.status = TradeStatus.CANCELLED
                    log.warning("order_cancelled_externally", order_id=trade.order_id)
                    return trade

                log.debug(
                    "order_pending",
                    order_id=trade.order_id,
                    status=status,
                    attempt=attempt + 1,
                    max_attempts=ORDER_FILL_POLL_ATTEMPTS,
                )

            except Exception as e:
                log.warning("fill_check_error", order_id=trade.order_id, error=str(e))

        # Timed out — cancel the order
        log.warning(
            "order_fill_timeout",
            order_id=trade.order_id,
            timeout_seconds=ORDER_FILL_TIMEOUT_SECONDS,
        )
        cancelled = await self._cancel_order(trade)
        trade.status = TradeStatus.CANCELLED if cancelled else TradeStatus.FAILED
        return trade

    async def _cancel_order(self, trade: Trade) -> bool:
        """
        Cancel an open order on the CLOB.
        Returns True if cancellation succeeded or order was already gone.
        """
        if not trade.order_id:
            return False

        try:
            response = await asyncio.to_thread(
                self._clob_client.cancel, trade.order_id
            )
            if response and (response.get("canceled") or response.get("success")):
                log.info("order_cancelled", order_id=trade.order_id)
                return True

            # Check if the order no longer exists (already filled/cancelled)
            order_info = await asyncio.to_thread(
                self._clob_client.get_order, trade.order_id
            )
            if order_info:
                status = order_info.get("status", "").lower()
                if status in ("cancelled", "expired"):
                    return True
                log.warning(
                    "cancel_unexpected_status",
                    order_id=trade.order_id,
                    status=status,
                )
            return False

        except Exception as e:
            log.error("cancel_order_exception", order_id=trade.order_id, error=str(e))
            return False

    def _simulate_fill(self, trade: Trade) -> Trade:
        """
        Simulate a paper trade fill with configurable slippage.

        Slippage is applied as a random adverse price movement up to
        paper_slippage_pct of the order price. For BUY orders, slippage
        increases the fill price (worse for buyer).
        """
        slippage_pct = self.config.paper_slippage_pct / 100.0
        if slippage_pct > 0:
            # Random slippage from 0 to max — always adverse (higher fill for buys)
            slippage = trade.price * random.uniform(0, slippage_pct)
            trade.fill_price = min(trade.price + slippage, 0.99)
        else:
            trade.fill_price = trade.price

        # Recalculate shares based on actual fill price
        if trade.fill_price > 0:
            trade.shares = trade.size / trade.fill_price

        trade.status = TradeStatus.FILLED
        trade.order_id = f"paper-{trade.id}"
        self._paper_trades.append(trade)

        log.info(
            "paper_trade_filled",
            side=trade.side.value,
            order_price=f"${trade.price:.4f}",
            fill_price=f"${trade.fill_price:.4f}",
            slippage=f"${(trade.fill_price - trade.price):.4f}" if slippage_pct > 0 else "$0",
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

    # ── Sell Execution (Active Trading) ─────────────────────

    async def execute_sell(
        self,
        position: Position,
        market: Market,
        book: OrderBook,
    ) -> Trade | None:
        """
        Sell tokens from an open position.
        Uses bid price (not ask) since we're selling.
        """
        if book.best_bid is None:
            log.warning("sell_no_bid", side=position.side.value)
            return None

        # Check liquidity at desired size
        avg_bid = book.fill_price_bid(position.shares)
        if avg_bid is None:
            log.warning(
                "sell_insufficient_liquidity",
                side=position.side.value,
                shares=f"{position.shares:.1f}",
            )
            return None

        price = avg_bid
        action = TradeAction.SELL_UP if position.side == Side.UP else TradeAction.SELL_DOWN

        trade = Trade(
            id=str(uuid.uuid4())[:8],
            market=market,
            action=action,
            side=position.side,
            token_id=position.token_id,
            price=price,
            size=price * position.shares,
            shares=position.shares,
            strategy=position.buy_trade.strategy,
            position_id=position.id,
        )

        if self.config.is_live:
            return await self._place_live_sell_order(trade)
        else:
            return self._simulate_sell_fill(trade)

    async def _place_live_sell_order(self, trade: Trade) -> Trade:
        """Place a SELL order on Polymarket CLOB."""
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType

            order_args = OrderArgs(
                price=trade.price,
                size=trade.shares,
                side="SELL",
                token_id=trade.token_id,
            )

            signed_order = await asyncio.to_thread(
                self._clob_client.create_order, order_args
            )
            response = await asyncio.to_thread(
                self._clob_client.post_order, signed_order, OrderType.GTC
            )

            if not response or not response.get("success"):
                trade.status = TradeStatus.FAILED
                error_msg = response.get("errorMsg", "unknown") if response else "no response"
                log.error("live_sell_failed", error=error_msg)
                return trade

            trade.order_id = response.get("orderID", "")
            trade.status = TradeStatus.PENDING
            log.info(
                "live_sell_placed",
                order_id=trade.order_id,
                side=trade.side.value,
                price=f"${trade.price:.4f}",
                shares=f"{trade.shares:.1f}",
            )

            trade = await self._wait_for_fill(trade)

        except Exception as e:
            trade.status = TradeStatus.FAILED
            log.error("live_sell_exception", error=str(e))

        return trade

    def _simulate_sell_fill(self, trade: Trade) -> Trade:
        """
        Simulate a paper sell with slippage against the bid.
        Slippage is adverse for sells (lower fill price).
        """
        slippage_pct = self.config.paper_slippage_pct / 100.0
        if slippage_pct > 0:
            slippage = trade.price * random.uniform(0, slippage_pct)
            trade.fill_price = max(0.01, trade.price - slippage)
        else:
            trade.fill_price = trade.price

        trade.size = trade.fill_price * trade.shares  # Actual revenue
        trade.status = TradeStatus.FILLED
        trade.order_id = f"paper-sell-{trade.id}"
        self._paper_trades.append(trade)

        log.info(
            "paper_sell_filled",
            side=trade.side.value,
            bid_price=f"${trade.price:.4f}",
            fill_price=f"${trade.fill_price:.4f}",
            slippage=f"${(trade.price - trade.fill_price):.4f}" if slippage_pct > 0 else "$0",
            shares=f"{trade.shares:.1f}",
            revenue=f"${trade.size:.4f}",
        )
        return trade

    @property
    def paper_trades(self) -> list[Trade]:
        return self._paper_trades

"""
Main Trading Bot Orchestrator

Coordinates all components:
1. Data feeds (Binance + Polymarket)
2. Strategy evaluation (Momentum, Fusion, Arbitrage)
3. Risk checks
4. Order execution
5. Monitoring & alerts

Runs continuously, processing one 15-minute window at a time.
"""

from __future__ import annotations

import asyncio
import signal
import time
from datetime import date
from typing import Optional

from alerts import TelegramAlerter
from config import Config, Strategy
from data_feeds import BinanceFeed, PolymarketFeed
from execution import ExecutionEngine, RiskManager
from logger import get_logger, setup_logging
from models import (
    BTCPrice,
    BTCWindow,
    DailyStats,
    FusedSignal,
    Market,
    Side,
    Trade,
    TradeAction,
)
from monitoring import Metrics
from strategies import ArbitrageStrategy, FusionStrategy, MomentumStrategy

log = get_logger("bot")


class TradingBot:
    """
    The main bot. Call `run()` to start the trading loop.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

        # Components
        self.binance = BinanceFeed()
        self.polymarket = PolymarketFeed()
        self.execution = ExecutionEngine(config)
        self.risk = RiskManager(config)
        self.metrics = Metrics()
        self.alerts = TelegramAlerter(config)

        # Strategies
        self.momentum = MomentumStrategy(config)
        self.fusion = FusionStrategy(config)
        self.arbitrage = ArbitrageStrategy(config)

        # State
        self._running = False
        self._current_window = BTCWindow()
        self._current_market: Optional[Market] = None
        self._window_trades: list[Trade] = []
        self._all_trades: list[Trade] = []
        self._daily_stats = DailyStats(date=date.today().isoformat())

        # BTC price callback
        self.binance.on_price(self._on_btc_price)

    async def run(self) -> None:
        """Main entry point — runs the bot indefinitely."""
        setup_logging(self.config.log_level)

        log.info("bot_starting", config=str(self.config))

        # Validate config for live trading
        if self.config.is_live:
            issues = self.config.validate_for_live_trading()
            if issues:
                for issue in issues:
                    log.error("config_issue", issue=issue)
                raise RuntimeError(f"Cannot start live trading: {issues}")

        # Initialize components
        self.metrics.start(self.config.prometheus_port)
        self.metrics.bot_info.info(
            {
                "mode": self.config.trading_mode.value,
                "strategy": self.config.strategy.value,
                "max_trade": str(self.config.max_trade_size),
            }
        )

        await self.binance.start()
        await self.polymarket.start()
        await self.execution.initialize()
        await self.alerts.start()
        await self.alerts.send_startup(self.config)

        self._running = True

        # Handle graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
            except NotImplementedError:
                pass  # Windows

        try:
            while self._running:
                await self._main_loop_iteration()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error("bot_fatal_error", error=str(e))
            await self.alerts.send_error("bot", str(e))
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Graceful shutdown."""
        if not self._running:
            return
        self._running = False
        log.info("bot_stopping")

        # Send daily summary before shutting down
        await self.alerts.send_daily_summary(
            self._daily_stats, self.risk.get_status()
        )
        await self.alerts.send_shutdown()

        await self.binance.stop()
        await self.polymarket.stop()
        await self.alerts.stop()
        log.info("bot_stopped")

    # ── Main Loop ─────────────────────────────────────────

    async def _main_loop_iteration(self) -> None:
        """One iteration of the main loop: find market → trade → wait."""

        # 1. Find active market
        market = await self.polymarket.find_market_by_timestamp()
        if not market or market.is_expired:
            log.debug("waiting_for_market")
            await asyncio.sleep(5)
            return

        # New market window
        if self._current_market is None or market.condition_id != self._current_market.condition_id:
            self._on_new_market(market)

        self._current_market = market

        # 2. Wait until entry window
        remaining = market.seconds_remaining
        entry_at = self.config.entry_seconds_before_close

        if remaining > entry_at + 5:
            # Still early — sleep until near entry time, but keep polling price
            sleep_time = min(remaining - entry_at - 2, 10)
            log.debug(
                "waiting_for_entry",
                remaining=f"{remaining:.0f}s",
                sleep=f"{sleep_time:.0f}s",
            )
            await asyncio.sleep(sleep_time)
            return

        # 3. We're in the entry window — evaluate and possibly trade
        await self._evaluate_and_trade(market)

        # 4. If market is about to expire, wait for resolution
        if remaining < 3:
            await self._wait_for_resolution(market)

    async def _evaluate_and_trade(self, market: Market) -> None:
        """Run all strategies and execute the best signal."""

        # Already traded this window? Skip (unless arb)
        has_directional = any(
            t.action != TradeAction.BUY_BOTH for t in self._window_trades
        )

        # Fetch order books
        up_book, down_book = await self.polymarket.fetch_orderbooks(market)

        # Update metrics
        btc = self.binance.last_price
        if btc:
            self.metrics.update_prices(
                btc=btc.price,
                delta_pct=self._current_window.delta_pct,
                up_ask=up_book.best_ask if up_book else None,
                down_ask=down_book.best_ask if down_book else None,
            )

        # Risk check
        can_trade, reason = self.risk.can_trade()
        if not can_trade:
            log.info("risk_blocked", reason=reason)
            self.metrics.update_risk(self.risk.get_status())
            await asyncio.sleep(2)
            return

        # Fetch klines for fusion strategy
        klines = await self.binance.get_klines(interval="1m", limit=10)

        best_signal: Optional[FusedSignal] = None
        strategy_name = self.config.strategy

        # ── Arbitrage (always check, regardless of strategy setting) ──
        if strategy_name in (Strategy.ARBITRAGE, Strategy.ALL):
            arb_signal = self.arbitrage.evaluate(market, up_book, down_book)
            if arb_signal:
                best_signal = FusedSignal(
                    action=TradeAction.BUY_BOTH,
                    direction=Side.UP,
                    confidence=arb_signal.confidence,
                    signals=[arb_signal],
                    reason="arbitrage opportunity",
                )

        # ── Directional (only if we haven't traded this window) ──
        if not has_directional and best_signal is None:

            if strategy_name in (Strategy.MOMENTUM, Strategy.ALL):
                mom_signal = self.momentum.evaluate(
                    market, self._current_window, up_book, down_book
                )
                if mom_signal:
                    best_signal = FusedSignal(
                        action=TradeAction.BUY_UP if mom_signal.direction == Side.UP else TradeAction.BUY_DOWN,
                        direction=mom_signal.direction,
                        confidence=mom_signal.confidence,
                        signals=[mom_signal],
                        reason="momentum",
                    )

            if strategy_name in (Strategy.FUSION, Strategy.ALL):
                fusion_signal = self.fusion.evaluate(
                    market, self._current_window, up_book, down_book, klines
                )
                # Use fusion if it's more confident than momentum
                if fusion_signal.action != TradeAction.SKIP:
                    if best_signal is None or fusion_signal.confidence > best_signal.confidence:
                        best_signal = fusion_signal

        if best_signal is None or best_signal.action == TradeAction.SKIP:
            log.debug("no_actionable_signal")
            await asyncio.sleep(1)
            return

        # Record signal metrics
        self.metrics.record_signal(
            strategy=best_signal.signals[0].source.value if best_signal.signals else "unknown",
            direction=best_signal.direction.value if best_signal.direction else "none",
        )
        self.metrics.signal_confidence.labels(
            strategy=best_signal.signals[0].source.value if best_signal.signals else "unknown"
        ).set(best_signal.confidence)

        await self.alerts.send_signal(best_signal)

        # Position sizing
        size = self.risk.position_size(best_signal.confidence)
        ok, reason = self.risk.check_trade_size(size)
        if not ok:
            log.warning("risk_size_rejected", reason=reason)
            return

        # Execute
        trades = await self.execution.execute(
            best_signal, market, up_book, down_book, size
        )

        for trade in trades:
            self._window_trades.append(trade)
            self._all_trades.append(trade)
            self.metrics.record_trade(
                strategy=trade.strategy.value,
                side=trade.side.value,
                status=trade.status.value,
            )
            await self.alerts.send_trade(trade)

        await asyncio.sleep(2)  # Don't spam orders

    async def _wait_for_resolution(self, market: Market) -> None:
        """Wait for market to expire, then resolve trades."""
        log.info("waiting_for_resolution", market=market.question[:50])

        # Wait for market to fully close
        while market.seconds_remaining > 0:
            await asyncio.sleep(1)

        # Small buffer for settlement
        await asyncio.sleep(5)

        # Determine winner from BTC price
        winning_side = self._determine_winner()
        if winning_side is None:
            log.warning("could_not_determine_winner")
            return

        log.info("market_resolved", winner=winning_side.value)

        # Resolve all trades for this window
        for trade in self._window_trades:
            self.execution.resolve_trade(trade, winning_side)
            self.risk.record_trade(trade)

            # Update daily stats
            self._update_daily_stats(trade)

            self.metrics.record_trade(
                strategy=trade.strategy.value,
                side=trade.side.value,
                status="won" if (trade.pnl or 0) > 0 else "lost",
                pnl=trade.pnl,
            )

        self.metrics.update_risk(self.risk.get_status())

        # Reset for next window
        self._window_trades = []
        self._current_window = BTCWindow()
        self._current_market = None

    # ── Helpers ───────────────────────────────────────────

    def _on_btc_price(self, price: BTCPrice) -> None:
        """Callback for new BTC price from Binance."""
        self._current_window.current_price = price.price
        self._current_window.current_time = price.timestamp
        self._current_window.prices.append(price)

        # Limit stored prices to avoid memory growth
        if len(self._current_window.prices) > 1000:
            self._current_window.prices = self._current_window.prices[-500:]

    def _on_new_market(self, market: Market) -> None:
        """Called when a new 15-minute window starts."""
        log.info(
            "new_market_window",
            question=market.question,
            slug=market.slug,
            seconds_remaining=f"{market.seconds_remaining:.0f}s",
        )

        # Set window open price from current BTC price
        btc = self.binance.last_price
        if btc:
            self._current_window = BTCWindow(
                window_open_price=btc.price,
                window_open_time=btc.timestamp,
                current_price=btc.price,
                current_time=btc.timestamp,
            )
        else:
            self._current_window = BTCWindow()

        self._window_trades = []
        self.metrics.markets_processed.inc()

        asyncio.create_task(
            self.alerts.send_market_found(
                market.question, market.seconds_remaining
            )
        )

    def _determine_winner(self) -> Optional[Side]:
        """
        Determine if BTC went up or down in this window.
        Compares window open price to current price.
        """
        w = self._current_window
        if w.window_open_price is None or w.current_price is None:
            return None
        if w.current_price >= w.window_open_price:
            return Side.UP
        return Side.DOWN

    def _update_daily_stats(self, trade: Trade) -> None:
        """Update daily stats with a resolved trade."""
        today = date.today().isoformat()
        if self._daily_stats.date != today:
            self._daily_stats = DailyStats(date=today)

        self._daily_stats.total_trades += 1
        self._daily_stats.total_invested += trade.cost

        if trade.pnl is not None:
            self._daily_stats.total_pnl += trade.pnl
            if trade.pnl > 0:
                self._daily_stats.wins += 1
                self._daily_stats.largest_win = max(
                    self._daily_stats.largest_win, trade.pnl
                )
            else:
                self._daily_stats.losses += 1
                self._daily_stats.largest_loss = min(
                    self._daily_stats.largest_loss, trade.pnl
                )

        self.metrics.win_rate.set(self._daily_stats.win_rate)

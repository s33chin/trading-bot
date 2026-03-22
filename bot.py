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
    Position,
    PositionStatus,
    Side,
    Trade,
    TradeAction,
    TradeStatus,
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
        self._observation_snapshots: list[dict] = []
        # Max snapshots per window: 15min / 5s polling = 180, with headroom
        self._max_observation_snapshots = 250
        # Track fire-and-forget tasks to prevent silent failures
        self._background_tasks: set[asyncio.Task] = set()
        # Active trading: open/closed positions within the current window
        self._open_positions: list[Position] = []
        self._closed_positions: list[Position] = []

        # BTC price callback
        self.binance.on_price(self._on_btc_price)

    async def run(self) -> None:
        """Main entry point — runs the bot indefinitely."""
        setup_logging(self.config.log_level)

        log.info("bot_starting", config=str(self.config))

        # Warn about plaintext credentials
        if self.config.has_plaintext_credentials:
            log.warning(
                "plaintext_credentials_detected",
                msg="Private keys loaded from .env file. "
                "For production, use a secrets manager (AWS Secrets Manager, HashiCorp Vault, etc.)",
            )

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
                pass  # Windows — KeyboardInterrupt handled below

        try:
            while self._running:
                await self._main_loop_iteration()
        except (asyncio.CancelledError, KeyboardInterrupt):
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
        """
        One iteration of the main loop.

        The bot operates in 3 phases per 15-minute window:

        Phase 1 — OBSERVE (0:00 → entry_time - 5s)
            Collect BTC prices, poll order books every 30s,
            track delta trend, monitor for arb opportunities.
            Build up signal history for the fusion strategy.

        Phase 2 — ANALYZE & EXECUTE (entry_time → close - 3s)
            Evaluate all strategies every 2 seconds.
            Execute when confidence threshold is met.
            Uses accumulated observation data for better decisions.

        Phase 3 — RESOLVE (close → close + 5s)
            Wait for market to expire, determine winner, settle P&L.
        """

        # ── If we have a current market, stay with it until resolved ──
        # This prevents the bot from jumping to the next market before
        # resolving trades from the current one.
        if self._current_market is not None:
            remaining = self._current_market.seconds_remaining

            # Phase 3: Market expired → resolve before doing anything else
            if remaining <= 0:
                await self._wait_for_resolution(self._current_market)
                # _wait_for_resolution sets self._current_market = None
                return

            # Phase 2: Entry window
            if remaining < 2:
                # Very close to expiry — go straight to resolution
                await self._wait_for_resolution(self._current_market)
                return

            entry_at = self.config.entry_seconds_before_close
            if remaining <= entry_at + 5:
                await self._evaluate_and_trade(self._current_market)
                return

            # Phase 1: Observe
            await self._observe(self._current_market, remaining, entry_at)
            return

        # ── No current market — find one ──
        market = await self.polymarket.find_market_by_timestamp()
        if not market or market.is_expired:
            log.debug("waiting_for_market")
            await asyncio.sleep(5)
            return

        # Found a new market — initialize
        self._on_new_market(market)
        self._current_market = market
        log.info(
            "locked_onto_market",
            slug=market.slug,
            seconds_remaining=f"{market.seconds_remaining:.0f}s",
            question=market.question[:60],
        )

    async def _observe(self, market: Market, remaining: float, entry_at: float) -> None:
        """
        Phase 1: Continuous observation during the window.
        Collects data that will inform the trade decision later.
        """
        # Poll order books periodically during observation
        # (every 30s in early window, every 10s as we get closer)
        time_until_entry = remaining - entry_at
        if time_until_entry > 120:
            poll_interval = 30.0
        elif time_until_entry > 30:
            poll_interval = 15.0
        else:
            poll_interval = 5.0

        # Fetch order books to track pricing trends
        up_book, down_book = await self.polymarket.fetch_orderbooks(market)

        # Update token price metrics so Grafana shows data all the time
        if up_book or down_book:
            self.metrics.update_prices(
                up_ask=up_book.best_ask if up_book else None,
                down_ask=down_book.best_ask if down_book else None,
            )

        # Check for arbitrage opportunities even during observation
        # (arb can happen at any time, not just near close)
        strategy_name = self.config.strategy
        if strategy_name in (Strategy.ARBITRAGE, Strategy.ALL):
            arb_signal = self.arbitrage.evaluate(market, up_book, down_book)
            if arb_signal:
                log.info(
                    "arb_opportunity_during_observation",
                    combined=arb_signal.metadata.get("combined_price"),
                    profit=arb_signal.metadata.get("total_profit"),
                )
                # Execute arb immediately — these are time-sensitive
                can_trade, reason = self.risk.can_trade()
                if can_trade:
                    fused = FusedSignal(
                        action=TradeAction.BUY_BOTH,
                        direction=Side.UP,
                        confidence=arb_signal.confidence,
                        signals=[arb_signal],
                        reason="arbitrage (observation phase)",
                    )
                    size = self.risk.position_size(arb_signal.confidence)
                    trades = await self.execution.execute(
                        fused, market, up_book, down_book, size
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

        # Store observation snapshot for fusion strategy
        snapshot = {
            "timestamp": time.time(),
            "remaining": remaining,
            "delta": self._current_window.delta,
            "delta_pct": self._current_window.delta_pct,
            "up_ask": up_book.best_ask if up_book else None,
            "down_ask": down_book.best_ask if down_book else None,
            "ob_imbalance": self.polymarket.orderbook_imbalance(),
            "volatility": self._current_window.volatility,
        }
        self._observation_snapshots.append(snapshot)
        if len(self._observation_snapshots) > self._max_observation_snapshots:
            self._observation_snapshots = self._observation_snapshots[-self._max_observation_snapshots:]

        # Push a preliminary confidence estimate to Grafana during observation
        # so the Signal Confidence chart isn't empty for 14+ minutes
        delta = self._current_window.delta
        if delta is not None:
            abs_delta = abs(delta)
            preliminary_conf = min(0.95, 0.3 + abs_delta * 100 * 3.0)
            self.metrics.signal_confidence.labels(strategy="preliminary").set(
                preliminary_conf
            )

        log.debug(
            "observation",
            remaining=f"{remaining:.0f}s",
            delta=f"{(self._current_window.delta_pct or 0):.4f}%",
            up_ask=f"${up_book.best_ask:.3f}" if up_book and up_book.best_ask else "N/A",
            down_ask=f"${down_book.best_ask:.3f}" if down_book and down_book.best_ask else "N/A",
            snapshots=len(self._observation_snapshots),
        )

        self.metrics.update_risk(self.risk.get_status())

        # ── Active trading: monitor open positions for TP/SL ──
        if self._open_positions:
            await self._monitor_positions(market, up_book, down_book)

        # ── Active trading: early entry if confidence is very high ──
        if (
            self.config.active_trading_enabled
            and self.config.early_entry_enabled
            and remaining <= self.config.early_entry_seconds_before_close
            and remaining > entry_at + 5  # Don't overlap with normal entry phase
            and not self._open_positions  # Don't stack early entries
        ):
            await self._try_early_entry(market, up_book, down_book)

        # Sleep until next observation
        sleep_time = min(poll_interval, max(1, remaining - entry_at - 2))
        # Monitor positions more frequently if any are open
        if self._open_positions:
            sleep_time = min(sleep_time, self.config.position_monitor_interval)
        await asyncio.sleep(sleep_time)

    async def _evaluate_and_trade(self, market: Market) -> None:
        """
        Phase 2: Analyze signals and execute if confident.

        This runs repeatedly (every ~2s) during the entry window.
        Uses observation data collected during Phase 1.
        """

        # Already traded this window? Only check arb
        has_directional = any(
            t.action != TradeAction.BUY_BOTH for t in self._window_trades
        )

        # Fetch fresh order books
        up_book, down_book = await self.polymarket.fetch_orderbooks(market)

        # Update all metrics
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
        remaining = market.seconds_remaining

        # ── Arbitrage (always check) ──
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
                if fusion_signal.action != TradeAction.SKIP:
                    if best_signal is None or fusion_signal.confidence > best_signal.confidence:
                        best_signal = fusion_signal

        # ── Boost confidence using observation history ──
        if best_signal and best_signal.direction and self._observation_snapshots:
            best_signal = self._apply_observation_boost(best_signal)

        # ── Force trade at T-5s if we have any signal at all ──
        # Better to trade at low confidence than miss the window entirely
        if best_signal and best_signal.action == TradeAction.SKIP and remaining < 5:
            if best_signal.confidence > 0.3 and best_signal.direction is not None:
                best_signal.action = (
                    TradeAction.BUY_UP
                    if best_signal.direction == Side.UP
                    else TradeAction.BUY_DOWN
                )
                best_signal.reason = f"forced at T-{remaining:.0f}s (confidence {best_signal.confidence:.2%})"
                log.info(
                    "forced_trade_near_close",
                    direction=best_signal.direction.value,
                    confidence=f"{best_signal.confidence:.2%}",
                    remaining=f"{remaining:.0f}s",
                    is_paper=not self.config.is_live,
                )

        # ── Always push signal confidence to Prometheus ──
        # This ensures Grafana shows data even when the bot skips
        if best_signal and best_signal.signals:
            source = best_signal.signals[0].source.value
            self.metrics.signal_confidence.labels(strategy=source).set(
                best_signal.confidence
            )
        elif best_signal:
            self.metrics.signal_confidence.labels(strategy="none").set(
                best_signal.confidence
            )

        if best_signal is None or best_signal.action == TradeAction.SKIP:
            skip_confidence = best_signal.confidence if best_signal else 0.0
            skip_reason = best_signal.reason if best_signal else "no signal produced"

            log.info(
                "window_skip_decision",
                remaining=f"{remaining:.0f}s",
                reason=skip_reason,
                confidence=f"{skip_confidence:.2%}",
                observations=len(self._observation_snapshots),
            )

            # Notify on Telegram about the skip (only once near the end)
            if remaining < 5:
                await self.alerts.send_skip(
                    market=market,
                    reason=skip_reason,
                    btc_window=self._current_window,
                    confidence=skip_confidence if skip_confidence > 0 else None,
                )

            await asyncio.sleep(2)
            return

        # Record signal metrics
        self.metrics.record_signal(
            strategy=best_signal.signals[0].source.value if best_signal.signals else "unknown",
            direction=best_signal.direction.value if best_signal.direction else "none",
        )

        await self.alerts.send_signal(best_signal)

        # Position sizing
        size = self.risk.position_size(best_signal.confidence)
        ok, reason = self.risk.check_trade_size(size)
        if not ok:
            log.warning("risk_size_rejected", reason=reason)
            return

        # Execute
        log.info(
            "executing_trade",
            action=best_signal.action.value,
            confidence=f"{best_signal.confidence:.2%}",
            reason=best_signal.reason,
            observations=len(self._observation_snapshots),
            remaining=f"{remaining:.0f}s",
            size=f"${size:.2f}",
        )

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

            # Active trading: wrap filled directional buys in Position
            if (
                self.config.active_trading_enabled
                and trade.status == TradeStatus.FILLED
                and trade.action not in (TradeAction.BUY_BOTH, TradeAction.SKIP)
                and trade.fill_price is not None
            ):
                position = Position(
                    id=trade.id,
                    buy_trade=trade,
                    side=trade.side,
                    token_id=trade.token_id,
                    entry_price=trade.fill_price,
                    shares=trade.shares,
                    cost=trade.fill_price * trade.shares,
                )
                position.compute_exit_prices(
                    self.config.take_profit_threshold,
                    self.config.stop_loss_threshold,
                )
                self._open_positions.append(position)
                self.risk.add_exposure(position.cost)
                self.metrics.open_positions.set(len(self._open_positions))
                log.info(
                    "position_opened",
                    side=position.side.value,
                    entry=f"${position.entry_price:.4f}",
                    tp=f"${position.take_profit_price:.4f}",
                    sl=f"${position.stop_loss_price:.4f}",
                    shares=f"{position.shares:.1f}",
                )
                await self.alerts.send_position_opened(position)

        # Monitor existing positions for TP/SL
        if self._open_positions:
            await self._monitor_positions(market, up_book, down_book)

        await asyncio.sleep(2)  # Don't spam orders

    def _apply_observation_boost(self, signal: FusedSignal) -> FusedSignal:
        """
        Use observation history to boost or reduce confidence.

        If the delta has been consistently in one direction throughout
        the window, the signal is more reliable. If it's been choppy
        or reversed, reduce confidence.
        """
        if not self._observation_snapshots or not signal.direction:
            return signal

        # Count how many snapshots agree with the signal direction
        agreeing = 0
        total = 0
        for snap in self._observation_snapshots:
            delta = snap.get("delta")
            if delta is None:
                continue
            total += 1
            snap_direction = Side.UP if delta >= 0 else Side.DOWN
            if snap_direction == signal.direction:
                agreeing += 1

        if total == 0:
            return signal

        consistency = agreeing / total

        # Strong consistency → boost confidence
        # Low consistency → reduce confidence
        if consistency > 0.8 and total >= 3:
            boost = min(0.10, (consistency - 0.7) * 0.3)
            signal.confidence = min(0.98, signal.confidence + boost)
            signal.reason += f" +obs_boost({consistency:.0%} consistent over {total} snapshots)"
        elif consistency < 0.4:
            penalty = min(0.15, (0.5 - consistency) * 0.3)
            signal.confidence = max(0.1, signal.confidence - penalty)
            signal.reason += f" -obs_penalty({consistency:.0%} consistent, choppy)"

        return signal

    async def _wait_for_resolution(self, market: Market) -> None:
        """
        Phase 3: Wait for market to expire, then resolve all trades.
        After resolution, clears current market so the next iteration
        searches for the next window.
        """
        trade_count = len(self._window_trades)
        log.info(
            "resolving_market",
            slug=market.slug,
            trades_to_resolve=trade_count,
            remaining=f"{market.seconds_remaining:.1f}s",
        )

        # Wait for market to fully close
        while market.seconds_remaining > 0:
            await asyncio.sleep(0.5)

        # Small buffer for settlement
        await asyncio.sleep(3)

        # Determine winner from BTC price
        winning_side = self._determine_winner()
        if winning_side is None:
            log.warning(
                "could_not_determine_winner",
                open_price=self._current_window.window_open_price,
                current_price=self._current_window.current_price,
            )
            await self.alerts.send_resolution_no_winner(market)
        else:
            log.info("market_resolved", winner=winning_side.value, trades=trade_count)

            # Mark remaining open positions as held-to-expiry
            for position in self._open_positions:
                position.status = PositionStatus.HELD_TO_EXPIRY
                position.closed_at = time.time()
                self.risk.remove_exposure(position.cost)

            # Resolve trades that were NOT already closed by a mid-window sell.
            # Trades with pnl already set (from a sell) are skipped.
            for trade in self._window_trades:
                if trade.pnl is not None:
                    # Already settled by a mid-window sell — just record stats
                    self.metrics.record_trade(
                        strategy=trade.strategy.value,
                        side=trade.side.value,
                        status="won" if trade.pnl > 0 else "lost",
                        pnl=trade.pnl,
                    )
                    continue

                self.execution.resolve_trade(trade, winning_side)
                self.risk.record_trade(trade)
                self._update_daily_stats(trade)

                self.metrics.record_trade(
                    strategy=trade.strategy.value,
                    side=trade.side.value,
                    status="won" if (trade.pnl or 0) > 0 else "lost",
                    pnl=trade.pnl,
                )

                log.info(
                    "trade_settled",
                    side=trade.side.value,
                    pnl=f"${trade.pnl:+.4f}" if trade.pnl is not None else "N/A",
                    strategy=trade.strategy.value,
                )

            # Send resolution to Telegram — always, even with 0 trades
            await self.alerts.send_resolution(
                market=market,
                winner=winning_side,
                btc_window=self._current_window,
                trades=self._window_trades,
            )

        self.metrics.update_risk(self.risk.get_status())
        self.metrics.open_positions.set(0)

        # ── Clean slate for next window ──
        # This is critical: setting _current_market = None tells the
        # main loop to search for the next market on the next iteration.
        self._window_trades = []
        self._observation_snapshots = []
        self._open_positions = []
        self._closed_positions = []
        self._current_window = BTCWindow()
        self._current_market = None

        log.info("market_cycle_complete", next_action="searching for next market")

    # ── Helpers ───────────────────────────────────────────

    def _on_btc_price(self, price: BTCPrice) -> None:
        """Callback for new BTC price from Binance."""
        self._current_window.current_price = price.price
        self._current_window.current_time = price.timestamp
        self._current_window.prices.append(price)

        # Push BTC price to Prometheus on EVERY update
        self.metrics.btc_price.set(price.price)
        delta_pct = self._current_window.delta_pct
        if delta_pct is not None:
            self.metrics.btc_delta.set(delta_pct)

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
        self._observation_snapshots = []
        self._open_positions = []
        self._closed_positions = []
        self.metrics.markets_processed.inc()

        self._create_tracked_task(
            self.alerts.send_market_found(
                market.question, market.seconds_remaining
            )
        )

    def _determine_winner(self) -> Optional[Side]:
        """
        Determine if BTC went up or down in this window.
        Compares window open price to current price.
        Returns None if exactly flat (no winner).
        """
        w = self._current_window
        if w.window_open_price is None or w.current_price is None:
            return None
        if w.current_price == w.window_open_price:
            return None  # Exactly flat — no clear winner
        if w.current_price > w.window_open_price:
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

    # ── Active Trading ─────────────────────────────────────

    async def _monitor_positions(
        self,
        market: Market,
        up_book: Optional["OrderBook"] = None,
        down_book: Optional["OrderBook"] = None,
    ) -> None:
        """
        Check all open positions for take-profit or stop-loss triggers.
        Uses pre-fetched order books if available, otherwise fetches fresh ones.
        """
        if not self._open_positions:
            return

        # Don't try to sell too close to expiry — let it resolve at binary price
        if market.seconds_remaining < self.config.min_sell_seconds_before_close:
            return

        if up_book is None or down_book is None:
            up_book, down_book = await self.polymarket.fetch_orderbooks(market)

        for position in list(self._open_positions):
            book = up_book if position.side == Side.UP else down_book
            if not book or book.best_bid is None:
                continue

            current_bid = book.best_bid

            if current_bid >= position.take_profit_price:
                log.info(
                    "take_profit_triggered",
                    side=position.side.value,
                    entry=f"${position.entry_price:.4f}",
                    bid=f"${current_bid:.4f}",
                    tp=f"${position.take_profit_price:.4f}",
                )
                await self._close_position(position, market, book, "take_profit")
            elif current_bid <= position.stop_loss_price:
                log.info(
                    "stop_loss_triggered",
                    side=position.side.value,
                    entry=f"${position.entry_price:.4f}",
                    bid=f"${current_bid:.4f}",
                    sl=f"${position.stop_loss_price:.4f}",
                )
                await self._close_position(position, market, book, "stop_loss")

    async def _close_position(
        self, position: Position, market: Market, book, reason: str
    ) -> None:
        """Sell tokens to close a position (take-profit or stop-loss)."""
        sell_trade = await self.execution.execute_sell(
            position=position,
            market=market,
            book=book,
        )
        if sell_trade and sell_trade.status == TradeStatus.FILLED:
            sell_revenue = sell_trade.fill_price * sell_trade.shares
            position.pnl = sell_revenue - position.cost
            position.status = (
                PositionStatus.CLOSED_TAKE_PROFIT
                if reason == "take_profit"
                else PositionStatus.CLOSED_STOP_LOSS
            )
            position.sell_trade = sell_trade
            position.closed_at = time.time()

            # Move from open to closed
            if position in self._open_positions:
                self._open_positions.remove(position)
            self._closed_positions.append(position)
            self.risk.remove_exposure(position.cost)
            self.metrics.open_positions.set(len(self._open_positions))

            # Record the sell P&L on the original buy trade so resolution skips it
            position.buy_trade.pnl = position.pnl

            # Track in window trades and stats
            self._window_trades.append(sell_trade)
            self._all_trades.append(sell_trade)
            self.risk.record_trade(sell_trade)
            self._update_daily_stats_from_position(position)

            self.metrics.record_trade(
                strategy=sell_trade.strategy.value,
                side=sell_trade.side.value,
                status="won" if position.pnl > 0 else "lost",
                pnl=position.pnl,
            )

            if reason == "take_profit":
                self.metrics.positions_closed_tp.inc()
            else:
                self.metrics.positions_closed_sl.inc()

            log.info(
                "position_closed",
                reason=reason,
                side=position.side.value,
                entry=f"${position.entry_price:.4f}",
                exit=f"${sell_trade.fill_price:.4f}",
                pnl=f"${position.pnl:+.4f}",
                shares=f"{position.shares:.1f}",
            )
            await self.alerts.send_position_closed(position, reason)
        else:
            log.warning(
                "position_close_failed",
                reason=reason,
                side=position.side.value,
                sell_status=sell_trade.status.value if sell_trade else "no_trade",
            )

    async def _try_early_entry(
        self, market: Market, up_book, down_book
    ) -> None:
        """
        Attempt early entry during observation phase when confidence is very high.
        Only triggers when active_trading_enabled and early_entry_enabled are True.
        """
        # Check position limits
        can_open, reason = self.risk.can_open_position(len(self._open_positions))
        if not can_open:
            return

        can_trade, reason = self.risk.can_trade()
        if not can_trade:
            return

        # Evaluate signals
        klines = await self.binance.get_klines(interval="1m", limit=10)
        strategy_name = self.config.strategy
        best_signal: Optional[FusedSignal] = None

        if strategy_name in (Strategy.MOMENTUM, Strategy.ALL):
            # Override entry timing check — we're doing early entry
            mom = self.momentum
            delta = self._current_window.delta
            if delta is not None and abs(delta) >= mom.min_delta / 100:
                direction = Side.UP if delta > 0 else Side.DOWN
                book = up_book if direction == Side.UP else down_book
                if book and book.best_ask and book.best_ask <= mom.max_token_price:
                    confidence = min(0.95, 0.4 + abs(delta) * 100 * 3.0)
                    if confidence >= self.config.early_entry_min_confidence:
                        from models import Signal, StrategySource
                        best_signal = FusedSignal(
                            action=TradeAction.BUY_UP if direction == Side.UP else TradeAction.BUY_DOWN,
                            direction=direction,
                            confidence=confidence,
                            signals=[Signal(
                                name="early_momentum",
                                direction=direction,
                                confidence=confidence,
                                source=StrategySource.MOMENTUM,
                            )],
                            reason=f"early entry: confidence {confidence:.2%}",
                        )

        if strategy_name in (Strategy.FUSION, Strategy.ALL):
            # Use fusion with a higher confidence bar
            fusion_signal = self.fusion.evaluate(
                market, self._current_window, up_book, down_book, klines
            )
            if (
                fusion_signal.action != TradeAction.SKIP
                and fusion_signal.confidence >= self.config.early_entry_min_confidence
            ):
                if best_signal is None or fusion_signal.confidence > best_signal.confidence:
                    best_signal = fusion_signal
                    best_signal.reason = f"early entry (fusion): {best_signal.reason}"

        if best_signal is None or best_signal.action == TradeAction.SKIP:
            return

        # Execute early entry
        size = self.risk.position_size(best_signal.confidence)
        ok, reason = self.risk.check_trade_size(size)
        if not ok:
            return

        remaining = market.seconds_remaining
        log.info(
            "early_entry_executing",
            action=best_signal.action.value,
            confidence=f"{best_signal.confidence:.2%}",
            remaining=f"{remaining:.0f}s",
            size=f"${size:.2f}",
        )

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

            # Wrap in Position for monitoring
            if (
                trade.status == TradeStatus.FILLED
                and trade.action not in (TradeAction.BUY_BOTH, TradeAction.SKIP)
                and trade.fill_price is not None
            ):
                position = Position(
                    id=trade.id,
                    buy_trade=trade,
                    side=trade.side,
                    token_id=trade.token_id,
                    entry_price=trade.fill_price,
                    shares=trade.shares,
                    cost=trade.fill_price * trade.shares,
                )
                position.compute_exit_prices(
                    self.config.take_profit_threshold,
                    self.config.stop_loss_threshold,
                )
                self._open_positions.append(position)
                self.risk.add_exposure(position.cost)
                self.metrics.open_positions.set(len(self._open_positions))
                log.info(
                    "early_position_opened",
                    side=position.side.value,
                    entry=f"${position.entry_price:.4f}",
                    tp=f"${position.take_profit_price:.4f}",
                    sl=f"${position.stop_loss_price:.4f}",
                )
                await self.alerts.send_position_opened(position)

    def _update_daily_stats_from_position(self, position: Position) -> None:
        """Update daily stats from a closed position."""
        today = date.today().isoformat()
        if self._daily_stats.date != today:
            self._daily_stats = DailyStats(date=today)

        self._daily_stats.total_trades += 1
        self._daily_stats.total_invested += position.cost

        if position.pnl is not None:
            self._daily_stats.total_pnl += position.pnl
            if position.pnl > 0:
                self._daily_stats.wins += 1
                self._daily_stats.largest_win = max(
                    self._daily_stats.largest_win, position.pnl
                )
            else:
                self._daily_stats.losses += 1
                self._daily_stats.largest_loss = min(
                    self._daily_stats.largest_loss, position.pnl
                )

        self.metrics.win_rate.set(self._daily_stats.win_rate)

    def _create_tracked_task(self, coro) -> asyncio.Task:
        """Create an asyncio task and track it to prevent silent failures."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._on_task_done)
        return task

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Callback for tracked tasks — log exceptions instead of swallowing them."""
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            log.error("background_task_error", error=str(exc), task=task.get_name())

"""
Telegram alert system.
Sends notifications for every meaningful bot decision:
- New market window found
- Trade decisions (execute or skip, with reasoning)
- Market resolution (UP/DOWN winner, P&L)
- Errors and daily summaries
"""

from __future__ import annotations

import asyncio
from typing import Optional

from config import AlertLevel, Config
from logger import get_logger
from models import BTCWindow, DailyStats, FusedSignal, Market, Side, Trade

log = get_logger("alerts.telegram")


class TelegramAlerter:
    """
    Sends formatted alerts to a Telegram chat.
    Non-blocking — failures are logged but don't interrupt trading.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._bot = None
        self._enabled = config.has_telegram

    async def start(self) -> None:
        if not self._enabled:
            log.info("telegram_disabled", reason="no credentials")
            return
        try:
            from telegram import Bot

            self._bot = Bot(token=self.config.telegram_bot_token)
            me = await self._bot.get_me()
            log.info("telegram_connected", bot_name=me.username)
        except ImportError:
            log.warning("telegram_not_installed", hint="pip install python-telegram-bot")
            self._enabled = False
        except Exception as e:
            log.error("telegram_init_error", error=str(e))
            self._enabled = False

    async def stop(self) -> None:
        self._bot = None

    # ── Trade Decision Notifications ──────────────────────

    async def send_trade(self, trade: Trade) -> None:
        """Send when a trade is executed."""
        if not self._should_send("trade"):
            return

        pnl_str = f"${trade.pnl:+.4f}" if trade.pnl is not None else "pending"
        emoji = "🟢" if (trade.pnl or 0) >= 0 else "🔴"

        msg = (
            f"{emoji} <b>Trade Executed</b>\n"
            f"├ Strategy: <code>{trade.strategy.value}</code>\n"
            f"├ Side: <code>{trade.side.value.upper()}</code>\n"
            f"├ Price: <code>${trade.price:.4f}</code>\n"
            f"├ Shares: <code>{trade.shares:.1f}</code>\n"
            f"├ Cost: <code>${trade.cost:.4f}</code>\n"
            f"└ P&L: <code>{pnl_str}</code>"
        )
        await self._send(msg)

    async def send_signal(self, signal: FusedSignal) -> None:
        """Send when a trade signal is generated (before execution)."""
        if not self._should_send("trade"):
            return

        direction = signal.direction.value.upper() if signal.direction else "N/A"
        msg = (
            f"📊 <b>Signal: {signal.action.value.upper()}</b>\n"
            f"├ Direction: <code>{direction}</code>\n"
            f"├ Confidence: <code>{signal.confidence:.1%}</code>\n"
            f"├ Signals: <code>{len(signal.signals)}</code>\n"
            f"└ Reason: {signal.reason}"
        )
        await self._send(msg)

    async def send_skip(
        self,
        market: Market,
        reason: str,
        btc_window: BTCWindow,
        confidence: Optional[float] = None,
    ) -> None:
        """Send when the bot decides NOT to trade a window."""
        if not self._should_send("trade"):
            return

        delta_str = f"{btc_window.delta_pct:+.4f}%" if btc_window.delta_pct is not None else "N/A"
        conf_str = f"{confidence:.1%}" if confidence is not None else "N/A"

        msg = (
            f"⏭️ <b>Window Skipped</b>\n"
            f"├ Market: <code>{market.slug}</code>\n"
            f"├ BTC Delta: <code>{delta_str}</code>\n"
            f"├ Best Confidence: <code>{conf_str}</code>\n"
            f"└ Reason: {reason}"
        )
        await self._send(msg)

    # ── Market Resolution ─────────────────────────────────

    async def send_resolution(
        self,
        market: Market,
        winner: Side,
        btc_window: BTCWindow,
        trades: list[Trade],
    ) -> None:
        """
        Send when a 15-min market resolves.
        Shows the winner (UP/DOWN), BTC delta, and trade results.
        """
        if not self._should_send("trade"):
            return

        emoji = "🟩" if winner == Side.UP else "🟥"
        delta_str = f"{btc_window.delta_pct:+.4f}%" if btc_window.delta_pct is not None else "N/A"
        open_str = f"${btc_window.window_open_price:,.0f}" if btc_window.window_open_price else "N/A"
        close_str = f"${btc_window.current_price:,.0f}" if btc_window.current_price else "N/A"

        msg = (
            f"{emoji} <b>Market Resolved: {winner.value.upper()}</b>\n"
            f"├ Market: <code>{market.slug}</code>\n"
            f"├ BTC Open: <code>{open_str}</code>\n"
            f"├ BTC Close: <code>{close_str}</code>\n"
            f"├ Delta: <code>{delta_str}</code>\n"
        )

        if trades:
            total_pnl = sum(t.pnl for t in trades if t.pnl is not None)
            wins = sum(1 for t in trades if (t.pnl or 0) > 0)
            losses = len(trades) - wins
            pnl_emoji = "💰" if total_pnl >= 0 else "💸"

            msg += (
                f"├ Trades: <code>{len(trades)}</code> ({wins}W / {losses}L)\n"
                f"└ {pnl_emoji} Window P&L: <code>${total_pnl:+.4f}</code>"
            )
        else:
            msg += f"└ No trades placed this window"

        await self._send(msg)

    async def send_resolution_no_winner(self, market: Market) -> None:
        """Send when the bot can't determine the winner."""
        if not self._should_send("trade"):
            return

        msg = (
            f"⚠️ <b>Market Resolved — Unknown Winner</b>\n"
            f"├ Market: <code>{market.slug}</code>\n"
            f"└ Could not determine BTC direction"
        )
        await self._send(msg)

    # ── Market Lifecycle ──────────────────────────────────

    async def send_market_found(self, question: str, seconds_remaining: float) -> None:
        """Notify when a new market window is found."""
        if not self._should_send("trade"):
            return
        msg = (
            f"🔍 <b>New Market</b>\n"
            f"├ {question}\n"
            f"└ Closes in: <code>{seconds_remaining:.0f}s</code>"
        )
        await self._send(msg)

    # ── System Notifications ──────────────────────────────

    async def send_error(self, component: str, error: str) -> None:
        """Send an error alert."""
        if not self._should_send("error"):
            return

        msg = (
            f"⚠️ <b>Error</b>\n"
            f"├ Component: <code>{component}</code>\n"
            f"└ Error: {error[:200]}"
        )
        await self._send(msg)

    async def send_daily_summary(self, stats: DailyStats, risk_status: dict) -> None:
        """Send end-of-day summary."""
        if not self._should_send("trade"):
            return

        emoji = "📈" if stats.total_pnl >= 0 else "📉"
        msg = (
            f"{emoji} <b>Daily Summary — {stats.date}</b>\n"
            f"├ Trades: <code>{stats.total_trades}</code>\n"
            f"├ Wins: <code>{stats.wins}</code> ({stats.win_rate:.0%})\n"
            f"├ Losses: <code>{stats.losses}</code>\n"
            f"├ P&L: <code>${stats.total_pnl:+.4f}</code>\n"
            f"├ ROI: <code>{stats.roi:+.1%}</code>\n"
            f"├ Best: <code>${stats.largest_win:+.4f}</code>\n"
            f"├ Worst: <code>${stats.largest_loss:+.4f}</code>\n"
            f"└ Invested: <code>${stats.total_invested:.2f}</code>"
        )
        await self._send(msg)

    async def send_startup(self, config: Config) -> None:
        """Send bot startup notification."""
        mode = "🔴 LIVE" if config.is_live else "📝 PAPER"
        msg = (
            f"🤖 <b>Bot Started</b>\n"
            f"├ Mode: <code>{mode}</code>\n"
            f"├ Strategy: <code>{config.strategy.value}</code>\n"
            f"├ Max Trade: <code>${config.max_trade_size:.2f}</code>\n"
            f"├ Daily Limit: <code>${config.daily_loss_limit:.2f}</code>\n"
            f"└ Entry: <code>{config.entry_seconds_before_close}s before close</code>"
        )
        await self._send(msg)

    async def send_shutdown(self, reason: str = "manual") -> None:
        """Send bot shutdown notification."""
        msg = f"🛑 <b>Bot Stopped</b>\n└ Reason: {reason}"
        await self._send(msg)

    # ── Internal ──────────────────────────────────────────

    def _should_send(self, msg_type: str) -> bool:
        if not self._enabled:
            return False
        level = self.config.telegram_alert_level
        if level == AlertLevel.ALL:
            return True
        if level == AlertLevel.TRADES_ONLY and msg_type == "trade":
            return True
        if level == AlertLevel.ERRORS_ONLY and msg_type == "error":
            return True
        return False

    async def _send(self, text: str) -> None:
        if not self._bot:
            return
        try:
            await self._bot.send_message(
                chat_id=self.config.telegram_chat_id,
                text=text,
                parse_mode="HTML",
            )
        except Exception as e:
            log.warning("telegram_send_error", error=str(e))

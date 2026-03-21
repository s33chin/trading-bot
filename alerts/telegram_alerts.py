"""
Telegram alert system.
Sends trade notifications, errors, and daily summaries.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from config import AlertLevel, Config
from logger import get_logger
from models import DailyStats, FusedSignal, Trade

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

    async def send_trade(self, trade: Trade) -> None:
        """Send a trade notification."""
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
        """Send a signal notification (when a trade decision is made)."""
        if not self._should_send("trade"):
            return

        if signal.action.value == "skip":
            return  # Don't alert on skips

        direction = signal.direction.value.upper() if signal.direction else "N/A"
        msg = (
            f"📊 <b>Signal: {signal.action.value.upper()}</b>\n"
            f"├ Direction: <code>{direction}</code>\n"
            f"├ Confidence: <code>{signal.confidence:.1%}</code>\n"
            f"├ Signals: <code>{len(signal.signals)}</code>\n"
            f"└ Reason: {signal.reason}"
        )
        await self._send(msg)

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

"""
Binance BTC price feed via WebSocket with REST fallback.
Provides real-time BTC/USDT prices for the trading engine.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Callable, Optional

import aiohttp

from logger import get_logger
from models import BTCPrice

log = get_logger("binance_feed")


class BinanceFeed:
    """
    Real-time BTC/USDT price stream from Binance.

    Uses WebSocket for low-latency updates with automatic reconnection.
    Falls back to REST polling if WebSocket fails.
    """

    WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"
    REST_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    KLINE_URL = "https://api.binance.com/api/v3/klines"

    def __init__(self) -> None:
        self._callbacks: list[Callable[[BTCPrice], None]] = []
        self._last_price: Optional[BTCPrice] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._ws_connected = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        self._rest_poll_interval = 2.0  # seconds between REST polls

    @property
    def last_price(self) -> Optional[BTCPrice]:
        return self._last_price

    def on_price(self, callback: Callable[[BTCPrice], None]) -> None:
        """Register a callback for new price updates."""
        self._callbacks.append(callback)

    async def start(self) -> None:
        """Start the price feed — WebSocket primary, REST fallback."""
        self._running = True
        self._session = aiohttp.ClientSession()

        # Fetch an initial price via REST immediately so we have data right away
        await self.get_price_rest()

        # Start both loops — REST polls continuously as a safety net,
        # WS provides low-latency updates when connected
        asyncio.create_task(self._ws_loop())
        asyncio.create_task(self._rest_poll_loop())
        log.info("binance_feed_started")

    async def stop(self) -> None:
        """Stop the price feed gracefully."""
        self._running = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        log.info("binance_feed_stopped")

    async def get_price_rest(self) -> Optional[BTCPrice]:
        """Fetch current BTC price via REST (fallback)."""
        try:
            if not self._session or self._session.closed:
                self._session = aiohttp.ClientSession()
            async with self._session.get(self.REST_URL, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price = BTCPrice(
                        price=float(data["price"]),
                        timestamp=time.time(),
                        source="binance_rest",
                    )
                    self._last_price = price
                    return price
                else:
                    log.warning("binance_rest_error", status=resp.status)
                    return None
        except Exception as e:
            log.error("binance_rest_exception", error=str(e))
            return None

    async def get_klines(
        self, interval: str = "1m", limit: int = 15
    ) -> list[dict]:
        """
        Fetch recent kline/candlestick data.
        Returns list of {open, high, low, close, volume, timestamp}.
        """
        try:
            if not self._session or self._session.closed:
                self._session = aiohttp.ClientSession()
            params = {"symbol": "BTCUSDT", "interval": interval, "limit": limit}
            async with self._session.get(
                self.KLINE_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    raw = await resp.json()
                    klines = []
                    for k in raw:
                        klines.append(
                            {
                                "timestamp": k[0] / 1000,
                                "open": float(k[1]),
                                "high": float(k[2]),
                                "low": float(k[3]),
                                "close": float(k[4]),
                                "volume": float(k[5]),
                            }
                        )
                    return klines
                return []
        except Exception as e:
            log.error("binance_klines_error", error=str(e))
            return []

    # ── internal ──────────────────────────────────────────

    async def _rest_poll_loop(self) -> None:
        """
        Continuous REST polling fallback.
        Always runs — provides data even when WebSocket is down.
        Polls more frequently when WS is disconnected.
        """
        while self._running:
            try:
                interval = 10.0 if self._ws_connected else self._rest_poll_interval
                await asyncio.sleep(interval)

                if not self._running:
                    break

                price = await self.get_price_rest()
                if price:
                    # Fire callbacks (same as WS would)
                    for cb in self._callbacks:
                        try:
                            cb(price)
                        except Exception as e:
                            log.error("rest_callback_error", error=str(e))

                    if not self._ws_connected:
                        log.debug(
                            "rest_poll_price",
                            price=f"${price.price:.2f}",
                            ws_connected=False,
                        )
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("rest_poll_error", error=str(e))
                await asyncio.sleep(5)

    async def _ws_loop(self) -> None:
        """Main WebSocket loop with automatic reconnection."""
        while self._running:
            try:
                log.info("binance_ws_connecting")
                async with self._session.ws_connect(
                    self.WS_URL,
                    heartbeat=30,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as ws:
                    self._ws = ws
                    self._ws_connected = True
                    self._reconnect_delay = 1.0
                    log.info("binance_ws_connected")

                    async for msg in ws:
                        if not self._running:
                            break
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            self._handle_message(msg.data)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            log.warning("binance_ws_error", error=str(ws.exception()))
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            break

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(
                    "binance_ws_reconnecting",
                    error=str(e),
                    delay=self._reconnect_delay,
                )

            self._ws_connected = False
            if self._running:
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self._max_reconnect_delay
                )

    def _handle_message(self, raw: str) -> None:
        """Parse a Binance aggTrade message."""
        try:
            data = json.loads(raw)
            price = BTCPrice(
                price=float(data["p"]),
                timestamp=float(data["T"]) / 1000,
                source="binance_ws",
                volume=float(data.get("q", 0)),
            )
            self._last_price = price
            for cb in self._callbacks:
                try:
                    cb(price)
                except Exception as e:
                    log.error("price_callback_error", error=str(e))
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("binance_ws_parse_error", error=str(e))

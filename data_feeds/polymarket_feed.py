"""
Polymarket data feed.
Handles market discovery, order book fetching, and price tracking
for BTC 15-minute Up/Down markets.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import aiohttp

from logger import get_logger
from models import Market, OrderBook, OrderBookLevel

log = get_logger("polymarket_feed")


class PolymarketFeed:
    """
    Fetches live market data from Polymarket's Gamma API and CLOB.

    - Discovers active BTC 15-min markets
    - Fetches order books for UP and DOWN tokens
    - Tracks market lifecycle (active → expired → next)
    """

    GAMMA_URL = "https://gamma-api.polymarket.com"
    CLOB_URL = "https://clob.polymarket.com"

    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None
        self._current_market: Optional[Market] = None
        self._up_book: Optional[OrderBook] = None
        self._down_book: Optional[OrderBook] = None

    @property
    def current_market(self) -> Optional[Market]:
        return self._current_market

    @property
    def up_orderbook(self) -> Optional[OrderBook]:
        return self._up_book

    @property
    def down_orderbook(self) -> Optional[OrderBook]:
        return self._down_book

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        log.info("polymarket_feed_started")

    async def stop(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        log.info("polymarket_feed_stopped")

    # ── Market Discovery ──────────────────────────────────

    async def find_active_market(self) -> Optional[Market]:
        """
        Find the currently active BTC 15-minute Up/Down market.
        Uses the Gamma API to search for active markets.
        """
        try:
            params = {
                "active": "true",
                "closed": "false",
                "limit": 10,
                "tag": "btc",
            }
            async with self._session.get(
                f"{self.GAMMA_URL}/markets",
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    log.warning("gamma_api_error", status=resp.status)
                    return None

                markets = await resp.json()

                # Filter for 15-minute BTC up/down markets
                for m in markets:
                    question = m.get("question", "").lower()
                    slug = m.get("slug", "").lower()

                    is_btc_15m = (
                        ("btc" in question or "bitcoin" in question)
                        and ("15" in question or "15m" in slug or "15min" in slug)
                        and ("up" in question.lower() or "down" in question.lower())
                    )

                    if not is_btc_15m:
                        continue

                    # Extract token IDs from outcomes
                    tokens = m.get("tokens", [])
                    if len(tokens) < 2:
                        continue

                    # Identify UP and DOWN tokens
                    up_token = None
                    down_token = None
                    for t in tokens:
                        outcome = t.get("outcome", "").lower()
                        if "up" in outcome or "yes" in outcome:
                            up_token = t.get("token_id")
                        elif "down" in outcome or "no" in outcome:
                            down_token = t.get("token_id")

                    if not up_token or not down_token:
                        continue

                    end_date = m.get("end_date_iso", "")
                    end_ts = m.get("end_date_timestamp")
                    if not end_ts:
                        continue

                    end_ts = float(end_ts)
                    start_ts = end_ts - 900  # 15 minutes

                    if time.time() >= end_ts:
                        continue  # already expired

                    market = Market(
                        condition_id=m.get("condition_id", ""),
                        question=m.get("question", ""),
                        up_token_id=up_token,
                        down_token_id=down_token,
                        end_timestamp=end_ts,
                        start_timestamp=start_ts,
                        slug=m.get("slug", ""),
                        neg_risk=m.get("neg_risk", True),
                    )
                    self._current_market = market
                    log.info(
                        "market_found",
                        question=market.question,
                        seconds_remaining=market.seconds_remaining,
                        slug=market.slug,
                    )
                    return market

                log.warning("no_active_btc_15m_market_found")
                return None

        except Exception as e:
            log.error("market_discovery_error", error=str(e))
            return None

    async def find_market_by_timestamp(self) -> Optional[Market]:
        """
        Deterministic market lookup based on current time.
        BTC 15-min markets follow timestamps divisible by 900.
        """
        now = time.time()
        window_ts = int(now - (now % 900))
        close_time = window_ts + 900
        slug = f"btc-updown-15m-{window_ts}"

        # Try to fetch this specific market
        try:
            async with self._session.get(
                f"{self.GAMMA_URL}/markets",
                params={"slug": slug},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    markets = await resp.json()
                    if markets:
                        m = markets[0]
                        tokens = m.get("tokens", [])
                        up_token = None
                        down_token = None
                        for t in tokens:
                            outcome = t.get("outcome", "").lower()
                            if "up" in outcome:
                                up_token = t.get("token_id")
                            elif "down" in outcome:
                                down_token = t.get("token_id")

                        if up_token and down_token:
                            market = Market(
                                condition_id=m.get("condition_id", ""),
                                question=m.get("question", ""),
                                up_token_id=up_token,
                                down_token_id=down_token,
                                end_timestamp=float(close_time),
                                start_timestamp=float(window_ts),
                                slug=slug,
                            )
                            self._current_market = market
                            return market
        except Exception as e:
            log.warning("timestamp_market_lookup_failed", slug=slug, error=str(e))

        # Fallback to search
        return await self.find_active_market()

    # ── Order Book ────────────────────────────────────────

    async def fetch_orderbooks(self, market: Optional[Market] = None) -> tuple[Optional[OrderBook], Optional[OrderBook]]:
        """
        Fetch order books for both UP and DOWN tokens concurrently.
        Returns (up_book, down_book).
        """
        m = market or self._current_market
        if not m:
            return None, None

        try:
            # Fetch both concurrently for lower latency
            up_task = self._fetch_single_book(m.up_token_id)
            down_task = self._fetch_single_book(m.down_token_id)
            self._up_book, self._down_book = await asyncio.gather(up_task, down_task)
            return self._up_book, self._down_book

        except Exception as e:
            log.error("orderbook_fetch_error", error=str(e))
            return None, None

    async def _fetch_single_book(self, token_id: str) -> Optional[OrderBook]:
        """Fetch order book for a single token."""
        try:
            async with self._session.get(
                f"{self.CLOB_URL}/book",
                params={"token_id": token_id},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()

                bids = [
                    OrderBookLevel(price=float(b["price"]), size=float(b["size"]))
                    for b in data.get("bids", [])
                ]
                asks = [
                    OrderBookLevel(price=float(a["price"]), size=float(a["size"]))
                    for a in data.get("asks", [])
                ]

                # Sort: bids descending, asks ascending
                bids.sort(key=lambda x: x.price, reverse=True)
                asks.sort(key=lambda x: x.price)

                return OrderBook(bids=bids, asks=asks)

        except Exception as e:
            log.warning("single_book_fetch_error", token_id=token_id[:8], error=str(e))
            return None

    # ── Convenience ───────────────────────────────────────

    def combined_ask_price(self) -> Optional[float]:
        """
        Total cost to buy 1 share of both UP and DOWN.
        If < 1.0, arbitrage opportunity exists.
        """
        if not self._up_book or not self._down_book:
            return None
        up_ask = self._up_book.best_ask
        down_ask = self._down_book.best_ask
        if up_ask is not None and down_ask is not None:
            return up_ask + down_ask
        return None

    def orderbook_imbalance(self) -> Optional[float]:
        """
        Order book imbalance: positive = more UP pressure, negative = more DOWN.
        Calculated as (up_bid_depth - down_bid_depth) / total.
        Range: -1.0 to 1.0.
        """
        if not self._up_book or not self._down_book:
            return None
        up_depth = sum(b.size for b in (self._up_book.bids or []))
        down_depth = sum(b.size for b in (self._down_book.bids or []))
        total = up_depth + down_depth
        if total == 0:
            return 0.0
        return (up_depth - down_depth) / total

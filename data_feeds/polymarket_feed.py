"""
Polymarket data feed.
Handles market discovery, order book fetching, and price tracking
for BTC 15-minute Up/Down markets.

Market discovery uses three approaches in priority order:
1. Slug-based lookup (try multiple timestamp candidates)
2. Gamma /markets search with slug prefix filtering
3. Gamma /events endpoint as final fallback

The slug format is: btc-updown-15m-{UNIX_TIMESTAMP}
The timestamp does NOT always align to clean 900-second boundaries
relative to Unix epoch — Polymarket may use its own schedule.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
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
    - Tracks market lifecycle (active -> expired -> next)
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

    # == Market Discovery (main entry point) ===============

    async def find_active_market(self) -> Optional[Market]:
        """
        Main entry point for market discovery.
        Tries multiple approaches in order of reliability.
        """
        # Approach 1: Try slug-based lookup with multiple timestamp candidates
        market = await self._find_by_slug_candidates()
        if market:
            return market

        # Approach 2: Search Gamma /markets with slug prefix
        market = await self._find_by_slug_prefix_search()
        if market:
            return market

        # Approach 3: Search Gamma /events for crypto 15-min
        market = await self._find_by_events_search()
        if market:
            return market

        log.warning("no_active_btc_15m_market_found", approaches_tried=3)
        return None

    async def find_market_by_timestamp(self) -> Optional[Market]:
        """Alias for find_active_market -- tries all approaches."""
        return await self.find_active_market()

    # == Approach 1: Slug candidates =======================

    async def _find_by_slug_candidates(self) -> Optional[Market]:
        """
        Generate multiple possible slug timestamps and try each.
        Polymarket's 15-min window timestamps may be offset from
        clean 900-second boundaries, so we try several candidates.
        """
        now = time.time()
        now_int = int(now)

        # Generate candidate timestamps:
        # Round down to nearest 900s, then try offsets
        base_ts = now_int - (now_int % 900)
        candidates = set()
        for offset in range(-1800, 2700, 300):
            candidates.add(base_ts + offset)

        # Also try rounding based on UTC minute boundaries
        dt = datetime.fromtimestamp(now, tz=timezone.utc)
        for minute_offset in [-30, -15, 0, 15, 30]:
            target = dt + timedelta(minutes=minute_offset)
            rounded_min = (target.minute // 15) * 15
            candidate_dt = target.replace(minute=rounded_min, second=0, microsecond=0)
            candidates.add(int(candidate_dt.timestamp()))

        # Sort by proximity to now (most likely current window first)
        sorted_candidates = sorted(candidates, key=lambda ts: abs(now - ts))

        # Try up to 10 closest candidates
        for ts in sorted_candidates[:10]:
            slug = f"btc-updown-15m-{ts}"
            market = await self._fetch_market_by_slug(slug)
            if market and not market.is_expired:
                log.info(
                    "market_found_by_slug",
                    slug=slug,
                    seconds_remaining=f"{market.seconds_remaining:.0f}s",
                )
                return market

        log.debug("slug_candidates_exhausted")
        return None

    async def _fetch_market_by_slug(self, slug: str) -> Optional[Market]:
        """Fetch a single market by its exact slug."""
        try:
            async with self._session.get(
                f"{self.GAMMA_URL}/markets",
                params={"slug": slug, "closed": "false"},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    return None

                data = await resp.json()
                markets = data if isinstance(data, list) else [data] if data else []

                for m in markets:
                    parsed = self._parse_gamma_market(m)
                    if parsed:
                        return parsed

        except Exception as e:
            log.debug("slug_lookup_miss", slug=slug, error=str(e))

        return None

    # == Approach 2: Slug prefix search ====================

    async def _find_by_slug_prefix_search(self) -> Optional[Market]:
        """
        Search Gamma API for markets whose slug starts with 'btc-updown-15m'.
        """
        try:
            params = {
                "slug": "btc-updown-15m",
                "closed": "false",
                "limit": 20,
                "order": "id",
                "ascending": "false",  # newest first
            }
            async with self._session.get(
                f"{self.GAMMA_URL}/markets",
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    log.warning("gamma_prefix_search_error", status=resp.status)
                    return None

                data = await resp.json()
                markets_list = data if isinstance(data, list) else []

                log.debug("gamma_prefix_search_results", count=len(markets_list))

                best_market = None
                best_remaining = float("inf")

                for m in markets_list:
                    parsed = self._parse_gamma_market(m)
                    if not parsed:
                        continue
                    if parsed.is_expired:
                        continue
                    if not parsed.is_active:
                        continue
                    if parsed.seconds_remaining < best_remaining:
                        best_remaining = parsed.seconds_remaining
                        best_market = parsed

                if best_market:
                    log.info(
                        "market_found_by_prefix_search",
                        slug=best_market.slug,
                        seconds_remaining=f"{best_market.seconds_remaining:.0f}s",
                        question=best_market.question,
                    )
                    return best_market

        except Exception as e:
            log.error("gamma_prefix_search_exception", error=str(e))

        return None

    # == Approach 3: Events search =========================

    async def _find_by_events_search(self) -> Optional[Market]:
        """
        Search via /events endpoint for crypto-related events,
        then drill into their markets.
        """
        try:
            params = {
                "closed": "false",
                "limit": 100,
                "order": "id",
                "ascending": "false",
            }
            async with self._session.get(
                f"{self.GAMMA_URL}/events",
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    log.warning("gamma_events_error", status=resp.status)
                    return None

                events = await resp.json()
                if not isinstance(events, list):
                    events = []

                log.debug("gamma_events_results", count=len(events))

                for event in events:
                    slug = event.get("slug", "").lower()
                    title = event.get("title", "").lower()

                    # Check if this is a BTC 15-min event
                    is_btc_15m = (
                        ("btc" in slug or "bitcoin" in title or "btc" in title)
                        and ("15m" in slug or "15 min" in title or "15-min" in title)
                    )

                    if not is_btc_15m:
                        continue

                    log.debug("btc_15m_event_found", slug=slug, title=title)

                    # Events contain nested markets
                    event_markets = event.get("markets", [])
                    for m in event_markets:
                        parsed = self._parse_gamma_market(m)
                        if parsed and parsed.is_active and not parsed.is_expired:
                            log.info(
                                "market_found_by_events",
                                slug=parsed.slug,
                                seconds_remaining=f"{parsed.seconds_remaining:.0f}s",
                            )
                            return parsed

        except Exception as e:
            log.error("gamma_events_exception", error=str(e))

        return None

    # == Gamma Market Parser ===============================

    def _parse_gamma_market(self, m: dict) -> Optional[Market]:
        """
        Parse a Gamma API market response into our Market model.

        Gamma API market fields include:
        - condition_id / conditionId
        - question, slug
        - clobTokenIds: JSON string '["id1", "id2"]'
        - tokens: list of {token_id, outcome, ...}
        - outcomes: JSON string '["Up", "Down"]'
        - endDate / end_date_iso: ISO date string
        - active, closed, archived
        - neg_risk: boolean
        """
        if not m:
            return None

        # Skip closed/inactive
        if m.get("closed") is True:
            return None
        # Note: 'active' might be False for just-created markets,
        # so we don't filter on it aggressively here

        slug = m.get("slug", "")
        question = m.get("question", "")

        # == Extract token IDs ==
        up_token = None
        down_token = None

        # Method 1: 'tokens' array (preferred — has outcome labels)
        tokens = m.get("tokens", [])
        if tokens and isinstance(tokens, list):
            for t in tokens:
                if not isinstance(t, dict):
                    continue
                outcome = str(t.get("outcome", "")).lower()
                token_id = t.get("token_id", "")
                if not token_id:
                    continue
                if outcome in ("up", "yes"):
                    up_token = token_id
                elif outcome in ("down", "no"):
                    down_token = token_id

        # Method 2: 'clobTokenIds' + 'outcomes' (fallback)
        if not up_token or not down_token:
            clob_ids_raw = m.get("clobTokenIds", m.get("clob_token_ids", ""))
            outcomes_raw = m.get("outcomes", "")

            try:
                if isinstance(clob_ids_raw, str) and clob_ids_raw:
                    clob_ids = json.loads(clob_ids_raw)
                elif isinstance(clob_ids_raw, list):
                    clob_ids = clob_ids_raw
                else:
                    clob_ids = []
            except (json.JSONDecodeError, TypeError):
                clob_ids = []

            try:
                if isinstance(outcomes_raw, str) and outcomes_raw:
                    outcomes = json.loads(outcomes_raw)
                elif isinstance(outcomes_raw, list):
                    outcomes = outcomes_raw
                else:
                    outcomes = []
            except (json.JSONDecodeError, TypeError):
                outcomes = []

            if isinstance(clob_ids, list) and len(clob_ids) >= 2:
                if isinstance(outcomes, list) and len(outcomes) >= 2:
                    for i, outcome in enumerate(outcomes):
                        ol = str(outcome).lower()
                        if i < len(clob_ids):
                            if ol in ("up", "yes"):
                                up_token = clob_ids[i]
                            elif ol in ("down", "no"):
                                down_token = clob_ids[i]

                # If still missing, assume first=Up, second=Down (common convention)
                if not up_token and not down_token and len(clob_ids) >= 2:
                    up_token = clob_ids[0]
                    down_token = clob_ids[1]
                    log.debug("token_assignment_assumed", slug=slug)

        if not up_token or not down_token:
            log.debug("market_missing_tokens", slug=slug, question=question[:60])
            return None

        # == Extract timestamps ==
        end_ts = self._extract_end_timestamp(m, slug)
        if not end_ts:
            log.debug("market_missing_end_date", slug=slug)
            return None

        start_ts = end_ts - 900  # 15-minute window

        condition_id = m.get("condition_id", "") or m.get("conditionId", "")
        neg_risk = m.get("neg_risk", True)
        if isinstance(neg_risk, str):
            neg_risk = neg_risk.lower() == "true"

        market = Market(
            condition_id=condition_id,
            question=question,
            up_token_id=up_token,
            down_token_id=down_token,
            end_timestamp=end_ts,
            start_timestamp=start_ts,
            slug=slug,
            neg_risk=neg_risk,
        )
        self._current_market = market
        return market

    def _extract_end_timestamp(self, m: dict, slug: str) -> Optional[float]:
        """Try multiple methods to extract the market end timestamp."""

        # Try ISO date fields
        for field in ("end_date_iso", "endDate", "end_date"):
            raw = m.get(field, "")
            if raw and isinstance(raw, str):
                try:
                    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                    return dt.timestamp()
                except (ValueError, AttributeError):
                    continue

        # Try numeric timestamp fields
        for field in ("end_date_timestamp", "endDateTimestamp"):
            raw = m.get(field)
            if raw is not None:
                try:
                    ts = float(raw)
                    if ts > 1e12:  # milliseconds
                        ts /= 1000
                    return ts
                except (ValueError, TypeError):
                    continue

        # Try extracting from slug: btc-updown-15m-{TIMESTAMP}
        if "15m" in slug:
            parts = slug.split("-")
            if len(parts) >= 4:
                try:
                    slug_ts = int(parts[-1])
                    # Slug timestamp is window START; end = start + 900
                    return slug_ts + 900
                except ValueError:
                    pass

        return None

    # == Order Book ========================================

    async def fetch_orderbooks(
        self, market: Optional[Market] = None
    ) -> tuple[Optional[OrderBook], Optional[OrderBook]]:
        """
        Fetch order books for both UP and DOWN tokens concurrently.
        Returns (up_book, down_book).
        """
        m = market or self._current_market
        if not m:
            return None, None

        try:
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
                    log.warning("book_http_error", status=resp.status, token=token_id[:16])
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

                bids.sort(key=lambda x: x.price, reverse=True)
                asks.sort(key=lambda x: x.price)

                return OrderBook(bids=bids, asks=asks)

        except Exception as e:
            log.warning("single_book_fetch_error", token=token_id[:16], error=str(e))
            return None

    # == Convenience =======================================

    def combined_ask_price(self) -> Optional[float]:
        """Total cost to buy 1 share of both UP and DOWN."""
        if not self._up_book or not self._down_book:
            return None
        up_ask = self._up_book.best_ask
        down_ask = self._down_book.best_ask
        if up_ask is not None and down_ask is not None:
            return up_ask + down_ask
        return None

    def orderbook_imbalance(self) -> Optional[float]:
        """Order book imbalance: positive = UP pressure, negative = DOWN."""
        if not self._up_book or not self._down_book:
            return None
        up_depth = sum(b.size for b in (self._up_book.bids or []))
        down_depth = sum(b.size for b in (self._down_book.bids or []))
        total = up_depth + down_depth
        if total == 0:
            return 0.0
        return (up_depth - down_depth) / total

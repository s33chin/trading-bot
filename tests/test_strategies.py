"""
Unit tests for strategies and risk management.
Run with: pytest tests/ -v
"""

import time

import pytest

from config import Config
from models import (
    BTCPrice,
    BTCWindow,
    Market,
    OrderBook,
    OrderBookLevel,
    Side,
    StrategySource,
    Trade,
    TradeAction,
    TradeStatus,
)
from strategies.momentum import MomentumStrategy
from strategies.arbitrage import ArbitrageStrategy
from strategies.fusion import FusionStrategy
from execution.risk_manager import RiskManager


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def config():
    return Config(
        max_trade_size=1.0,
        daily_loss_limit=10.0,
        max_trades_per_hour=8,
        entry_seconds_before_close=30,
        min_delta_threshold=0.02,
        max_token_price=0.85,
        arb_threshold=0.995,
        arb_shares=10,
        min_fusion_confidence=0.55,
    )


@pytest.fixture
def market():
    now = time.time()
    return Market(
        condition_id="test-condition-123",
        question="Will BTC go up in the next 15 minutes?",
        up_token_id="up-token-abc",
        down_token_id="down-token-xyz",
        end_timestamp=now + 25,  # 25 seconds remaining
        start_timestamp=now - 875,  # started 875s ago (of 900s)
        slug="btc-updown-15m-test",
    )


@pytest.fixture
def btc_window_up():
    return BTCWindow(
        window_open_price=100000.0,
        window_open_time=time.time() - 870,
        current_price=100050.0,  # +0.05%
        current_time=time.time(),
    )


@pytest.fixture
def btc_window_down():
    return BTCWindow(
        window_open_price=100000.0,
        window_open_time=time.time() - 870,
        current_price=99950.0,  # -0.05%
        current_time=time.time(),
    )


@pytest.fixture
def up_book():
    return OrderBook(
        bids=[OrderBookLevel(0.58, 100), OrderBookLevel(0.55, 200)],
        asks=[OrderBookLevel(0.62, 100), OrderBookLevel(0.65, 200)],
    )


@pytest.fixture
def down_book():
    return OrderBook(
        bids=[OrderBookLevel(0.38, 100), OrderBookLevel(0.35, 200)],
        asks=[OrderBookLevel(0.42, 100), OrderBookLevel(0.45, 200)],
    )


@pytest.fixture
def arb_up_book():
    return OrderBook(
        asks=[OrderBookLevel(0.48, 100), OrderBookLevel(0.50, 200)],
        bids=[OrderBookLevel(0.46, 100)],
    )


@pytest.fixture
def arb_down_book():
    return OrderBook(
        asks=[OrderBookLevel(0.50, 100), OrderBookLevel(0.52, 200)],
        bids=[OrderBookLevel(0.48, 100)],
    )


# ── Momentum Strategy Tests ──────────────────────────────

class TestMomentumStrategy:

    def test_signal_up_when_btc_up(self, config, market, btc_window_up, up_book, down_book):
        strat = MomentumStrategy(config)
        signal = strat.evaluate(market, btc_window_up, up_book, down_book)
        assert signal is not None
        assert signal.direction == Side.UP
        assert signal.confidence > 0.5

    def test_signal_down_when_btc_down(self, config, market, btc_window_down, up_book, down_book):
        strat = MomentumStrategy(config)
        signal = strat.evaluate(market, btc_window_down, up_book, down_book)
        assert signal is not None
        assert signal.direction == Side.DOWN

    def test_no_signal_when_too_early(self, config, btc_window_up, up_book, down_book):
        strat = MomentumStrategy(config)
        now = time.time()
        early_market = Market(
            condition_id="test",
            question="test",
            up_token_id="up",
            down_token_id="down",
            end_timestamp=now + 600,  # 10 minutes remaining
            start_timestamp=now - 300,
        )
        signal = strat.evaluate(early_market, btc_window_up, up_book, down_book)
        assert signal is None

    def test_no_signal_when_delta_too_small(self, config, market, up_book, down_book):
        strat = MomentumStrategy(config)
        tiny_window = BTCWindow(
            window_open_price=100000.0,
            window_open_time=time.time() - 870,
            current_price=100001.0,  # +0.001% — below threshold
            current_time=time.time(),
        )
        signal = strat.evaluate(market, tiny_window, up_book, down_book)
        assert signal is None

    def test_no_signal_when_token_too_expensive(self, config, market, btc_window_up, down_book):
        strat = MomentumStrategy(config)
        expensive_book = OrderBook(
            asks=[OrderBookLevel(0.92, 100)],
            bids=[OrderBookLevel(0.90, 100)],
        )
        signal = strat.evaluate(market, btc_window_up, expensive_book, down_book)
        assert signal is None


# ── Arbitrage Strategy Tests ──────────────────────────────

class TestArbitrageStrategy:

    def test_arb_when_combined_below_threshold(self, config, market, arb_up_book, arb_down_book):
        strat = ArbitrageStrategy(config)
        signal = strat.evaluate(market, arb_up_book, arb_down_book)
        assert signal is not None
        assert signal.source == StrategySource.ARBITRAGE
        assert float(signal.metadata["combined_price"].strip("$")) < 1.0

    def test_no_arb_when_combined_above_threshold(self, config, market):
        strat = ArbitrageStrategy(config)
        expensive_up = OrderBook(
            asks=[OrderBookLevel(0.55, 100)],
            bids=[OrderBookLevel(0.53, 100)],
        )
        expensive_down = OrderBook(
            asks=[OrderBookLevel(0.50, 100)],
            bids=[OrderBookLevel(0.48, 100)],
        )
        signal = strat.evaluate(market, expensive_up, expensive_down)
        assert signal is None

    def test_no_arb_when_insufficient_liquidity(self, config, market):
        strat = ArbitrageStrategy(config)
        thin_book = OrderBook(
            asks=[OrderBookLevel(0.48, 2)],  # only 2 shares available
            bids=[],
        )
        signal = strat.evaluate(market, thin_book, thin_book)
        assert signal is None


# ── Fusion Strategy Tests ─────────────────────────────────

class TestFusionStrategy:

    def test_fusion_produces_signal(self, config, market, btc_window_up, up_book, down_book):
        strat = FusionStrategy(config)
        klines = [
            {"open": 99990, "close": 100010, "high": 100020, "low": 99980, "volume": 1},
            {"open": 100010, "close": 100030, "high": 100040, "low": 100000, "volume": 1},
            {"open": 100030, "close": 100040, "high": 100050, "low": 100020, "volume": 1},
            {"open": 100040, "close": 100050, "high": 100060, "low": 100030, "volume": 1},
            {"open": 100050, "close": 100055, "high": 100060, "low": 100040, "volume": 1},
        ]
        result = strat.evaluate(market, btc_window_up, up_book, down_book, klines)
        assert result is not None
        assert result.action in (TradeAction.BUY_UP, TradeAction.BUY_DOWN, TradeAction.SKIP)

    def test_fusion_skips_when_too_early(self, config, btc_window_up, up_book, down_book):
        strat = FusionStrategy(config)
        now = time.time()
        early_market = Market(
            condition_id="test",
            question="test",
            up_token_id="up",
            down_token_id="down",
            end_timestamp=now + 600,
            start_timestamp=now - 300,
        )
        result = strat.evaluate(early_market, btc_window_up, up_book, down_book)
        assert result.action == TradeAction.SKIP


# ── Risk Manager Tests ────────────────────────────────────

class TestRiskManager:

    def test_can_trade_initially(self, config):
        rm = RiskManager(config)
        allowed, reason = rm.can_trade()
        assert allowed is True

    def test_daily_loss_limit(self, config):
        rm = RiskManager(config)
        now = time.time()
        for i in range(15):
            trade = Trade(
                id=str(i),
                market=Market("c", "q", "u", "d", now + 100, now, "s"),
                action=TradeAction.BUY_UP,
                side=Side.UP,
                token_id="t",
                price=0.50,
                size=1.0,
                shares=2.0,
                status=TradeStatus.FILLED,
                pnl=-1.0,
            )
            rm.record_trade(trade)

        allowed, reason = rm.can_trade()
        assert allowed is False
        assert "daily loss limit" in reason

    def test_trade_frequency_limit(self, config):
        config.max_trades_per_hour = 3
        rm = RiskManager(config)
        now = time.time()
        for i in range(5):
            rm.state.recent_trade_times.append(now - 10)

        allowed, reason = rm.can_trade()
        assert allowed is False
        assert "frequency" in reason

    def test_position_sizing_scales_with_confidence(self, config):
        rm = RiskManager(config)
        size_low = rm.position_size(0.5)
        size_high = rm.position_size(0.9)
        assert size_high > size_low

    def test_position_sizing_reduces_after_losses(self, config):
        rm = RiskManager(config)
        rm.state.consecutive_losses = 3
        size = rm.position_size(0.8)
        normal_size = config.max_trade_size * 0.8
        assert size < normal_size


# ── Model Tests ───────────────────────────────────────────

class TestOrderBook:

    def test_fill_price_walks_book(self):
        book = OrderBook(
            asks=[
                OrderBookLevel(0.50, 10),
                OrderBookLevel(0.55, 20),
                OrderBookLevel(0.60, 30),
            ]
        )
        # 10 shares at $0.50, need 5 more at $0.55
        fill = book.fill_price(15)
        assert fill is not None
        expected = (10 * 0.50 + 5 * 0.55) / 15
        assert abs(fill - expected) < 0.001

    def test_fill_price_insufficient_liquidity(self):
        book = OrderBook(asks=[OrderBookLevel(0.50, 5)])
        fill = book.fill_price(100)
        assert fill is None

    def test_best_bid_ask(self):
        book = OrderBook(
            bids=[OrderBookLevel(0.48, 10), OrderBookLevel(0.45, 20)],
            asks=[OrderBookLevel(0.52, 10), OrderBookLevel(0.55, 20)],
        )
        assert book.best_bid == 0.48
        assert book.best_ask == 0.52
        assert abs(book.spread - 0.04) < 0.001


class TestBTCWindow:

    def test_delta_positive(self):
        w = BTCWindow(window_open_price=100000, current_price=100100)
        assert w.delta > 0
        assert w.direction == Side.UP

    def test_delta_negative(self):
        w = BTCWindow(window_open_price=100000, current_price=99900)
        assert w.delta < 0
        assert w.direction == Side.DOWN

    def test_delta_none_when_no_data(self):
        w = BTCWindow()
        assert w.delta is None
        assert w.direction is None

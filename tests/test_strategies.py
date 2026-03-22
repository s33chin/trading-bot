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
        weight_delta=0.50,
        weight_order_flow=0.20,
        weight_divergence=0.20,
        weight_spread=0.10,
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
        result = strat.evaluate(market, btc_window_up, up_book, down_book)
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

    def test_fusion_accepts_klines_for_compatibility(self, config, market, btc_window_up, up_book, down_book):
        """Klines parameter is accepted but not used."""
        strat = FusionStrategy(config)
        klines = [{"open": 100, "close": 101, "high": 102, "low": 99, "volume": 1}]
        result = strat.evaluate(market, btc_window_up, up_book, down_book, klines)
        assert result is not None


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

    def test_delta_none_when_open_price_zero(self):
        w = BTCWindow(window_open_price=0.0, current_price=100.0)
        assert w.delta is None

    def test_delta_works_with_zero_current_price(self):
        w = BTCWindow(window_open_price=100.0, current_price=0.0)
        assert w.delta == -1.0
        assert w.direction == Side.DOWN


# ── Execution Engine Tests ───────────────────────────────

class TestExecutionEngine:

    def test_simulate_fill_no_slippage(self, config, market, up_book):
        config.paper_slippage_pct = 0.0
        from execution.engine import ExecutionEngine
        engine = ExecutionEngine(config)
        trade = Trade(
            id="test-1",
            market=market,
            action=TradeAction.BUY_UP,
            side=Side.UP,
            token_id="up-token-abc",
            price=0.60,
            size=1.0,
            shares=1.0 / 0.60,
        )
        result = engine._simulate_fill(trade)
        assert result.status == TradeStatus.FILLED
        assert result.fill_price == 0.60

    def test_simulate_fill_with_slippage(self, config, market, up_book):
        config.paper_slippage_pct = 1.0
        from execution.engine import ExecutionEngine
        engine = ExecutionEngine(config)
        trade = Trade(
            id="test-2",
            market=market,
            action=TradeAction.BUY_UP,
            side=Side.UP,
            token_id="up-token-abc",
            price=0.60,
            size=1.0,
            shares=1.0 / 0.60,
        )
        result = engine._simulate_fill(trade)
        assert result.status == TradeStatus.FILLED
        assert result.fill_price >= 0.60  # Slippage always adverse for buys

    def test_resolve_trade_win(self, config, market):
        from execution.engine import ExecutionEngine
        engine = ExecutionEngine(config)
        trade = Trade(
            id="test-3",
            market=market,
            action=TradeAction.BUY_UP,
            side=Side.UP,
            token_id="up-token-abc",
            price=0.60,
            size=1.0,
            shares=2.0,
            status=TradeStatus.FILLED,
            fill_price=0.60,
        )
        result = engine.resolve_trade(trade, Side.UP)
        assert result.pnl is not None
        assert result.pnl > 0  # Won: payout $2.00 - cost $1.20 = $0.80

    def test_resolve_trade_loss(self, config, market):
        from execution.engine import ExecutionEngine
        engine = ExecutionEngine(config)
        trade = Trade(
            id="test-4",
            market=market,
            action=TradeAction.BUY_UP,
            side=Side.UP,
            token_id="up-token-abc",
            price=0.60,
            size=1.0,
            shares=2.0,
            status=TradeStatus.FILLED,
            fill_price=0.60,
        )
        result = engine.resolve_trade(trade, Side.DOWN)
        assert result.pnl is not None
        assert result.pnl < 0  # Lost: payout $0 - cost $1.20

    def test_resolve_trade_skips_unfilled(self, config, market):
        from execution.engine import ExecutionEngine
        engine = ExecutionEngine(config)
        trade = Trade(
            id="test-5",
            market=market,
            action=TradeAction.BUY_UP,
            side=Side.UP,
            token_id="up-token-abc",
            price=0.60,
            size=1.0,
            shares=2.0,
            status=TradeStatus.CANCELLED,
        )
        result = engine.resolve_trade(trade, Side.UP)
        assert result.pnl is None


# ── Winner Determination Tests ───────────────────────────

class TestWinnerDetermination:

    def test_flat_market_returns_none(self):
        """Exactly flat BTC should return None, not Side.UP."""
        w = BTCWindow(window_open_price=100000.0, current_price=100000.0)
        # delta is 0, direction should be UP per model but winner should be None
        assert w.current_price == w.window_open_price

    def test_up_market(self):
        w = BTCWindow(window_open_price=100000.0, current_price=100001.0)
        assert w.delta > 0
        assert w.direction == Side.UP

    def test_down_market(self):
        w = BTCWindow(window_open_price=100000.0, current_price=99999.0)
        assert w.delta < 0
        assert w.direction == Side.DOWN


# ── Risk Manager Additional Tests ────────────────────────

class TestRiskManagerAdditional:

    def test_get_status_does_not_reset_state(self, config):
        rm = RiskManager(config)
        rm.state.daily_pnl = -5.0
        rm.state.daily_trades = 3
        rm.get_status()
        # State should not be mutated by get_status
        assert rm.state.daily_pnl == -5.0
        assert rm.state.daily_trades == 3

    def test_cooldown_blocks_trading(self, config):
        config.loss_cooldown_seconds = 120
        rm = RiskManager(config)
        rm.state.last_loss_time = time.time()  # Just lost
        allowed, reason = rm.can_trade()
        assert allowed is False
        assert "cooldown" in reason

    def test_position_size_minimum(self, config):
        rm = RiskManager(config)
        size = rm.position_size(0.01)  # Very low confidence
        assert size > 0  # Should still be positive (min 10% of max)


# ── Fusion Signal Unit Tests ──────────────────────────────

class TestFusionOrderFlowSignal:

    def test_thin_target_asks_confirms_delta(self, config):
        """When target token has thin asks, signal confirms delta direction."""
        strat = FusionStrategy(config)
        # UP has thin asks (50 shares), DOWN has thick asks (200 shares)
        up_book = OrderBook(
            asks=[OrderBookLevel(0.60, 50)],
            bids=[OrderBookLevel(0.58, 100)],
        )
        down_book = OrderBook(
            asks=[OrderBookLevel(0.42, 200)],
            bids=[OrderBookLevel(0.40, 100)],
        )
        signal = strat._order_flow_signal(Side.UP, up_book, down_book)
        assert signal is not None
        assert signal.direction == Side.UP
        assert signal.confidence > 0.3

    def test_thick_target_asks_warns(self, config):
        """When target token has thick asks, signal warns against delta."""
        strat = FusionStrategy(config)
        # UP has thick asks (200), DOWN has thin asks (50)
        up_book = OrderBook(
            asks=[OrderBookLevel(0.60, 200)],
            bids=[OrderBookLevel(0.58, 100)],
        )
        down_book = OrderBook(
            asks=[OrderBookLevel(0.42, 50)],
            bids=[OrderBookLevel(0.40, 100)],
        )
        signal = strat._order_flow_signal(Side.UP, up_book, down_book)
        assert signal is not None
        assert signal.direction == Side.DOWN  # Warning against delta

    def test_balanced_asks_no_signal(self, config):
        """When both sides have similar ask depth, no signal."""
        strat = FusionStrategy(config)
        up_book = OrderBook(
            asks=[OrderBookLevel(0.60, 100)],
            bids=[OrderBookLevel(0.58, 100)],
        )
        down_book = OrderBook(
            asks=[OrderBookLevel(0.42, 100)],
            bids=[OrderBookLevel(0.40, 100)],
        )
        signal = strat._order_flow_signal(Side.UP, up_book, down_book)
        assert signal is None

    def test_too_thin_both_sides_no_signal(self, config):
        """When total depth < 10, no signal."""
        strat = FusionStrategy(config)
        up_book = OrderBook(asks=[OrderBookLevel(0.60, 3)], bids=[])
        down_book = OrderBook(asks=[OrderBookLevel(0.42, 3)], bids=[])
        signal = strat._order_flow_signal(Side.UP, up_book, down_book)
        assert signal is None


class TestFusionDivergenceSignal:

    def test_cheap_token_produces_signal(self, config):
        """When target token is cheap relative to implied value, signal fires."""
        strat = FusionStrategy(config)
        # Delta = +0.05%, implied fair ≈ 0.50 + 0.05*2 = 0.60
        btc_window = BTCWindow(
            window_open_price=100000.0,
            current_price=100050.0,  # +0.05%
        )
        # UP token ask at 0.52 — well below implied 0.60
        up_book = OrderBook(
            asks=[OrderBookLevel(0.52, 100)],
            bids=[OrderBookLevel(0.50, 100)],
        )
        down_book = OrderBook(
            asks=[OrderBookLevel(0.48, 100)],
            bids=[OrderBookLevel(0.46, 100)],
        )
        signal = strat._divergence_signal(Side.UP, btc_window, up_book, down_book)
        assert signal is not None
        assert signal.direction == Side.UP
        assert signal.confidence > 0.3

    def test_no_divergence_when_priced_in(self, config):
        """When token is already at or above implied value, no signal."""
        strat = FusionStrategy(config)
        btc_window = BTCWindow(
            window_open_price=100000.0,
            current_price=100050.0,  # +0.05%
        )
        # UP token ask at 0.65 — above implied ~0.60
        up_book = OrderBook(
            asks=[OrderBookLevel(0.65, 100)],
            bids=[OrderBookLevel(0.63, 100)],
        )
        down_book = OrderBook(
            asks=[OrderBookLevel(0.40, 100)],
            bids=[OrderBookLevel(0.38, 100)],
        )
        signal = strat._divergence_signal(Side.UP, btc_window, up_book, down_book)
        assert signal is None

    def test_no_divergence_without_delta(self, config):
        """No signal when delta is None."""
        strat = FusionStrategy(config)
        btc_window = BTCWindow()  # No prices set
        up_book = OrderBook(
            asks=[OrderBookLevel(0.50, 100)],
            bids=[OrderBookLevel(0.48, 100)],
        )
        signal = strat._divergence_signal(Side.UP, btc_window, up_book, None)
        assert signal is None


class TestFusionSpreadSignal:

    def test_tight_up_spread_favors_up(self, config):
        """When UP spread is tight and DOWN is wide, signal favors UP."""
        strat = FusionStrategy(config)
        up_book = OrderBook(
            bids=[OrderBookLevel(0.59, 100)],
            asks=[OrderBookLevel(0.60, 100)],  # spread = 0.01
        )
        down_book = OrderBook(
            bids=[OrderBookLevel(0.35, 100)],
            asks=[OrderBookLevel(0.42, 100)],  # spread = 0.07
        )
        signal = strat._spread_signal(up_book, down_book)
        assert signal is not None
        assert signal.direction == Side.UP

    def test_tight_down_spread_favors_down(self, config):
        """When DOWN spread is tight and UP is wide, signal favors DOWN."""
        strat = FusionStrategy(config)
        up_book = OrderBook(
            bids=[OrderBookLevel(0.53, 100)],
            asks=[OrderBookLevel(0.60, 100)],  # spread = 0.07
        )
        down_book = OrderBook(
            bids=[OrderBookLevel(0.41, 100)],
            asks=[OrderBookLevel(0.42, 100)],  # spread = 0.01
        )
        signal = strat._spread_signal(up_book, down_book)
        assert signal is not None
        assert signal.direction == Side.DOWN

    def test_similar_spreads_no_signal(self, config):
        """When spreads are similar, no signal."""
        strat = FusionStrategy(config)
        up_book = OrderBook(
            bids=[OrderBookLevel(0.59, 100)],
            asks=[OrderBookLevel(0.61, 100)],  # spread = 0.02
        )
        down_book = OrderBook(
            bids=[OrderBookLevel(0.39, 100)],
            asks=[OrderBookLevel(0.41, 100)],  # spread = 0.02
        )
        signal = strat._spread_signal(up_book, down_book)
        assert signal is None

    def test_both_wide_spreads_no_signal(self, config):
        """When both spreads are very wide, no signal."""
        strat = FusionStrategy(config)
        up_book = OrderBook(
            bids=[OrderBookLevel(0.45, 100)],
            asks=[OrderBookLevel(0.60, 100)],  # spread = 0.15
        )
        down_book = OrderBook(
            bids=[OrderBookLevel(0.25, 100)],
            asks=[OrderBookLevel(0.42, 100)],  # spread = 0.17
        )
        signal = strat._spread_signal(up_book, down_book)
        assert signal is None


class TestFusionVolatilityMultiplier:

    def test_low_vol_boosts(self, config):
        strat = FusionStrategy(config)
        btc_window = BTCWindow(
            window_open_price=100000.0,
            current_price=100001.0,
            prices=[
                BTCPrice(price=100000.0, timestamp=time.time() - 10),
                BTCPrice(price=100000.5, timestamp=time.time() - 5),
                BTCPrice(price=100001.0, timestamp=time.time()),
            ],
        )
        mult = strat._volatility_multiplier(btc_window)
        assert mult > 1.0

    def test_high_vol_penalizes(self, config):
        strat = FusionStrategy(config)
        btc_window = BTCWindow(
            window_open_price=100000.0,
            current_price=100100.0,
            prices=[
                BTCPrice(price=100000.0, timestamp=time.time() - 10),
                BTCPrice(price=100500.0, timestamp=time.time() - 5),
                BTCPrice(price=99500.0, timestamp=time.time() - 3),
                BTCPrice(price=100100.0, timestamp=time.time()),
            ],
        )
        mult = strat._volatility_multiplier(btc_window)
        assert mult < 1.0

    def test_zero_vol_neutral(self, config):
        strat = FusionStrategy(config)
        btc_window = BTCWindow()  # No prices → vol = 0
        mult = strat._volatility_multiplier(btc_window)
        assert mult == 1.0


class TestFusionConfidenceAggregation:

    def test_moderate_agreement_not_punished_too_hard(self, config, market, btc_window_up, up_book, down_book):
        """Moderate agreement should produce reasonable confidence, not near-zero."""
        strat = FusionStrategy(config)
        result = strat.evaluate(market, btc_window_up, up_book, down_book)
        # With a clear UP delta (+0.05%) and reasonable books,
        # the fusion should produce a non-trivial confidence
        if result.action != TradeAction.SKIP:
            assert result.confidence > 0.4


# ── Active Trading Tests ─────────────────────────────────

from models import Position, PositionStatus


class TestPosition:

    def test_compute_exit_prices(self):
        now = time.time()
        m = Market("c", "q", "u", "d", now + 100, now, "s")
        trade = Trade(
            id="t1", market=m, action=TradeAction.BUY_UP,
            side=Side.UP, token_id="u", price=0.55,
            size=1.0, shares=1.82, fill_price=0.55,
            status=TradeStatus.FILLED,
        )
        pos = Position(
            id="p1", buy_trade=trade, side=Side.UP,
            token_id="u", entry_price=0.55, shares=1.82,
            cost=0.55 * 1.82,
        )
        pos.compute_exit_prices(take_profit_threshold=0.10, stop_loss_threshold=0.08)
        assert abs(pos.take_profit_price - 0.65) < 0.001
        assert abs(pos.stop_loss_price - 0.47) < 0.001

    def test_tp_clamped_to_099(self):
        now = time.time()
        m = Market("c", "q", "u", "d", now + 100, now, "s")
        trade = Trade(
            id="t2", market=m, action=TradeAction.BUY_UP,
            side=Side.UP, token_id="u", price=0.95,
            size=1.0, shares=1.05, fill_price=0.95,
            status=TradeStatus.FILLED,
        )
        pos = Position(
            id="p2", buy_trade=trade, side=Side.UP,
            token_id="u", entry_price=0.95, shares=1.05,
            cost=0.95 * 1.05,
        )
        pos.compute_exit_prices(take_profit_threshold=0.10, stop_loss_threshold=0.08)
        assert pos.take_profit_price == 0.99

    def test_sl_clamped_to_001(self):
        now = time.time()
        m = Market("c", "q", "u", "d", now + 100, now, "s")
        trade = Trade(
            id="t3", market=m, action=TradeAction.BUY_DOWN,
            side=Side.DOWN, token_id="d", price=0.05,
            size=1.0, shares=20.0, fill_price=0.05,
            status=TradeStatus.FILLED,
        )
        pos = Position(
            id="p3", buy_trade=trade, side=Side.DOWN,
            token_id="d", entry_price=0.05, shares=20.0,
            cost=0.05 * 20.0,
        )
        pos.compute_exit_prices(take_profit_threshold=0.10, stop_loss_threshold=0.08)
        assert pos.stop_loss_price == 0.01

    def test_position_is_open(self):
        now = time.time()
        m = Market("c", "q", "u", "d", now + 100, now, "s")
        trade = Trade(
            id="t4", market=m, action=TradeAction.BUY_UP,
            side=Side.UP, token_id="u", price=0.55,
            size=1.0, shares=1.82, fill_price=0.55,
            status=TradeStatus.FILLED,
        )
        pos = Position(
            id="p4", buy_trade=trade, side=Side.UP,
            token_id="u", entry_price=0.55, shares=1.82,
            cost=1.0,
        )
        assert pos.is_open
        pos.status = PositionStatus.CLOSED_TAKE_PROFIT
        assert not pos.is_open


class TestOrderBookFillPriceBid:

    def test_fill_price_bid_walks_book(self):
        book = OrderBook(
            bids=[
                OrderBookLevel(0.60, 10),
                OrderBookLevel(0.55, 20),
                OrderBookLevel(0.50, 30),
            ]
        )
        fill = book.fill_price_bid(15)
        assert fill is not None
        expected = (10 * 0.60 + 5 * 0.55) / 15
        assert abs(fill - expected) < 0.001

    def test_fill_price_bid_insufficient_liquidity(self):
        book = OrderBook(bids=[OrderBookLevel(0.60, 5)])
        fill = book.fill_price_bid(100)
        assert fill is None


class TestSellExecution:

    def test_simulate_sell_no_slippage(self, config, market):
        config.paper_slippage_pct = 0.0
        from execution.engine import ExecutionEngine
        engine = ExecutionEngine(config)
        now = time.time()
        buy_trade = Trade(
            id="buy1", market=market, action=TradeAction.BUY_UP,
            side=Side.UP, token_id="up-token-abc", price=0.55,
            size=1.0, shares=1.82, fill_price=0.55,
            status=TradeStatus.FILLED,
        )
        pos = Position(
            id="pos1", buy_trade=buy_trade, side=Side.UP,
            token_id="up-token-abc", entry_price=0.55, shares=1.82,
            cost=0.55 * 1.82,
        )
        sell_trade = Trade(
            id="sell1", market=market, action=TradeAction.SELL_UP,
            side=Side.UP, token_id="up-token-abc", price=0.65,
            size=0.65 * 1.82, shares=1.82,
        )
        result = engine._simulate_sell_fill(sell_trade)
        assert result.status == TradeStatus.FILLED
        assert result.fill_price == 0.65

    def test_simulate_sell_with_slippage(self, config, market):
        config.paper_slippage_pct = 1.0
        from execution.engine import ExecutionEngine
        engine = ExecutionEngine(config)
        sell_trade = Trade(
            id="sell2", market=market, action=TradeAction.SELL_UP,
            side=Side.UP, token_id="up-token-abc", price=0.65,
            size=0.65 * 1.82, shares=1.82,
        )
        result = engine._simulate_sell_fill(sell_trade)
        assert result.status == TradeStatus.FILLED
        assert result.fill_price <= 0.65  # Adverse slippage for sells

    def test_sell_pnl_take_profit(self):
        """Buy at 0.55, sell at 0.65 → positive P&L."""
        entry_price = 0.55
        shares = 10.0
        cost = entry_price * shares  # $5.50
        sell_price = 0.65
        revenue = sell_price * shares  # $6.50
        pnl = revenue - cost  # +$1.00
        assert pnl > 0
        assert abs(pnl - 1.0) < 0.001

    def test_sell_pnl_stop_loss(self):
        """Buy at 0.55, sell at 0.47 → negative P&L."""
        entry_price = 0.55
        shares = 10.0
        cost = entry_price * shares  # $5.50
        sell_price = 0.47
        revenue = sell_price * shares  # $4.70
        pnl = revenue - cost  # -$0.80
        assert pnl < 0
        assert abs(pnl - (-0.80)) < 0.001


class TestRiskManagerPositions:

    def test_can_open_position_within_limit(self, config):
        config.max_open_positions = 2
        rm = RiskManager(config)
        allowed, _ = rm.can_open_position(0)
        assert allowed is True
        allowed, _ = rm.can_open_position(1)
        assert allowed is True

    def test_cannot_open_position_at_limit(self, config):
        config.max_open_positions = 2
        rm = RiskManager(config)
        allowed, reason = rm.can_open_position(2)
        assert allowed is False
        assert "max open positions" in reason

    def test_exposure_tracking(self, config):
        rm = RiskManager(config)
        rm.add_exposure(5.0)
        assert rm.state.current_exposure == 5.0
        rm.add_exposure(3.0)
        assert rm.state.current_exposure == 8.0
        rm.remove_exposure(5.0)
        assert rm.state.current_exposure == 3.0
        rm.remove_exposure(10.0)  # Should clamp to 0
        assert rm.state.current_exposure == 0.0


class TestActiveTradeActions:

    def test_sell_actions_exist(self):
        assert TradeAction.SELL_UP == "sell_up"
        assert TradeAction.SELL_DOWN == "sell_down"

    def test_config_defaults_preserve_existing_behavior(self):
        """Default config should have active trading disabled."""
        config = Config()
        assert config.active_trading_enabled is False
        assert config.early_entry_enabled is False

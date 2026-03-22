"""
Fusion Strategy — Multi-Signal Combination

Combines genuinely independent signals into a single trading decision:
1. Window delta (primary — BTC price direction)
2. Order flow (ask-side thinness on target token)
3. Price-book divergence (token price lagging implied fair value)
4. Spread asymmetry (tight spread = market maker confidence)

Volatility is applied as a confidence multiplier, not a voting signal.

Each signal produces a direction + confidence, then a weighted
vote determines the final action.
"""

from __future__ import annotations

from config import Config
from logger import get_logger
from models import (
    BTCWindow,
    FusedSignal,
    Market,
    OrderBook,
    Side,
    Signal,
    StrategySource,
    TradeAction,
)

log = get_logger("strategy.fusion")


class FusionStrategy:
    """
    Multi-signal fusion engine.

    Weighted voting across 4 independent signal sources:
    - Delta: BTC price direction within the window
    - Order flow: ask-side thinness favoring the target direction
    - Divergence: token price lagging what delta implies
    - Spread: tight spread on one side = market maker confidence

    Volatility modulates the final confidence (not a voter).
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.weights = {
            "delta": config.weight_delta,
            "order_flow": config.weight_order_flow,
            "divergence": config.weight_divergence,
            "spread": config.weight_spread,
        }
        self.min_confidence = config.min_fusion_confidence
        self.entry_seconds = config.entry_seconds_before_close
        self.vol_multiplier_low = config.vol_multiplier_low
        self.vol_multiplier_high = config.vol_multiplier_high

    def evaluate(
        self,
        market: Market,
        btc_window: BTCWindow,
        up_book: OrderBook | None,
        down_book: OrderBook | None,
        recent_klines: list[dict] | None = None,
    ) -> FusedSignal:
        """
        Run all sub-signals, fuse, and return a FusedSignal.

        recent_klines is accepted for interface compatibility but not used —
        the old kline momentum signal was correlated with delta and has been
        replaced with genuinely independent structural signals.
        """
        remaining = market.seconds_remaining
        if remaining > self.entry_seconds or remaining < 3:
            return FusedSignal(
                action=TradeAction.SKIP,
                direction=None,
                confidence=0.0,
                reason=f"timing: {remaining:.0f}s remaining",
            )

        signals: list[Signal] = []

        # 1. Delta signal (primary — BTC price direction)
        delta_signal = self._delta_signal(btc_window)
        if delta_signal:
            signals.append(delta_signal)

        # Determine delta direction for dependent signals
        delta_direction = delta_signal.direction if delta_signal else btc_window.direction

        if delta_direction is not None:
            # 2. Order flow — ask-side thinness on target token
            of_signal = self._order_flow_signal(delta_direction, up_book, down_book)
            if of_signal:
                signals.append(of_signal)

            # 3. Price-book divergence — token price lagging implied value
            div_signal = self._divergence_signal(
                delta_direction, btc_window, up_book, down_book
            )
            if div_signal:
                signals.append(div_signal)

        # 4. Spread asymmetry — fully independent, doesn't need delta direction
        spread_signal = self._spread_signal(up_book, down_book)
        if spread_signal:
            signals.append(spread_signal)

        if not signals:
            return FusedSignal(
                action=TradeAction.SKIP,
                direction=None,
                confidence=0.0,
                signals=signals,
                reason="no signals produced",
            )

        # Fuse signals via weighted voting
        return self._fuse(signals, btc_window, up_book, down_book)

    # ── Sub-signals ───────────────────────────────────────

    def _delta_signal(self, btc_window: BTCWindow) -> Signal | None:
        """Window delta — primary signal. Unchanged from before."""
        delta = btc_window.delta
        if delta is None:
            return None

        abs_delta = abs(delta)
        if abs_delta < 0.0001:  # < 0.01% — noise
            return None

        direction = Side.UP if delta > 0 else Side.DOWN
        # Confidence: sigmoid-ish scaling
        confidence = min(0.95, 0.3 + abs_delta * 100 * 4.0)

        return Signal(
            name="delta",
            direction=direction,
            confidence=confidence,
            source=StrategySource.FUSION,
            metadata={"delta_pct": f"{delta * 100:.4f}"},
        )

    def _order_flow_signal(
        self,
        delta_direction: Side,
        up_book: OrderBook | None,
        down_book: OrderBook | None,
    ) -> Signal | None:
        """
        Order flow signal based on ask-side thinness.

        Thin asks on the target token = cheap to accumulate = edge.
        Measures something genuinely different from delta: market
        microstructure and liquidity, not price direction.
        """
        if not up_book or not down_book:
            return None

        target_book = up_book if delta_direction == Side.UP else down_book
        opposing_book = down_book if delta_direction == Side.UP else up_book

        target_ask_depth = sum(a.size for a in target_book.asks[:5])
        opposing_ask_depth = sum(a.size for a in opposing_book.asks[:5])

        # Too thin on both sides to read meaningfully
        if target_ask_depth + opposing_ask_depth < 10:
            return None

        # Ratio: high = target asks are thin relative to opposing (good)
        thinness_ratio = opposing_ask_depth / (target_ask_depth + 1e-9)

        if thinness_ratio > 1.2:
            # Target token has thinner asks — easier to buy, confirms delta
            confidence = min(0.80, 0.3 + (thinness_ratio - 1.0) * 0.4)
            direction = delta_direction
        elif thinness_ratio < 0.8:
            # Opposing token has thinner asks — warning against delta
            confidence = min(0.65, 0.25 + (1.0 - thinness_ratio) * 0.3)
            direction = Side.DOWN if delta_direction == Side.UP else Side.UP
        else:
            return None  # No meaningful imbalance

        return Signal(
            name="order_flow",
            direction=direction,
            confidence=confidence,
            source=StrategySource.FUSION,
            metadata={
                "thinness_ratio": f"{thinness_ratio:.3f}",
                "target_ask_depth": f"{target_ask_depth:.0f}",
                "opposing_ask_depth": f"{opposing_ask_depth:.0f}",
            },
        )

    def _divergence_signal(
        self,
        delta_direction: Side,
        btc_window: BTCWindow,
        up_book: OrderBook | None,
        down_book: OrderBook | None,
    ) -> Signal | None:
        """
        Price-book divergence: is the target token cheaper than it should be?

        If BTC delta says UP but the UP token is still trading well below
        the implied fair value, the market hasn't priced in the move yet.
        That's an opportunity.
        """
        delta = btc_window.delta
        if delta is None:
            return None

        target_book = up_book if delta_direction == Side.UP else down_book
        if not target_book or target_book.best_ask is None:
            return None

        # Simple implied fair value model for binary outcome tokens:
        # At delta=0, fair value ≈ 0.50 (coin flip).
        # As |delta| grows, the winning side's fair value increases.
        abs_delta = abs(delta)
        implied = min(0.95, max(0.05, 0.50 + abs_delta * 100 * 2.0))

        actual_ask = target_book.best_ask
        discount = implied - actual_ask

        if discount < 0.01:
            # Token is already priced at or above implied — no divergence
            return None

        confidence = min(0.85, 0.3 + discount * 3.0)

        return Signal(
            name="divergence",
            direction=delta_direction,
            confidence=confidence,
            source=StrategySource.FUSION,
            metadata={
                "implied_fair": f"{implied:.4f}",
                "actual_ask": f"{actual_ask:.4f}",
                "discount": f"{discount:.4f}",
            },
        )

    def _spread_signal(
        self, up_book: OrderBook | None, down_book: OrderBook | None
    ) -> Signal | None:
        """
        Spread asymmetry — fully independent of BTC price direction.

        Tight spread = market makers are confident in that outcome.
        Wide spread = uncertainty/illiquidity.
        Compares UP vs DOWN spreads to determine which side has more conviction.
        """
        if not up_book or not down_book:
            return None

        up_spread = up_book.spread
        down_spread = down_book.spread
        if up_spread is None or down_spread is None:
            return None

        # Both too tight — no differential to read
        if up_spread < 0.005 and down_spread < 0.005:
            return None

        # Both too wide — both sides illiquid
        if up_spread > 0.10 and down_spread > 0.10:
            return None

        spread_diff = down_spread - up_spread  # Positive = UP is tighter

        if abs(spread_diff) < 0.005:
            return None  # No meaningful difference

        if spread_diff > 0:
            direction = Side.UP
        else:
            direction = Side.DOWN

        confidence = min(0.70, 0.25 + abs(spread_diff) * 5.0)

        return Signal(
            name="spread",
            direction=direction,
            confidence=confidence,
            source=StrategySource.FUSION,
            metadata={
                "up_spread": f"{up_spread:.4f}",
                "down_spread": f"{down_spread:.4f}",
                "spread_diff": f"{spread_diff:.4f}",
            },
        )

    def _volatility_multiplier(self, btc_window: BTCWindow) -> float:
        """
        Volatility as a confidence multiplier, not a voting signal.

        Low vol = clean trend, boost confidence.
        High vol = choppy, reduce confidence.
        """
        vol = btc_window.volatility
        if vol == 0:
            return 1.0

        if vol < 0.0005:
            return self.vol_multiplier_low  # Clean trend → boost
        elif vol < 0.001:
            return 1.0  # Neutral
        elif vol < 0.002:
            return 0.85  # Slightly choppy
        else:
            return self.vol_multiplier_high  # Very choppy → penalize

    # ── Fusion Logic ──────────────────────────────────────

    def _fuse(
        self,
        signals: list[Signal],
        btc_window: BTCWindow,
        up_book: OrderBook | None,
        down_book: OrderBook | None,
    ) -> FusedSignal:
        """
        Weighted voting across all signals with volatility multiplier.
        """
        weight_map = {
            "delta": self.weights["delta"],
            "order_flow": self.weights["order_flow"],
            "divergence": self.weights["divergence"],
            "spread": self.weights["spread"],
        }

        up_score = 0.0
        down_score = 0.0
        total_weight = 0.0

        for signal in signals:
            w = weight_map.get(signal.name, 0.1)
            weighted_conf = signal.confidence * w
            total_weight += w

            if signal.direction == Side.UP:
                up_score += weighted_conf
            else:
                down_score += weighted_conf

        if total_weight == 0:
            return FusedSignal(
                action=TradeAction.SKIP,
                direction=None,
                confidence=0.0,
                signals=signals,
                reason="zero total weight",
            )

        # Normalize
        up_score /= total_weight
        down_score /= total_weight

        # Direction = higher score
        if up_score > down_score:
            direction = Side.UP
            confidence = up_score
            opposing = down_score
        else:
            direction = Side.DOWN
            confidence = down_score
            opposing = up_score

        # Agreement-based confidence adjustment:
        # agreement_ratio: 0 when perfectly split, 1 when fully unanimous
        # Penalty is mild for decent agreement, strong for near-split
        agreement_ratio = (confidence - opposing) / (confidence + opposing + 1e-9)
        adjusted_confidence = confidence * (0.7 + 0.3 * agreement_ratio)

        # Apply volatility multiplier
        vol_mult = self._volatility_multiplier(btc_window)
        adjusted_confidence = min(0.95, adjusted_confidence * vol_mult)

        # Check token price feasibility
        book = up_book if direction == Side.UP else down_book
        max_price = self.config.max_token_price
        if book and book.best_ask and book.best_ask > max_price:
            return FusedSignal(
                action=TradeAction.SKIP,
                direction=direction,
                confidence=adjusted_confidence,
                signals=signals,
                reason=f"token too expensive: ${book.best_ask:.3f} > ${max_price:.2f}",
            )

        if adjusted_confidence < self.min_confidence:
            return FusedSignal(
                action=TradeAction.SKIP,
                direction=direction,  # Preserve direction so force-trade at T-5s can use it
                confidence=adjusted_confidence,
                signals=signals,
                reason=f"confidence {adjusted_confidence:.2%} < {self.min_confidence:.2%}",
            )

        action = TradeAction.BUY_UP if direction == Side.UP else TradeAction.BUY_DOWN

        log.info(
            "fusion_signal",
            direction=direction.value,
            confidence=f"{adjusted_confidence:.2%}",
            up_score=f"{up_score:.3f}",
            down_score=f"{down_score:.3f}",
            vol_mult=f"{vol_mult:.2f}",
            signal_count=len(signals),
            signal_names=[s.name for s in signals],
        )

        return FusedSignal(
            action=action,
            direction=direction,
            confidence=adjusted_confidence,
            signals=signals,
            reason=f"fusion: {len(signals)} signals, agreement={agreement_ratio:.2f}, vol_mult={vol_mult:.2f}",
        )

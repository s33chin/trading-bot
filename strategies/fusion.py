"""
Fusion Strategy — Multi-Signal Combination

Combines multiple signals into a single trading decision:
1. Window delta (primary)
2. Order book imbalance
3. Volatility regime
4. Short-term momentum (1-min candle trend)

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

    Weighted voting across 4 signal sources.
    Signals that agree amplify confidence; conflicting signals reduce it.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.weights = {
            "delta": config.weight_delta,
            "orderbook": config.weight_orderbook,
            "volatility": config.weight_volatility,
            "momentum": config.weight_momentum,
        }
        self.min_confidence = config.min_fusion_confidence
        self.entry_seconds = config.entry_seconds_before_close

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

        # 1. Delta signal
        delta_signal = self._delta_signal(btc_window)
        if delta_signal:
            signals.append(delta_signal)

        # 2. Order book imbalance
        ob_signal = self._orderbook_signal(up_book, down_book)
        if ob_signal:
            signals.append(ob_signal)

        # 3. Volatility regime
        vol_signal = self._volatility_signal(btc_window)
        if vol_signal:
            signals.append(vol_signal)

        # 4. Short-term momentum from klines
        mom_signal = self._momentum_signal(recent_klines)
        if mom_signal:
            signals.append(mom_signal)

        if not signals:
            return FusedSignal(
                action=TradeAction.SKIP,
                direction=None,
                confidence=0.0,
                signals=signals,
                reason="no signals produced",
            )

        # Fuse signals via weighted voting
        return self._fuse(signals, up_book, down_book)

    # ── Sub-signals ───────────────────────────────────────

    def _delta_signal(self, btc_window: BTCWindow) -> Signal | None:
        """Window delta — primary signal."""
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

    def _orderbook_signal(
        self, up_book: OrderBook | None, down_book: OrderBook | None
    ) -> Signal | None:
        """Order book imbalance signal."""
        if not up_book or not down_book:
            return None

        up_bid_depth = sum(b.size for b in up_book.bids[:5])
        down_bid_depth = sum(b.size for b in down_book.bids[:5])
        total = up_bid_depth + down_bid_depth

        if total < 10:  # Too thin to be meaningful
            return None

        imbalance = (up_bid_depth - down_bid_depth) / total
        if abs(imbalance) < 0.1:  # Less than 10% imbalance — noise
            return None

        direction = Side.UP if imbalance > 0 else Side.DOWN
        confidence = min(0.80, 0.3 + abs(imbalance) * 0.8)

        return Signal(
            name="orderbook_imbalance",
            direction=direction,
            confidence=confidence,
            source=StrategySource.FUSION,
            metadata={"imbalance": f"{imbalance:.3f}", "total_depth": f"{total:.0f}"},
        )

    def _volatility_signal(self, btc_window: BTCWindow) -> Signal | None:
        """
        Volatility regime signal.
        High vol = less confident in direction (market is choppy).
        Low vol = more confident (trend is clean).
        This adjusts confidence, not direction.
        """
        vol = btc_window.volatility
        if vol == 0:
            return None

        direction = btc_window.direction
        if direction is None:
            return None

        # Low volatility → higher confidence in the existing direction
        # High volatility → lower confidence
        if vol < 0.0005:
            confidence = 0.70  # Very calm — trend likely holds
        elif vol < 0.001:
            confidence = 0.55
        elif vol < 0.002:
            confidence = 0.40
        else:
            confidence = 0.25  # Very choppy — uncertain

        return Signal(
            name="volatility_regime",
            direction=direction,
            confidence=confidence,
            source=StrategySource.FUSION,
            metadata={"volatility": f"{vol:.6f}"},
        )

    def _momentum_signal(self, klines: list[dict] | None) -> Signal | None:
        """
        Short-term momentum from recent 1-min candles.
        Checks if last N candles are consistently moving in one direction.
        """
        if not klines or len(klines) < 3:
            return None

        # Use last 5 candles (or fewer if not available)
        recent = klines[-5:]
        up_candles = sum(1 for k in recent if k["close"] > k["open"])
        down_candles = len(recent) - up_candles

        if up_candles == down_candles:
            return None

        direction = Side.UP if up_candles > down_candles else Side.DOWN
        dominance = max(up_candles, down_candles) / len(recent)
        confidence = min(0.75, 0.3 + dominance * 0.5)

        return Signal(
            name="candle_momentum",
            direction=direction,
            confidence=confidence,
            source=StrategySource.FUSION,
            metadata={
                "up_candles": up_candles,
                "down_candles": down_candles,
                "candle_count": len(recent),
            },
        )

    # ── Fusion Logic ──────────────────────────────────────

    def _fuse(
        self,
        signals: list[Signal],
        up_book: OrderBook | None,
        down_book: OrderBook | None,
    ) -> FusedSignal:
        """
        Weighted voting across all signals.
        """
        weight_map = {
            "delta": self.weights["delta"],
            "orderbook_imbalance": self.weights["orderbook"],
            "volatility_regime": self.weights["volatility"],
            "candle_momentum": self.weights["momentum"],
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

        # Reduce confidence if signals disagree
        agreement = confidence - opposing
        adjusted_confidence = min(0.95, confidence * (0.5 + agreement))

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
                direction=direction,
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
            signal_count=len(signals),
        )

        return FusedSignal(
            action=action,
            direction=direction,
            confidence=adjusted_confidence,
            signals=signals,
            reason=f"fusion: {len(signals)} signals, agreement={agreement:.2f}",
        )

"""
Prometheus metrics exporter for Grafana monitoring.
Exposes trading metrics at /metrics endpoint.
"""

from __future__ import annotations

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    start_http_server,
)

from logger import get_logger

log = get_logger("metrics")


class Metrics:
    """
    Prometheus metrics for the trading bot.
    Start the HTTP server, then update metrics as events occur.
    """

    def __init__(self) -> None:
        # Info
        self.bot_info = Info("polybot", "Bot configuration")

        # Counters
        self.trades_total = Counter(
            "polybot_trades_total",
            "Total trades executed",
            ["strategy", "side", "status"],
        )
        self.signals_total = Counter(
            "polybot_signals_total",
            "Total signals generated",
            ["strategy", "direction"],
        )
        self.errors_total = Counter(
            "polybot_errors_total",
            "Total errors",
            ["component"],
        )
        self.markets_processed = Counter(
            "polybot_markets_processed_total",
            "Total 15-min markets processed",
        )

        # Gauges
        self.daily_pnl = Gauge(
            "polybot_daily_pnl_usd",
            "Daily P&L in USD",
        )
        self.daily_trades = Gauge(
            "polybot_daily_trades",
            "Number of trades today",
        )
        self.win_rate = Gauge(
            "polybot_win_rate",
            "Current win rate (rolling)",
        )
        self.btc_price = Gauge(
            "polybot_btc_price_usd",
            "Current BTC price",
        )
        self.btc_delta = Gauge(
            "polybot_btc_delta_pct",
            "BTC window delta percentage",
        )
        self.up_ask = Gauge(
            "polybot_up_ask_price",
            "UP token best ask",
        )
        self.down_ask = Gauge(
            "polybot_down_ask_price",
            "DOWN token best ask",
        )
        self.combined_ask = Gauge(
            "polybot_combined_ask_price",
            "Combined UP+DOWN ask price",
        )
        self.signal_confidence = Gauge(
            "polybot_signal_confidence",
            "Latest signal confidence",
            ["strategy"],
        )
        self.consecutive_losses = Gauge(
            "polybot_consecutive_losses",
            "Consecutive losing trades",
        )
        self.balance_usd = Gauge(
            "polybot_balance_usd",
            "Current USDC balance",
        )

        # Active Trading
        self.open_positions = Gauge(
            "polybot_open_positions",
            "Number of currently open positions",
        )
        self.positions_closed_tp = Counter(
            "polybot_positions_closed_tp_total",
            "Positions closed by take-profit",
        )
        self.positions_closed_sl = Counter(
            "polybot_positions_closed_sl_total",
            "Positions closed by stop-loss",
        )

        # Histograms
        self.trade_pnl = Histogram(
            "polybot_trade_pnl_usd",
            "Per-trade P&L distribution",
            buckets=[-1.0, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1.0],
        )
        self.fill_latency = Histogram(
            "polybot_fill_latency_seconds",
            "Order fill latency",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        )

    def start(self, port: int = 8000) -> None:
        """Start the Prometheus metrics HTTP server."""
        try:
            start_http_server(port)
            log.info("prometheus_started", port=port)
        except Exception as e:
            log.error("prometheus_start_error", error=str(e))

    def record_trade(
        self,
        strategy: str,
        side: str,
        status: str,
        pnl: float | None = None,
    ) -> None:
        self.trades_total.labels(
            strategy=strategy, side=side, status=status
        ).inc()
        if pnl is not None:
            self.trade_pnl.observe(pnl)

    def record_signal(self, strategy: str, direction: str) -> None:
        self.signals_total.labels(strategy=strategy, direction=direction).inc()

    def record_error(self, component: str) -> None:
        self.errors_total.labels(component=component).inc()

    def update_prices(
        self,
        btc: float | None = None,
        delta_pct: float | None = None,
        up_ask: float | None = None,
        down_ask: float | None = None,
    ) -> None:
        if btc is not None:
            self.btc_price.set(btc)
        if delta_pct is not None:
            self.btc_delta.set(delta_pct)
        if up_ask is not None:
            self.up_ask.set(up_ask)
        if down_ask is not None:
            self.down_ask.set(down_ask)
        if up_ask is not None and down_ask is not None:
            self.combined_ask.set(up_ask + down_ask)

    def update_risk(self, risk_status: dict) -> None:
        self.daily_pnl.set(risk_status.get("daily_pnl", 0))
        self.daily_trades.set(risk_status.get("daily_trades", 0))
        self.consecutive_losses.set(risk_status.get("consecutive_losses", 0))

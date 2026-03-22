"""
Configuration management with validation.
All settings loaded from environment variables with sensible defaults.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class TradingMode(str, Enum):
    LIVE = "live"
    PAPER = "paper"


class Strategy(str, Enum):
    MOMENTUM = "momentum"
    FUSION = "fusion"
    ARBITRAGE = "arbitrage"
    ALL = "all"


class AlertLevel(str, Enum):
    ALL = "all"
    TRADES_ONLY = "trades_only"
    ERRORS_ONLY = "errors_only"


class Config(BaseSettings):
    """Bot configuration — all values from .env or environment variables."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # --- Polymarket Credentials ---
    polymarket_private_key: str = Field(default="", description="Polygon private key")
    polymarket_proxy_address: str = Field(default="", description="Proxy/safe wallet")
    polymarket_api_key: str = Field(default="")
    polymarket_api_secret: str = Field(default="")
    polymarket_api_passphrase: str = Field(default="")

    # --- Trading ---
    max_trade_size: float = Field(default=1.0, ge=0.1, le=100.0)
    daily_loss_limit: float = Field(default=10.0, ge=1.0)
    max_trades_per_hour: int = Field(default=8, ge=1, le=60)
    strategy: Strategy = Field(default=Strategy.ALL)
    trading_mode: TradingMode = Field(default=TradingMode.PAPER)

    # --- Momentum Strategy ---
    entry_seconds_before_close: int = Field(default=30, ge=5, le=120)
    min_delta_threshold: float = Field(default=0.02, ge=0.001, le=1.0)
    max_token_price: float = Field(default=0.85, ge=0.50, le=0.99)

    # --- Arbitrage Strategy ---
    arb_threshold: float = Field(default=0.995, ge=0.95, le=1.0)
    arb_shares: int = Field(default=10, ge=1, le=1000)

    # --- Fusion Strategy ---
    weight_delta: float = Field(default=0.50)
    weight_order_flow: float = Field(default=0.20)
    weight_divergence: float = Field(default=0.20)
    weight_spread: float = Field(default=0.10)
    min_fusion_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    vol_multiplier_low: float = Field(default=1.15, ge=1.0, le=1.5)
    vol_multiplier_high: float = Field(default=0.70, ge=0.3, le=1.0)

    # --- Risk ---
    stop_loss_pct: float = Field(default=0.30, ge=0.05, le=0.50)
    take_profit_pct: float = Field(default=0.20, ge=0.05, le=0.50)
    loss_cooldown_seconds: int = Field(default=120, ge=0, le=600)

    # --- Paper Trading ---
    paper_slippage_pct: float = Field(
        default=0.5,
        ge=0.0,
        le=5.0,
        description="Simulated slippage as a percentage of price (0.0 = no slippage)",
    )

    # --- Data Sources ---
    binance_ws_url: str = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"
    binance_rest_url: str = "https://api.binance.com/api/v3"
    polymarket_clob_url: str = "https://clob.polymarket.com"
    polymarket_gamma_url: str = "https://gamma-api.polymarket.com"

    # --- Telegram ---
    telegram_bot_token: str = Field(default="")
    telegram_chat_id: str = Field(default="")
    telegram_alert_level: AlertLevel = Field(default=AlertLevel.ALL)

    # --- Monitoring ---
    prometheus_port: int = Field(default=8000)
    grafana_port: int = Field(default=3000)

    # --- General ---
    log_level: str = Field(default="INFO")

    @field_validator("weight_delta", "weight_order_flow", "weight_divergence", "weight_spread")
    @classmethod
    def weights_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Signal weights must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_fusion_weights_sum(self) -> "Config":
        weight_sum = (
            self.weight_delta
            + self.weight_order_flow
            + self.weight_divergence
            + self.weight_spread
        )
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(
                f"Fusion strategy weights must sum to 1.0, got {weight_sum:.4f} "
                f"(delta={self.weight_delta}, order_flow={self.weight_order_flow}, "
                f"divergence={self.weight_divergence}, spread={self.weight_spread})"
            )
        return self

    @property
    def fusion_weights_sum(self) -> float:
        return (
            self.weight_delta
            + self.weight_order_flow
            + self.weight_divergence
            + self.weight_spread
        )

    @property
    def has_polymarket_credentials(self) -> bool:
        return bool(
            self.polymarket_private_key
            and self.polymarket_api_key
            and self.polymarket_api_secret
            and self.polymarket_api_passphrase
        )

    @property
    def has_telegram(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def is_live(self) -> bool:
        return self.trading_mode == TradingMode.LIVE

    def validate_for_live_trading(self) -> list[str]:
        """Return list of issues that would prevent live trading."""
        issues = []
        if not self.has_polymarket_credentials:
            issues.append("Missing Polymarket API credentials")
        if not self.polymarket_proxy_address:
            issues.append("Missing proxy wallet address")
        return issues

    @property
    def has_plaintext_credentials(self) -> bool:
        """Check if credentials are loaded from a plaintext .env file."""
        env_path = Path(".env")
        return env_path.exists() and self.has_polymarket_credentials

    def __str__(self) -> str:
        return (
            f"Config(mode={self.trading_mode.value}, strategy={self.strategy.value}, "
            f"max_trade=${self.max_trade_size:.2f}, daily_limit=${self.daily_loss_limit:.2f})"
        )

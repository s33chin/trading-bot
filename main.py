#!/usr/bin/env python3
"""
Polymarket BTC 15-Minute Trading Bot — Entry Point

Usage:
    python main.py              # Paper trading (default)
    python main.py --live       # Live trading (REAL MONEY)
    python main.py --strategy momentum
    python main.py --strategy arbitrage
    python main.py --strategy fusion
    python main.py --strategy all
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from bot import TradingBot
from config import Config, Strategy, TradingMode
from logger import setup_logging, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Polymarket BTC 15-Minute Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable LIVE trading with real money (default: paper)",
    )
    parser.add_argument(
        "--strategy",
        choices=["momentum", "fusion", "arbitrage", "all"],
        default=None,
        help="Override strategy from .env",
    )
    parser.add_argument(
        "--max-trade",
        type=float,
        default=None,
        help="Override max trade size (USD)",
    )
    parser.add_argument(
        "--daily-limit",
        type=float,
        default=None,
        help="Override daily loss limit (USD)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Override log level",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config from .env
    config = Config()

    # Apply CLI overrides
    if args.live:
        config.trading_mode = TradingMode.LIVE
    if args.strategy:
        config.strategy = Strategy(args.strategy)
    if args.max_trade is not None:
        config.max_trade_size = args.max_trade
    if args.daily_limit is not None:
        config.daily_loss_limit = args.daily_limit
    if args.log_level:
        config.log_level = args.log_level

    setup_logging(config.log_level)
    log = get_logger("main")

    # Safety confirmation for live trading
    if config.is_live:
        print("\n" + "=" * 60)
        print("⚠️  WARNING: LIVE TRADING MODE")
        print("=" * 60)
        print(f"  Strategy:    {config.strategy.value}")
        print(f"  Max Trade:   ${config.max_trade_size:.2f}")
        print(f"  Daily Limit: ${config.daily_loss_limit:.2f}")
        print(f"  Entry:       {config.entry_seconds_before_close}s before close")
        print("=" * 60)

        issues = config.validate_for_live_trading()
        if issues:
            print("\n❌ Cannot start — configuration issues:")
            for issue in issues:
                print(f"   • {issue}")
            sys.exit(1)

        confirm = input("\nType 'YES' to start live trading: ")
        if confirm != "YES":
            print("Aborted.")
            sys.exit(0)
    else:
        print("\n📝 Starting in PAPER trading mode")
        print(f"   Strategy: {config.strategy.value}")
        print(f"   Max Trade: ${config.max_trade_size:.2f}\n")

    log.info("starting_bot", config=str(config))

    bot = TradingBot(config)
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()

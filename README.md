# Polymarket BTC 15-Minute Trading Bot

A production-grade trading bot for Polymarket's BTC 15-minute Up/Down prediction markets.

## Strategies

1. **Window Delta Momentum** — Primary signal. Trades based on BTC price movement relative to the 15-min window open.
2. **Multi-Signal Fusion** — Combines delta, order book imbalance, volatility regime, and momentum indicators.
3. **Arbitrage** — Buys both UP and DOWN when combined ask price < $1.00.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Data Feeds  │────▶│   Strategy   │────▶│  Execution   │
│  (Binance,   │     │  Engine      │     │  Engine      │
│  Polymarket) │     │  (3 modes)   │     │  (CLOB API)  │
└─────────────┘     └──────────────┘     └──────────────┘
       │                    │                     │
       ▼                    ▼                     ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Metrics     │────▶│   Risk       │────▶│  Alerts      │
│  (Prometheus)│     │   Manager    │     │  (Telegram)  │
└─────────────┘     └──────────────┘     └──────────────┘
```

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd polymarket-bot

# Option 1: Docker (recommended)
cp .env.example .env
# Edit .env with your credentials
docker-compose up -d

# Option 2: Local
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Configuration

Copy `.env.example` to `.env` and fill in your credentials. See `config.py` for all options.

## Monitoring

- Grafana dashboard at http://localhost:3000 (admin/admin)
- Prometheus metrics at http://localhost:9090
- Telegram alerts for trades, errors, and daily summaries

## Disclaimer

This bot trades with real money. Cryptocurrency and prediction markets carry significant risk.
Past performance does not guarantee future results. Use at your own risk.

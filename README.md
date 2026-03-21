# Polymarket BTC 15-Minute Trading Bot

A production-grade trading bot for Polymarket's BTC 15-minute Up/Down prediction markets.

## Strategies

1. **Window Delta Momentum** — Primary signal. Trades based on BTC price movement relative to the 15-min window open.
2. **Multi-Signal Fusion** — Combines delta, order book imbalance, volatility regime, and momentum indicators with configurable weights (must sum to 1.0).
3. **Arbitrage** — Buys both UP and DOWN when combined ask price < $1.00. Includes automatic leg cancellation if one side fails.

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

### 3-Phase Window Lifecycle

Each 15-minute market window is processed in three phases:

| Phase | Timing | What Happens |
|-------|--------|--------------|
| **Observe** | Window open → entry time | Collect BTC prices, poll order books, track delta trends, detect early arb opportunities |
| **Analyze & Execute** | Entry time → close - 3s | Evaluate strategies every ~2s, execute when confidence threshold is met |
| **Resolve** | Close → close + 5s | Wait for expiry, determine winner, settle P&L |

### Execution Engine

- **Live mode**: Places orders via py-clob-client on the Polymarket CLOB with fill verification polling (up to 30s timeout) and automatic cancellation of unfilled orders
- **Paper mode**: Simulates fills with configurable slippage (`PAPER_SLIPPAGE_PCT`, default 0.5%) for more realistic backtesting results
- **Arbitrage safety**: If one leg of an arb trade fails, the other leg is automatically cancelled to prevent orphaned positions

### Market Discovery

The bot discovers active BTC 15-min markets through three fallback approaches:
1. Slug-based lookup with multiple timestamp candidates
2. Gamma API `/markets` prefix search
3. Gamma API `/events` endpoint search

Each stage produces detailed diagnostic logs if discovery fails, making it easy to debug API format changes.

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd trading-bot

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

### CLI Options

```bash
python main.py --mode paper          # Paper trading (default)
python main.py --mode live           # Live trading
python main.py --strategy momentum   # Single strategy
python main.py --strategy all        # All strategies (default)
python main.py --max-trade 2.00      # Override max trade size
python main.py --log-level DEBUG     # Verbose logging
```

## Configuration

Copy `.env.example` to `.env` and fill in your credentials. All settings are documented there and validated at startup.

### Key Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `TRADING_MODE` | `paper` | `paper` or `live` |
| `STRATEGY` | `all` | `momentum`, `fusion`, `arbitrage`, or `all` |
| `MAX_TRADE_SIZE` | `1.00` | Maximum USD per trade |
| `DAILY_LOSS_LIMIT` | `10.00` | Circuit breaker — stops trading after this daily loss |
| `MAX_TRADES_PER_HOUR` | `8` | Trade frequency cap |
| `LOSS_COOLDOWN_SECONDS` | `120` | Pause after a losing trade |
| `PAPER_SLIPPAGE_PCT` | `0.5` | Simulated slippage % for paper trading (0.0 = perfect fills) |

### Fusion Strategy Weights

The fusion strategy combines four signals. Weights **must sum to 1.0** (validated at startup):

| Signal | Default Weight | Description |
|--------|---------------|-------------|
| `WEIGHT_DELTA` | `0.50` | BTC window price direction |
| `WEIGHT_ORDERBOOK` | `0.20` | Bid depth imbalance UP vs DOWN |
| `WEIGHT_VOLATILITY` | `0.15` | Confidence adjustment based on choppiness |
| `WEIGHT_MOMENTUM` | `0.15` | Last 5 klines trend |

### Risk Controls

The risk manager enforces multiple safety layers:

- **Daily loss limit**: Stops all trading when cumulative daily P&L hits the limit
- **Trade frequency cap**: Maximum trades per rolling hour
- **Post-loss cooldown**: Enforced pause after a losing trade
- **Consecutive loss scaling**: Position size reduced to 75% after 2 losses, 50% after 3+
- **Confidence-based sizing**: Higher signal confidence = larger position (up to max)

## Credentials & Security

**For paper trading**, no credentials are needed — the bot simulates all fills locally.

**For live trading**, you need Polymarket CLOB API credentials and a Polygon wallet. The bot will warn at startup if it detects private keys in a plaintext `.env` file.

For production deployments, use a secrets manager instead of `.env`:
- AWS Secrets Manager / SSM Parameter Store
- HashiCorp Vault
- Docker Secrets
- GCP Secret Manager

## Monitoring

- **Grafana dashboard** at http://localhost:3000 (admin/admin) — pre-provisioned with the polybot dashboard
- **Prometheus metrics** at http://localhost:8000 — BTC price, token asks, delta, win rate, trade counts, signal confidence, P&L histograms
- **Telegram alerts** for trades, signals, skips, resolutions, errors, and daily summaries

## Testing

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Project Structure

```
trading-bot/
├── main.py                  # CLI entry point
├── bot.py                   # Main orchestrator (3-phase loop)
├── config.py                # Pydantic config with validation
├── models.py                # Domain models (Market, Trade, Signal, etc.)
├── logger.py                # Structured JSON logging
├── data_feeds/
│   ├── binance_feed.py      # BTC price via WebSocket + REST fallback
│   └── polymarket_feed.py   # Market discovery + order books
├── strategies/
│   ├── momentum.py          # Window delta momentum
│   ├── fusion.py            # Multi-signal weighted voting
│   └── arbitrage.py         # Riskless arb detection
├── execution/
│   ├── engine.py            # Order placement, fill verification, paper sim
│   └── risk_manager.py      # Pre-trade risk checks + position sizing
├── alerts/
│   └── telegram_alerts.py   # Telegram bot notifications
├── monitoring/
│   ├── metrics.py           # Prometheus metrics exporter
│   ├── prometheus.yml       # Prometheus scrape config
│   └── grafana/             # Dashboard + provisioning
├── tests/
│   └── test_strategies.py   # Unit tests for strategies, risk, models
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Disclaimer

This bot trades with real money in live mode. Cryptocurrency and prediction markets carry significant risk. Past performance does not guarantee future results. Use at your own risk. Always start with paper trading to validate your configuration.

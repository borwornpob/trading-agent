# Gold Trading Agent

An ML-driven gold (XAUUSD) trading agent with regime-adaptive execution, S/R-based smart grid recovery, multi-timeframe ensemble prediction, and GenAI news sentiment analysis. Executes via MT5 demo bridge or cTrader API with a React + FastAPI dashboard for monitoring and live parameter tweaking.

Built on the [Machine Learning for Trading](https://github.com/stefan-jansen/machine-learning-for-trading) curriculum concepts (Chapters 02-24).

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │           Data Layer                     │
                        │  Yahoo OHLCV · News RSS · Broker Feed    │
                        └─────────────┬───────────────────────────┘
                                      │
                        ┌─────────────▼───────────────────────────┐
                        │         Feature Engine                    │
                        │  TA Indicators · S/R Features            │
                        │  Session Ranges · GenAI Sentiment        │
                        └─────────────┬───────────────────────────┘
                                      │
                        ┌─────────────▼───────────────────────────┐
                        │          Model Layer                     │
                        │  LightGBM (3 timeframes)                 │
                        │  GMM Regime · GARCH Vol · S/R Predictor  │
                        │  Ensemble Blender                        │
                        └─────────────┬───────────────────────────┘
                                      │
                        ┌─────────────▼───────────────────────────┐
                        │      Adaptive Strategy                    │
                        │  Trend Mode · Range/Grid Mode · Flat     │
                        │  (routed by regime detector)             │
                        └─────────────┬───────────────────────────┘
                                      │
                        ┌─────────────▼───────────────────────────┐
                        │        Risk & Governance                  │
                        │  Kill Switch · Grid Cap · Session Risk   │
                        │  Anti-Martingale Sizing · Dedup          │
                        └─────────────┬───────────────────────────┘
                                      │
                        ┌─────────────▼───────────────────────────┐
                        │         Execution                        │
                        │  MT5/cTrader Gateway · Position Manager  │
                        │  Market/Limit Orders · SL/TP             │
                        └─────────────────────────────────────────┘
```

## Strategy

### Three Trading Modes

The system detects market regime and routes to the appropriate execution mode:

**Trend Mode** (regime = trending_up / trending_down):
- Single entry on ensemble signal
- Trailing stop based on GARCH volatility
- No grid, no averaging down

**Range Mode with Smart Grid** (regime = ranging):
- Enter on primary signal
- If price moves against, place recovery orders at **predicted S/R levels** (not arbitrary)
- Max 3 grid levels, anti-martingale sizing: 1.0x, 0.7x, 0.5x
- Hard stop below predicted daily range
- Disabled during event risk (NFP, FOMC, CPI)

**Flat Mode** (regime = volatile / signals disagree):
- No new trades
- Close existing positions

### S/R Prediction Engine

The S/R model predicts daily high/low and bounce probability from:
- Previous day high/low/close/midpoint
- Swing points, volume profile, round numbers
- Classic pivot points (PP, R1, R2, S1, S2)
- ATR-based zone width

### Multi-Timeframe Ensemble

Three LightGBM models vote across timeframes:
- 15m: entry timing
- 4h: medium-term direction
- Daily: overall conviction

Majority vote required. Disagreement = flat.

### GenAI in the Loop

News headlines scored via GLM (OpenAI-compatible API):
- Sentiment: -1.0 to +1.0
- Confidence: 0.0 to 1.0
- Event risk flag: NFP, FOMC, CPI, rate decisions
- Acts as both feature and post-hoc filter

## Quick Start

### 1. Install dependencies

```bash
cd trading-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your broker and GenAI settings
```

For MT5 demo execution, clone the bridge and point `MT5_API_PATH` at it:

```bash
git clone https://github.com/borwornpob/mt5_api ../mt5_api
# In .env:
HARD_BROKER=mt5
MT5_API_PATH=../mt5_api
MT5_HOST=0.0.0.0
MT5_PORT=1111
MT5_VOLUME_LOTS=0.01
HARD_MAX_POSITION_VOLUME=0.05
HARD_MAX_ORDER_VOLUME=0.02
```

Then copy `../mt5_api/mql5/Experts/MT5SocketClient.mq5` into MT5, compile it in MetaEditor, attach the EA to a demo chart, and set its `ServerAddress` and `ServerPort` to the Python server.

For cTrader, use:

```bash
HARD_BROKER=ctrader
# Then fill CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET, CTRADER_ACCESS_TOKEN,
# CTRADER_ACCOUNT_ID, and CTRADER_SYMBOL_ID.
```

### 3. Fetch data and train

```bash
# Fetch gold data
python -m scripts.fetch_gold_data --start 2023-01-01 --end 2025-01-01

# Train all timeframe models, including the 15m model used for MT5 execution
python -m scripts.train_all_models

# Or train a single primary model
python -m scripts.train_model

# Train S/R prediction model
python -m scripts.train_sr_model
```

### 4. Backtest

```bash
python -m scripts.run_backtest
```

### 5. Run in shadow mode (default)

```bash
# Single cycle
python -m scripts.run_live

# Continuous loop (5 min intervals)
python -m scripts.run_live --continuous --interval 300
```

### 6. Dashboard

```bash
# Terminal 1: Start API server
uvicorn api.app:app --reload --port 8000

# Terminal 2: Start React dev server
cd dashboard && npm install && npm run dev
```

Open http://localhost:3000

## Project Structure

```
trading-agent/
  agent/
    config.py                # Environment-backed configuration
    mt5_gateway.py           # MT5 socket bridge adapter for demo/live execution
    ctrader_gateway.py       # cTrader OpenAPI with limit orders, SL/TP, close
    autonomous_loop.py       # Main perceive → infer → govern → execute cycle
    data/
      pipeline.py            # OHLCV fetch, TA indicators, triple-barrier labels
      news_fetcher.py        # RSS feed collection with dedup
      genai_sentiment.py     # GLM sentiment scoring via OpenAI-compatible API
      sr_features.py         # S/R features: pivots, swing points, round numbers
      session_analyzer.py    # Asian/London/NY session detection
    models/
      lgbm_model.py          # LightGBM training, prediction, persistence
      ensemble.py             # Multi-model + multi-timeframe blending
      regime.py               # GMM regime detection (4 regimes)
      volatility.py           # GARCH(1,1) volatility forecasting
      sr_predictor.py         # Daily high/low + bounce probability prediction
      mtf_signal.py           # Multi-timeframe signal aggregator
    strategy/
      gold_strategy.py       # Signal → TradeSignal mapping
      adaptive_executor.py   # Regime router: trend / range_grid / flat
      smart_grid.py           # S/R-based grid recovery engine
      trend_executor.py      # Trend mode with trailing stop
      signal_gates.py         # Statistical acceptance gates
    risk/
      risk_guard.py           # Kill switch, daily loss, position limits
      position_manager.py    # Target → order intents, reconciliation
      grid_risk.py            # Grid exposure cap, recovery sizing
      session_risk.py         # Session-aware position sizing
    backtest/
      engine.py               # Walk-forward backtest engine
      metrics.py              # Sharpe, drawdown, bootstrap gates
      grid_simulator.py       # Smart grid backtest simulation
  api/
    app.py                    # FastAPI application
    routes/
      dashboard.py            # Dashboard + refresh + regime + news endpoints
      backtest.py             # Backtest runner endpoint
      config_tweak.py         # Live parameter tweaking
      orders.py               # Order history + grid status
    ws.py                     # WebSocket real-time updates
  dashboard/
    src/                      # React + TypeScript + Recharts
  scripts/
    fetch_gold_data.py        # Fetch and build training dataset
    train_model.py            # Train primary LightGBM model
    train_sr_model.py         # Train S/R prediction model
    run_backtest.py           # Run holdout backtest
    run_live.py               # Run live trading cycle
  tests/                      # Full test suite (12 test files)
```

## ML4T Curriculum Mapping

| Chapter | Concept | Implementation |
|---------|---------|----------------|
| Ch 02 | Market data, bar construction | `data/pipeline.py` |
| Ch 03 | Alternative data scoring | `data/news_fetcher.py` |
| Ch 04 | Alpha factors, TA indicators | `data/pipeline.py`, `data/sr_features.py` |
| Ch 05 | Strategy evaluation, risk metrics | `backtest/metrics.py`, `risk/risk_guard.py` |
| Ch 06 | Walk-forward CV, embargo | `backtest/engine.py` |
| Ch 09 | GARCH volatility | `models/volatility.py` |
| Ch 11 | Ensemble methods | `models/ensemble.py` |
| Ch 12 | LightGBM, gradient boosting | `models/lgbm_model.py` |
| Ch 13 | Regime detection (GMM) | `models/regime.py` |
| Ch 14 | NLP, sentiment analysis | `data/genai_sentiment.py` |
| Ch 23 | Kill switches, monitoring | `risk/risk_guard.py`, `risk/grid_risk.py` |

## Risk Controls

- **Kill switch**: Manual + automatic (max daily loss hit)
- **Shadow mode**: Default on — logs decisions without submitting orders
- **Grid exposure cap**: Total grid size limited to 2x initial position
- **Anti-martingale**: Grid sizing decreases (1.0, 0.7, 0.5), never increases
- **Recovery mode**: After 3 consecutive losses, position size halved
- **Session risk**: Position size reduced outside London/NY overlap
- **Event risk**: Grid disabled when GenAI flags NFP/FOMC/CPI
- **Dedup window**: Prevents duplicate orders within 60 seconds

## Warning

This is an educational/research project. Live trading involves significant financial risk. Always start in demo/shadow mode. Past backtest performance does not guarantee future results.

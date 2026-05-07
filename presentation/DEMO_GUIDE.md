# 🎬 Gold Trading Agent — Demo Guide

> **Step-by-Step Walkthrough for Presenting the XAU/USD ML Trading System**
> Regime-Adaptive Execution · S/R Smart Grid · Multi-TF Ensemble · GenAI Sentiment

---

## 📋 Table of Contents

1. [Prerequisites](#-prerequisites)
2. [Environment Setup](#-environment-setup)
3. [Starting the System](#-starting-the-system)
4. [Running a Trading Cycle](#-running-a-trading-cycle)
5. [Dashboard Panel Walkthrough](#-dashboard-panel-walkthrough)
6. [GenAI Sentiment Deep-Dive](#-genai-sentiment-deep-dive)
7. [Backtest Walkthrough](#-backtest-walkthrough)
8. [Screenshots & Capture Guide](#-screenshots--capture-guide)
9. [Talking Points Cheat Sheet](#-talking-points-cheat-sheet)
10. [Troubleshooting](#-troubleshooting)

---

## ✅ Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.11+ | 3.12 |
| **Node.js** | 18+ | 20 LTS |
| **RAM** | 4 GB | 8 GB |
| **Disk** | 500 MB | 1 GB |
| **Internet** | Stable (for API calls) | Broadband |

### Accounts & API Keys

| Service | Purpose | Required? | Get It |
|---------|---------|-----------|--------|
| **ZhipuAI (GLM-4)** | GenAI sentiment analysis | ✅ Yes | [open.bigmodel.cn](https://open.bigmodel.cn) |
| **cTrader Open API** | Broker execution (demo) | ⚠️ Optional (paper mode available) | [ctrader.com](https://ctrader.com) |
| **Yahoo Finance** | Market data (free) | ❌ No key needed | Built-in via `yfinance` |

### Install Dependencies

```bash
# Clone / navigate to the project
cd machine-learning-for-trading/trading-agent

# ──────────────────────────────────
# Python backend
# ──────────────────────────────────
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

```bash
# ──────────────────────────────────
# React dashboard
# ──────────────────────────────────
cd dashboard
npm install
cd ..
```

---

## 🔧 Environment Setup

### Step 1: Create `.env` file

```bash
cp .env.example .env
```

### Step 2: Edit `.env` with your keys

```
# ═══════════════════════════════════════════
# 🔑  API KEYS — Gold Trading Agent
# ═══════════════════════════════════════════

# GenAI Sentiment (ZhipuAI GLM-4)
ZHIPUAI_API_KEY=zhp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# cTrader Open API (optional — leave blank for paper mode)
CTRADER_CLIENT_ID=
CTRADER_CLIENT_SECRET=
CTRADER_ACCESS_TOKEN=
CTRADER_REFRESH_TOKEN=

# Trading Mode
TRADING_MODE=paper                  # paper | live
CTRADER_BROKER_NAME= Pepperstone    # or your broker

# Dashboard
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
REACT_PORT=3000

# Logging
LOG_LEVEL=INFO
```

> ⚠️ **SECURITY**: Never commit `.env` to version control. It is already in `.gitignore`.

### Step 3: Verify Installation

```bash
# Quick health check
python -c "
import lightgbm;    print(f'✅ LightGBM  {lightgbm.__version__}')
import sklearn;     print(f'✅ sklearn   {sklearn.__version__}')
import pandas;      print(f'✅ pandas    {pandas.__version__}')
import fastapi;     print(f'✅ FastAPI   {fastapi.__version__}')
import feedparser;  print(f'✅ feedparser OK')
from zhipuai import ZhipuAI; print('✅ ZhipuAI SDK OK')
print('\n🎉 All dependencies installed!')
"
```

---

## 🚀 Starting the System

### Overview — Start Order

```
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │   START ORDER                                               │
  │                                                             │
  │   1️⃣  API Server (FastAPI)        ← :8000                  │
  │         ↓                                                   │
  │   2️⃣  React Dashboard             ← :3000                  │
  │         ↓                                                   │
  │   3️⃣  Run a Trading Cycle         ← POST /api/trade/cycle  │
  │         ↓                                                   │
  │   4️⃣  Watch the dashboard update  ← WebSocket real-time    │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
```

### Step 1: Start the API Server

```bash
# From the project root
# Ensure venv is activated
source .venv/bin/activate

# Start FastAPI with uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Expected output:**

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345]
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Verify:**

```bash
curl http://localhost:8000/api/health
# Expected: {"status":"healthy","mode":"paper","models_loaded":true}
```

> 📸 **SCREENSHOT**: Capture the terminal showing the server running.

### Step 2: Start the React Dashboard

Open a **new terminal**:

```bash
cd machine-learning-for-trading/trading-agent/dashboard

npm run dev
```

**Expected output:**

```
  VITE v5.x.x  ready in 300 ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: http://192.168.x.x:3000/
```

Open your browser → **http://localhost:3000**

> 📸 **SCREENSHOT**: Capture the dashboard loading in the browser.

### Step 3: Confirm WebSocket Connection

Look at the bottom-right of the dashboard:

```
  🟢 Connected — Receiving real-time updates
```

If you see:

| Status | Meaning | Action |
|--------|---------|--------|
| 🟢 Connected | WebSocket active | ✅ Proceed |
| 🟡 Reconnecting… | API server not ready | Wait / restart API |
| 🔴 Disconnected | API server is down | Start API server first |

---

## 🔄 Running a Trading Cycle

### What is a "Cycle"?

```
  ┌──────────────────────────────────────────────────────────┐
  │              ONE TRADING CYCLE (≈ 3–8 seconds)           │
  │                                                          │
  │   👁 PERCEIVE    Fetch OHLCV, news, account state        │
  │        ↓                                                 │
  │   🧠 INFER      Run models → signal + regime + vol      │
  │        ↓                                                 │
  │   ⚖️ GOVERN      Risk checks → approve / reject          │
  │        ↓                                                 │
  │   ⚡ EXECUTE     Send order (or log rejection)           │
  │                                                          │
  └──────────────────────────────────────────────────────────┘
```

### Trigger a Cycle via API

```bash
# Option A: Single cycle (one-shot)
curl -X POST http://localhost:8000/api/trade/cycle \
  -H "Content-Type: application/json" | python -m json.tool
```

**Expected response:**

```json
{
  "cycle_id": "cyc_20250115_143022",
  "timestamp": "2025-01-15T14:30:22Z",
  "perceive": {
    "ohlcv_bars_fetched": 1200,
    "news_articles_analyzed": 8,
    "account_equity": 10542.30
  },
  "infer": {
    "signal": "BUY",
    "confidence": 0.72,
    "regime": "TRENDING_UP",
    "volatility": "MEDIUM",
    "sentiment_score": 0.45
  },
  "govern": {
    "approved": true,
    "mode": "TREND",
    "risk_flags": [],
    "position_size": 0.05
  },
  "execute": {
    "action": "PLACED",
    "order_id": "ORD-12345",
    "entry": 2345.20,
    "sl": 2338.00,
    "tp": 2360.00,
    "slippage": 0.01
  }
}
```

### Trigger via Dashboard

Click the **"▶ Run Cycle"** button in the **Agent Loop** panel.

> 📸 **SCREENSHOT**: Capture the API response and the dashboard updating.

### Trigger Continuous Mode

```bash
# Option B: Continuous mode (auto-cycle every 5 min)
curl -X POST http://localhost:8000/api/trade/start \
  -H "Content-Type: application/json" \
  -d '{"interval_seconds": 300}'
```

To stop:

```bash
curl -X POST http://localhost:8000/api/trade/stop
```

---

## 🖥️ Dashboard Panel Walkthrough

The dashboard has **8 panels** arranged in a responsive grid:

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                        📊 GOLD TRADING AGENT                         │
  │                                                                      │
  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐│
  │  │ 📈 Signal   │  │ 🧪 Senti-  │  │ ⚖️ Risk     │  │ 🌀 Regime   ││
  │  │    Panel    │  │   ment      │  │   Panel     │  │   Panel     ││
  │  │             │  │   Panel     │  │             │  │             ││
  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘│
  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────────┐│
  │  │ 📊 Backtest │  │ ⚙️ Config   │  │ 🔄 Agent Loop Status        ││
  │  │    Panel    │  │   Panel     │  │                              ││
  │  │             │  │             │  │                              ││
  │  └─────────────┘  └─────────────┘  └──────────────────────────────┘│
  └──────────────────────────────────────────────────────────────────────┘
```

---

### 📈 Panel 1: Signal Panel

**What it shows:**

| Element | Description |
|---------|-------------|
| **Signal Direction** | `BUY` / `SELL` / `FLAT` with color coding (🟢/🔴/⚪) |
| **Confidence Gauge** | Semicircular gauge showing confidence 0–100% |
| **Source Timeframe** | Which TF dominates: `M5`, `M15`, or `H1` |
| **Signal Breakdown** | Individual model scores: `LGB_M5: +0.4`, `LGB_M15: +0.6`, `LGB_H1: +0.8` |
| **Ensemble Weight** | Visualization of the blending weights |
| **Last Updated** | Timestamp of the latest cycle |

**What to point out:**

> 🗣️ *"The signal panel shows the ensemble output of three LightGBM models, each trained on a different timeframe. The blender uses weights `[0.2, 0.3, 0.5]` to favor the higher timeframe for directional accuracy while keeping the lower timeframes for timing precision."*

> 📸 **SCREENSHOT**: Capture the Signal Panel showing a BUY signal with confidence gauge at ~72%.

---

### 🧪 Panel 2: Sentiment Panel

**What it shows:**

| Element | Description |
|---------|-------------|
| **Sentiment Score** | Large number display: `-1.0` to `+1.0` |
| **Sentiment Badge** | 🔴 BEARISH / ⚪ NEUTRAL / 🟢 BULLISH |
| **Trend Sparkline** | Mini chart of sentiment score over last 24h |
| **Key Topics** | Tags: `Fed Policy`, `Geopolitical Risk`, `Inflation` |
| **Fed Signal** | 🦅 Hawkish / 🕊️ Dovish / ➖ Neutral |
| **Analyzed Articles** | Count of articles processed in current window |
| **GLM-4 Status** | API call latency, last call timestamp |

**What to point out:**

> 🗣️ *"This is the GenAI integration. We feed financial news articles from RSS feeds into the GLM-4 large language model. It returns a structured JSON with sentiment score, confidence, and key topics. Notice the sentiment is used both as a **feature** in the LightGBM models AND as a **filter** — if the sentiment confidence is very low, we reduce position size."*

> 🗣️ *"The Fed policy signal is extracted separately. A hawkish stance is typically bearish for gold, dovish is bullish. This gives us an edge that pure technical models cannot capture."*

> 📸 **SCREENSHOT**: Capture the Sentiment Panel showing a BULLISH score with Fed signal and topic tags.

---

### ⚖️ Panel 3: Risk Panel

**What it shows:**

| Element | Description |
|---------|-------------|
| **Kill Switch** | 🟢 Active / 🔴 Triggered — with current drawdown % |
| **Grid Exposure** | Bar chart: `0/4` to `4/4` grid levels used |
| **Session Window** | Current session: `🏔 Asia` / `🏛️ London` / `🗽 New York` / `🌑 Off-hours` |
| **Risk Flags** | Active warnings: `⚠️ NFP in 15min`, `⚠️ Wide spread` |
| **Account Health** | Equity, margin usage %, floating P&L |
| **Position Count** | Current open positions vs max allowed |

**What to point out:**

> 🗣️ *"The risk panel is the heart of the GOVERN phase. The kill switch is the last line of defense — if drawdown hits 10%, all trading halts. The grid cap ensures we never exceed 4 recovery levels. Session risk blocks trades during high-impact news events like NFP or FOMC decisions."*

> 🗣️ *"Notice the anti-martingale indicator. After a losing trade, the system halves the position size automatically. This prevents emotional revenge trading — the algorithm enforces discipline."*

> 📸 **SCREENSHOT**: Capture the Risk Panel showing green kill switch, 1/4 grid, and London session active.

---

### 🌀 Panel 4: Regime Panel

**What it shows:**

| Element | Description |
|---------|-------------|
| **Current Regime** | Large label: `TRENDING_UP` / `TRENDING_DOWN` / `RANGING` / `FLAT` |
| **Regime Probability** | Pie chart or bar showing GMM state probabilities |
| **Transition Matrix** | Probabilities of switching to each regime |
| **Regime History** | Mini timeline of regime labels over last 24h |
| **Strategy Mode** | Active mode badge: `📈 TREND` / `📊 GRID` / `😐 FLAT` |

**What to point out:**

> 🗣️ *"The regime detector uses a 3-state Gaussian Mixture Model trained on returns and volatility features. This is crucial because the same signal means different things in different regimes — a breakout in TREND mode is a continuation signal, but in RANGE mode, it's likely a false breakout and we should expect a reversal to the mean."*

> 🗣️ *"The strategy router maps regime directly to execution mode. TREND → single directional trade with trailing stop. RANGE → smart grid recovery around S/R levels. FLAT → no trade, preserve capital."*

> 📸 **SCREENSHOT**: Capture the Regime Panel showing TRENDING_UP with probability bars.

---

### 🔄 Panel 5: Agent Loop Status

**What it shows:**

| Element | Description |
|---------|-------------|
| **Current Phase** | Animated indicator cycling through the 4 phases |
| **Phase Progress** | Step indicator: `👁 PERCEIVE → 🧠 INFER → ⚖️ GOVERN → ⚡ EXECUTE` |
| **Cycle Counter** | Number of cycles completed this session |
| **Last Cycle Time** | Duration of the most recent cycle (ms) |
| **Phase Details** | Expandable log showing what happened in each phase |
| **Run Cycle Button** | Manual trigger button: **"▶ Run Cycle"** |

**What to point out:**

> 🗣️ *"This panel shows the agent loop in real-time. When you click 'Run Cycle', you'll see the phase indicator animate through PERCEIVE → INFER → GOVERN → EXECUTE. Each phase logs its results — you can expand to see exactly what data was fetched, what the models predicted, what risk checks passed, and what order was placed."*

> 🗣️ *"The cycle typically completes in 3–8 seconds. The bottleneck is usually the GLM-4 sentiment API call. We cache results for 30 minutes to avoid unnecessary latency."*

> 📸 **SCREENSHOT**: Capture the Agent Loop panel mid-cycle showing the INFER phase highlighted.

---

### 📊 Panel 6: Backtest Panel

**What it shows:**

| Element | Description |
|---------|-------------|
| **Equity Curve** | Line chart of equity over time |
| **Drawdown Chart** | Underwater equity curve (max DD highlighted) |
| **Trade Markers** | Entry/exit points on the equity curve |
| **Performance Metrics** | Table with key stats (see below) |
| **Time Range Selector** | Date picker for backtest period |
| **Strategy Comparison** | Overlay trend-only vs grid-only vs combined |

**Performance Metrics Table:**

| Metric | Description |
|--------|-------------|
| **Total Return** | Cumulative P&L % |
| **Sharpe Ratio** | Risk-adjusted return |
| **Max Drawdown** | Worst peak-to-trough decline |
| **Win Rate** | % of profitable trades |
| **Profit Factor** | Gross profit / gross loss |
| **Avg Win / Avg Loss** | Reward-to-risk per trade |
| **Total Trades** | Number of round-trip trades |
| **Regime Accuracy** | % correct regime classification |

**What to point out:**

> 🗣️ *"The backtest panel shows historical performance with realistic assumptions — slippage, commissions, and realistic fill modeling. Notice how the regime-adaptive approach outperforms a static strategy — the equity curve has smaller drawdowns because the system switches to FLAT mode during choppy markets."*

> 🗣️ *"The drawdown chart is especially important. Our maximum drawdown target is under 10%, enforced by the kill switch. If we see the backtest hitting that threshold, we know we need to tune the risk parameters."*

> 📸 **SCREENSHOT**: Capture the Backtest Panel with equity curve, drawdown chart, and metrics table.

---

### ⚙️ Panel 7: Config Panel

**What it shows:**

| Element | Description |
|---------|-------------|
| **Model Config** | Ensemble weights, confidence thresholds, feature toggles |
| **Risk Config** | Kill switch %, grid cap, max positions, risk per trade |
| **Sentiment Config** | GLM-4 model selection, cache TTL, RSS feed URLs |
| **Session Config** | Trading hours, blocked events, session definitions |
| **Save / Reset** | Buttons to persist or revert config changes |

**What to point out:**

> 🗣️ *"All configuration is centralized and hot-reloadable. You can adjust the ensemble weights in real-time and see how the signal changes on the next cycle. Risk parameters require a confirmation step to prevent accidental changes."*

> 📸 **SCREENSHOT**: Capture the Config Panel showing risk parameters.

---

## 🤖 GenAI Sentiment Deep-Dive

### Why GenAI Matters for Gold Trading

```
  ┌────────────────────────────────────────────────────────────────┐
  │                                                                │
  │   🥇 GOLD is uniquely sensitive to NEWS & MACRO events:       │
  │                                                                │
  │   • Fed interest rate decisions                                │
  │   • Inflation data (CPI, PCE)                                  │
  │   • Geopolitical tensions (Middle East, Ukraine)               │
  │   • Central bank gold purchases                                │
  │   • Dollar strength (DXY)                                      │
  │                                                                │
  │   Traditional TA indicators CANNOT capture these signals.      │
  │   GenAI (GLM-4) bridges this gap.                             │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```

### Live Demo: Sentiment Analysis

#### Step 1: Show the RSS Feed Collection

```bash
# Call the sentiment endpoints
curl http://localhost:8000/api/sentiment/articles | python -m json.tool
```

**Expected output:**

```json
{
  "articles": [
    {
      "title": "Gold Rises as Fed Signals Potential Rate Cuts in 2025",
      "source": "Reuters",
      "published": "2025-01-15T12:30:00Z",
      "sentiment": {
        "score": 0.65,
        "confidence": 0.88,
        "key_topics": ["Fed Policy", "Interest Rates", "Rate Cuts"],
        "fed_policy_signal": "dovish",
        "geopolitical_risk": 0.2,
        "impact_horizon": "medium",
        "summary": "Gold prices rise as the Federal Reserve hints at potential rate cuts, weakening the dollar and boosting bullion appeal."
      }
    },
    {
      "title": "Middle East Tensions Escalate, Safe-Haven Demand Surges",
      "source": "Bloomberg",
      "published": "2025-01-15T10:15:00Z",
      "sentiment": {
        "score": 0.82,
        "confidence": 0.91,
        "key_topics": ["Geopolitical Risk", "Safe Haven", "Middle East"],
        "fed_policy_signal": "neutral",
        "geopolitical_risk": 0.85,
        "impact_horizon": "short",
        "summary": "Escalating tensions drive investors toward gold as a safe-haven asset."
      }
    }
  ],
  "composite_score": 0.73,
  "article_count": 8
}
```

#### Step 2: Show the GLM-4 Prompt

Point out the structured prompt design in the dashboard or code:

```
  ┌──────────────────────────────────────────────────────────┐
  │  📝 GLM-4 PROMPT TEMPLATE                               │
  │                                                          │
  │  System: You are a financial sentiment analyzer          │
  │          specializing in gold/XAUUSD markets.            │
  │          Return ONLY valid JSON.                         │
  │                                                          │
  │  User:   Analyze this gold market news article:          │
  │          Title: {title}                                  │
  │          Body: {body}                                    │
  │          Date: {timestamp}                               │
  │                                                          │
  │          Return JSON: {                                  │
  │            "sentiment_score": <-1.0 to 1.0>,             │
  │            "confidence": <0.0 to 1.0>,                   │
  │            "key_topics": [<str>...],                     │
  │            "fed_policy_signal": "hawkish|dovish|neutral", │
  │            "geopolitical_risk": <0.0 to 1.0>,            │
  │            "impact_horizon": "short|medium|long"         │
  │          }                                               │
  └──────────────────────────────────────────────────────────┘
```

#### Step 3: Explain the Dual Usage

```
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │   Sentiment Score is used in TWO ways:                 │
  │                                                         │
  │   ┌───────────────────┐     ┌────────────────────┐     │
  │   │  🔧 As a FEATURE   │     │  🛡️ As a FILTER    │     │
  │   │                   │     │                    │     │
  │   │  Fed into LightGBM│     │  If confidence     │     │
  │   │  as an input      │     │  < 0.3 → skip     │     │
  │   │  column. The ML   │     │  trade entirely    │     │
  │   │  model learns     │     │                    │     │
  │   │  the nonlinear    │     │  If |score| > 0.8  │     │
  │   │  relationship     │     │  → extreme event,  │     │
  │   │  between news     │     │  reduce size 50%   │     │
  │   │  and price.       │     │                    │     │
  │   └───────────────────┘     └────────────────────┘     │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
```

### Key Talking Points for GenAI Feature

> 🗣️ **Talking Point 1 — Why GenAI over traditional NLP:**
> *"Traditional sentiment analysis uses Bag-of-Words or VADER, which struggle with financial nuance. For example, 'Fed signals patience on rates' is mildly dovish, but a simple keyword search wouldn't catch that. GLM-4 understands context, tone, and financial domain knowledge."*

> 🗣️ **Talking Point 2 — Structured output:**
> *"We force the LLM to return structured JSON. This is critical for production systems — we can't parse free-text reliably. The schema ensures we always get a valid sentiment score, confidence level, and topic tags."*

> 🗣️ **Talking Point 3 — Latency management:**
> *"GLM-4 calls add ~2–4 seconds per cycle. We mitigate this with a 30-minute TTL cache. News sentiment doesn't change every 5 minutes, so caching is appropriate. Fresh articles trigger a new API call; otherwise we use the cached score."*

> 🗣️ **Talking Point 4 — Cost efficiency:**
> *"At ~8 articles per fetch and 4 fetches per day, we make roughly 32 GLM-4 API calls daily. With GLM-4-Flash pricing, this costs less than $0.10/day — a trivial cost for the information edge it provides."*

> 🗣️ **Talking Point 5 — Human-in-the-loop potential:**
> *"The sentiment panel serves as an interpretability layer. A human trader can verify the GenAI's assessment and override if needed. This is the future of AI-assisted trading — not fully autonomous, but AI-augmented."*

---

## 📊 Backtest Walkthrough

### Running a Backtest

```bash
# Trigger backtest via API
curl -X POST http://localhost:8000/api/backtest/run \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2024-06-01",
    "end_date": "2024-12-31",
    "initial_capital": 10000,
    "strategy": "ensemble"
  }' | python -m json.tool
```

**Expected response:**

```json
{
  "backtest_id": "bt_20250115_143500",
  "period": "2024-06-01 to 2024-12-31",
  "results": {
    "total_return_pct": 18.4,
    "sharpe_ratio": 1.65,
    "max_drawdown_pct": 7.2,
    "win_rate_pct": 58.3,
    "profit_factor": 1.89,
    "total_trades": 142,
    "avg_win_pips": 28.5,
    "avg_loss_pips": -15.2,
    "best_trade_pips": 145.0,
    "worst_trade_pips": -42.0,
    "regime_accuracy_pct": 71.2
  }
}
```

### What to Highlight in the Backtest

| Point | What to Show | Why It Matters |
|-------|-------------|----------------|
| **Equity curve smoothness** | The line should trend upward with controlled pullbacks | Demonstrates risk-managed approach |
| **Max drawdown < 10%** | The underwater chart should stay above the -10% line | Validates the kill switch |
| **Regime transitions** | Color-coded segments on the equity curve | Shows the strategy adapts to market conditions |
| **Sentiment impact** | Compare backtest with/without sentiment feature | Quantifies the GenAI edge |
| **Grid recovery trades** | Mark grid trades differently from trend trades | Shows the S/R recovery mechanism |

> 📸 **SCREENSHOT**: Capture the backtest equity curve with drawdown overlay.

---

## 📸 Screenshots & Capture Guide

### Required Screenshots for Presentation

| # | Screenshot | What to Capture | File Name |
|---|-----------|-----------------|-----------|
| 1 | **Full Dashboard** | All 8 panels loaded | `01_full_dashboard.png` |
| 2 | **Signal Panel** | BUY signal with confidence gauge | `02_signal_panel.png` |
| 3 | **Sentiment Panel** | BULLISH score with topics | `03_sentiment_panel.png` |
| 4 | **Risk Panel** | Green status, session active | `04_risk_panel.png` |
| 5 | **Regime Panel** | TRENDING_UP state | `05_regime_panel.png` |
| 6 | **Agent Loop** | Mid-cycle phase indicator | `06_agent_loop.png` |
| 7 | **Backtest Results** | Equity curve + drawdown | `07_backtest_results.png` |
| 8 | **Config Panel** | Risk parameters view | `08_config_panel.png` |
| 9 | **Terminal — API** | Server running with logs | `09_terminal_api.png` |
| 10 | **Terminal — Cycle** | Cycle JSON response | `10_terminal_cycle.png` |
| 11 | **Architecture Diagram** | From ARCHITECTURE.md mermaid | `11_architecture_diagram.png` |
| 12 | **System Loop** | From SYSTEM_LOOP.md | `12_system_loop.png` |

### Screenshot Tips

```
  ┌─────────────────────────────────────────────────────────┐
  │  📸 SCREENSHOT BEST PRACTICES                          │
  │                                                         │
  │  • Use 1440×900 or 1920×1080 resolution                │
  │  • Hide personal bookmarks / browser tabs              │
  │  • Use a clean browser profile (no extensions)         │
  │  • Dark mode preferred (matches the trading theme)     │
  │  • Crop to the relevant panel (don't show empty space) │
  │  • Annotate key areas with arrows or boxes             │
  │  • Save as PNG (not JPG) for crisp text                │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
```

---

## 🗣️ Talking Points Cheat Sheet

### 30-Second Elevator Pitch

> *"We built an autonomous gold trading agent that uses machine learning for price prediction, regime detection for strategy adaptation, and GenAI for news sentiment analysis. The system runs a continuous PERCEIVE-INFER-GOVERN-EXECUTE loop, with multiple risk controls including a kill switch, grid cap, and anti-martingale sizing."*

### 5-Minute Presentation Flow

| Time | Section | Key Message |
|------|---------|-------------|
| 0:00–0:30 | **Intro** | Gold trading system with ML + GenAI |
| 0:30–1:00 | **Architecture** | Show the 4-phase loop diagram |
| 1:00–2:00 | **Demo: Cycle** | Run a live cycle, show dashboard update |
| 2:00–3:00 | **Demo: Sentiment** | Show GenAI analyzing news in real-time |
| 3:00–4:00 | **Demo: Backtest** | Show historical performance |
| 4:00–4:30 | **Risk Controls** | Emphasize safety mechanisms |
| 4:30–5:00 | **Q&A** | Open floor |

### Key Differentiators

| # | Differentiator | Why It Matters |
|---|---------------|----------------|
| 1️⃣ | **Regime-adaptive** | Same signal ≠ same action in different markets |
| 2️⃣ | **GenAI sentiment** | Captures macro context that TA cannot |
| 3️⃣ | **Smart grid recovery** | Turns losing trades into recovery opportunities |
| 4️⃣ | **Multi-timeframe ensemble** | Combines short-term timing with directional bias |
| 5️⃣ | **5-layer risk defense** | Kill switch → Grid cap → Session → Anti-Mart → Dedup |

### Expected Questions & Answers

**Q: What happens when the GLM-4 API is down?**

> A: The system degrades gracefully. If the sentiment API is unavailable, we use the last cached score (TTL 30 min). If no cache exists, sentiment features are set to neutral (0.0), and the models fall back to technical-only predictions. A warning flag appears on the dashboard.

**Q: How do you prevent overfitting in the LightGBM models?**

> A: Three mechanisms: (1) Purged K-Fold cross-validation with embargo to prevent data leakage, (2) strict feature importance analysis with SHAP values to remove noise features, (3) walk-forward validation on out-of-sample data. We also limit tree depth and use regularization.

**Q: What's the latency of a full cycle?**

> A: 3–8 seconds. Feature engineering takes ~1s, model inference ~0.5s, risk checks ~0.1s, GLM-4 API ~2–4s (cached), order execution ~0.5s. The GLM-4 call is the bottleneck, which is why we cache aggressively.

**Q: Is this safe for live trading?**

> A: The system defaults to paper mode. Live mode requires explicit configuration and manual enablement. Even in live mode, the kill switch enforces a maximum 10% drawdown, and all risk parameters have hardcoded floors. We recommend at least 3 months of paper trading before going live.

**Q: How does the smart grid recovery work?**

> A: In RANGE mode, if the initial trade goes against us, instead of a single stop-loss, we place up to 4 grid levels between the current price and the nearest S/R level. Each level has a reduced lot size (anti-martingale). When price reverts to the mean, all grid legs close at a net profit. The grid cap prevents unlimited exposure.

---

## 🔧 Troubleshooting

### Common Issues

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `❌ ZHIPUAI_API_KEY not set` | Missing `.env` file | `cp .env.example .env` and add your key |
| `🔌 WebSocket Disconnected` | API server not running | Start FastAPI: `uvicorn api.main:app` |
| `📊 No OHLCV data` | Yahoo Finance timeout | Check internet; wait and retry |
| `🤖 GLM-4 rate limit` | Too many API calls | Increase `SENTIMENT_CACHE_TTL` in config |
| `⚛️ Dashboard blank` | npm not installed | `cd dashboard && npm install` |
| `🚫 Port 8000 in use` | Another process using port | `lsof -ti:8000 \| xargs kill` |
| `⚠️ Model not found` | Models not trained yet | Run `python scripts/train_models.py` |

### Health Check Commands

```bash
# Check all services
curl http://localhost:8000/api/health | python -m json.tool

# Check sentiment pipeline
curl http://localhost:8000/api/sentiment/current | python -m json.tool

# Check regime detector
curl http://localhost:8000/api/regime/current | python -m json.tool

# Check risk engine
curl http://localhost:8000/api/risk/status | python -m json.tool
```

### Reset Everything

```bash
# Nuclear option — start fresh
pkill -f uvicorn
pkill -f vite

# Clear caches
rm -rf __pycache__ .pytest_cache dashboard/node_modules/.vite

# Restart
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
cd dashboard && npm run dev &
```

---

## 📝 Demo Checklist

Use this checklist before your demo:

```
  ┌────────────────────────────────────────────────────────────┐
  │  ✅  PRE-DEMO CHECKLIST                                    │
  │                                                            │
  │  □  .env configured with valid API keys                    │
  │  □  Python venv activated, all packages installed          │
  │  □  Models are trained and cached                          │
  │  □  FastAPI server starts without errors                   │
  │  □  React dashboard loads at http://localhost:3000         │
  │  □  WebSocket shows 🟢 Connected                           │
  │  □  Health endpoint returns {"status": "healthy"}          │
  │  □  At least one trading cycle completed successfully      │
  │  □  Sentiment panel shows articles and scores              │
  │  □  Backtest results available for display                 │
  │  □  Browser zoom set to 100% (for screenshots)            │
  │  □  Terminal windows arranged for easy switching           │
  │  □  Presentation slides ready (SYSTEM_LOOP + ARCHITECTURE) │
  │                                                            │
  └────────────────────────────────────────────────────────────┘
```

---

## 🎯 Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────┐
│                    ⚡ QUICK REFERENCE                           │
│                                                                 │
│  🟢 Start API:     uvicorn api.main:app --reload               │
│  🟢 Start Dashboard: cd dashboard && npm run dev               │
│  🔄 Run Cycle:     POST http://localhost:8000/api/trade/cycle  │
│  📊 Health:        GET  http://localhost:8000/api/health        │
│  🧪 Sentiment:     GET  http://localhost:8000/api/sentiment    │
│  🌀 Regime:        GET  http://localhost:8000/api/regime       │
│  ⚖️ Risk:          GET  http://localhost:8000/api/risk         │
│  📊 Backtest:      POST http://localhost:8000/api/backtest/run │
│  ⚙️ Config:        GET  http://localhost:8000/api/config       │
│  🛑 Stop Trading:  POST http://localhost:8000/api/trade/stop   │
│                                                                 │
│  📁 Presentations:                                              │
│     • SYSTEM_LOOP.md     ← Visual system loop                  │
│     • ARCHITECTURE.md    ← Component & data flow diagrams      │
│     • DEMO_GUIDE.md      ← This file                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

*Built with ❤️ for ML4T — Machine Learning for Trading*
*Last updated: January 2025*
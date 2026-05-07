"""OHLCV data fetching and multi-timeframe bar generation.

Concepts: Ch 02 (market data, bar construction), Ch 04 (alpha factors), Ch 24 (factor library).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl


def fetch_ohlcv_yahoo(
    symbol: str,
    start: str,
    end: str,
    *,
    frequency: str = "daily",
    period: str | None = None,
) -> pl.DataFrame:
    """Fetch OHLCV from Yahoo Finance via yfinance."""
    import yfinance as yf

    interval_map = {
        "daily": "1d",
        "1d": "1d",
        "15m": "15m",
        "15minute": "15m",
        "1h": "1h",
        "hourly": "1h",
        "4h": "1h",
    }
    yf_interval = interval_map.get(frequency.lower(), "1d")

    if period:
        df = yf.download(symbol, period=period, interval=yf_interval, progress=False)
    else:
        df = yf.download(symbol, start=start, end=end, interval=yf_interval, progress=False)
    if df.empty:
        return pl.DataFrame(schema={
            "timestamp": pl.Datetime("ms"),
            "open": pl.Float64, "high": pl.Float64,
            "low": pl.Float64, "close": pl.Float64,
            "volume": pl.Float64,
        })

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df = df.reset_index()
    col_map = {"Date": "timestamp", "Datetime": "timestamp", "Open": "open",
               "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    return pl.from_pandas(df).with_columns(
        pl.col("timestamp").cast(pl.Datetime("ms"))
    ).select("timestamp", "open", "high", "low", "close", "volume").sort("timestamp")


def resample_bars(
    df: pl.DataFrame,
    target_frequency: Literal["4h", "1h"],
) -> pl.DataFrame:
    """Resample minute/bars into higher timeframe bars.

    Groups by calendar boundary and aggregates OHLCV.
    """
    if target_frequency == "1h":
        group_expr = pl.col("timestamp").dt.truncate("1h")
    elif target_frequency == "4h":
        group_expr = (pl.col("timestamp").dt.epoch("ms") // (4 * 3600 * 1000)) * (4 * 3600 * 1000)
        group_expr = (group_expr * 1_000).cast(pl.Datetime("ms"))
    else:
        raise ValueError(f"Unsupported target frequency: {target_frequency}")

    return df.sort("timestamp").group_by(group_expr.alias("_group"), maintain_order=True).agg(
        pl.col("timestamp").first().alias("timestamp"),
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
    ).drop("_group").sort("timestamp")


def compute_returns(df: pl.DataFrame, periods: tuple[int, ...] = (1, 5, 21)) -> pl.DataFrame:
    """Add return columns: ret_1, ret_5, ret_21 (or custom periods)."""
    exprs = [pl.col("close").pct_change(p).alias(f"ret_{p}") for p in periods]
    return df.with_columns(exprs)


def compute_ta_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """Compute technical indicators using vectorized operations.

    RSI, MACD, ATR, ADX, CCI, OBV, Stochastic — matching Ch 04 / Ch 24 factor library.
    """
    out = df.sort("timestamp")
    c = pl.col("close")
    h = pl.col("high")
    l = pl.col("low")
    v = pl.col("volume")

    # RSI(14)
    delta = c.diff()
    gain = delta.clip(lower_bound=0)
    loss = (-delta).clip(lower_bound=0)
    avg_gain = gain.rolling_mean(window_size=14)
    avg_loss = loss.rolling_mean(window_size=14)
    rs = avg_gain / avg_loss
    out = out.with_columns((100 - 100 / (1 + rs)).alias("rsi"))

    # MACD(12, 26, 9)
    ema12 = c.ewm_mean(span=12, adjust=False)
    ema26 = c.ewm_mean(span=26, adjust=False)
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm_mean(span=9, adjust=False)
    out = out.with_columns([
        macd_line.alias("macd"),
        (macd_line - signal_line).alias("macd_signal"),
    ])

    # ATR(14)
    tr = pl.max_horizontal([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ])
    out = out.with_columns(tr.rolling_mean(window_size=14).alias("atr"))

    # ADX(14)
    plus_dm = (h - h.shift(1)).clip(lower_bound=0)
    minus_dm = (l.shift(1) - l).clip(lower_bound=0)
    tr14 = tr.rolling_mean(window_size=14)
    plus_di = 100 * plus_dm.rolling_mean(window_size=14) / tr14
    minus_di = 100 * minus_dm.rolling_mean(window_size=14) / tr14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    out = out.with_columns(dx.rolling_mean(window_size=14).alias("adx"))

    # CCI(20)
    tp = (h + l + c) / 3
    sma_tp = tp.rolling_mean(window_size=20)
    mad = (tp - sma_tp).abs().rolling_mean(window_size=20)
    out = out.with_columns(((tp - sma_tp) / (0.015 * mad + 1e-10)).alias("cci"))

    # OBV
    obv = (
        pl.when(c > c.shift(1)).then(v)
        .when(c < c.shift(1)).then(-v)
        .otherwise(0)
        .cum_sum()
    )
    out = out.with_columns(obv.alias("obv"))

    # Stochastic(14, 3)
    lowest_low = l.rolling_min(window_size=14)
    highest_high = h.rolling_max(window_size=14)
    pct_k = 100 * (c - lowest_low) / (highest_high - lowest_low + 1e-10)
    out = out.with_columns([
        pct_k.alias("stoch_k"),
        pct_k.rolling_mean(window_size=3).alias("stoch_d"),
    ])

    return out


def compute_intraday_returns(df: pl.DataFrame) -> pl.DataFrame:
    """Add ret_b1, ret_b4, ret_b26 for intraday bar frequencies."""
    return df.with_columns([
        pl.col("close").pct_change(1).alias("ret_b1"),
        pl.col("close").pct_change(4).alias("ret_b4"),
        pl.col("close").pct_change(26).alias("ret_b26"),
    ])


# Feature column sets for different frequencies
NUMERIC_FEATURE_COLUMNS_DAILY: tuple[str, ...] = (
    "rsi", "macd", "macd_signal", "atr", "adx", "cci", "obv", "stoch_k", "stoch_d",
    "ret_1", "ret_5", "ret_21",
)

NUMERIC_FEATURE_COLUMNS_15M: tuple[str, ...] = (
    "rsi", "macd", "macd_signal", "atr", "adx", "cci", "obv", "stoch_k", "stoch_d",
    "ret_b1", "ret_b4", "ret_b26",
)

NUMERIC_FEATURE_COLUMNS_4H: tuple[str, ...] = (
    "rsi", "macd", "macd_signal", "atr", "adx", "cci", "obv", "stoch_k", "stoch_d",
    "ret_1", "ret_5",
)


INTRADAY_FREQ = frozenset({"15minute", "15m", "15min", "5minute", "5m", "30minute", "30m"})


def is_intraday(freq: str) -> bool:
    return freq.strip().lower() in INTRADAY_FREQ


def feature_columns_for(freq: str) -> tuple[str, ...]:
    if freq.strip().lower() in ("4h", "4hour"):
        return NUMERIC_FEATURE_COLUMNS_4H
    if is_intraday(freq):
        return NUMERIC_FEATURE_COLUMNS_15M
    return NUMERIC_FEATURE_COLUMNS_DAILY


def build_feature_table(
    raw: pl.DataFrame,
    *,
    bar_frequency: str = "daily",
    upper_barrier: float = 0.02,
    lower_barrier: float = 0.01,
    max_holding_period: int = 10,
) -> pl.DataFrame:
    """Full feature engineering pipeline: indicators + returns + triple-barrier labels."""
    out = compute_ta_indicators(raw)

    if is_intraday(bar_frequency):
        out = compute_intraday_returns(out)
    else:
        out = compute_returns(out)

    out = _triple_barrier_labels(out, upper_barrier, lower_barrier, max_holding_period)
    return out


def _triple_barrier_labels(
    df: pl.DataFrame,
    upper: float,
    lower: float,
    max_holding: int,
) -> pl.DataFrame:
    """Compute triple-barrier labels: 0=down, 1=neutral, 2=up.

    Simplified vectorized version — looks forward up to max_holding bars.
    """
    close = df["close"].to_numpy()
    n = len(close)
    labels = np.ones(n, dtype=np.int64)

    for i in range(n - 1):
        barrier_idx = min(i + max_holding, n - 1)
        future = close[i + 1 : barrier_idx + 1]
        if len(future) == 0:
            continue
        upper_price = close[i] * (1 + upper)
        lower_price = close[i] * (1 - lower)

        hit_upper = np.where(future >= upper_price)[0]
        hit_lower = np.where(future <= lower_price)[0]

        first_upper = hit_upper[0] + i + 1 if len(hit_upper) > 0 else n
        first_lower = hit_lower[0] + i + 1 if len(hit_lower) > 0 else n

        if first_upper < first_lower and first_upper <= barrier_idx:
            labels[i] = 2
        elif first_lower < first_upper and first_lower <= barrier_idx:
            labels[i] = 0

    return df.with_columns([
        pl.Series("label", labels),
        (pl.Series("label", labels) + pl.Series("offset", np.ones(n, dtype=np.int64))).alias("target"),
    ])


def feature_columns_present(df: pl.DataFrame) -> tuple[str, ...]:
    """Return ordered feature names present in the frame.

    Picks the correct base set by frequency, then adds any S/R, session,
    and GenAI columns that are present.
    """
    from agent.data.sr_features import SR_FEATURE_COLUMNS
    from agent.data.session_analyzer import SESSION_FEATURE_COLUMNS

    # Pick base set
    base_cols: tuple[str, ...] = NUMERIC_FEATURE_COLUMNS_DAILY
    for cols in (NUMERIC_FEATURE_COLUMNS_15M, NUMERIC_FEATURE_COLUMNS_4H, NUMERIC_FEATURE_COLUMNS_DAILY):
        if any(c in df.columns for c in cols):
            # Check which set has the highest match rate
            match = sum(1 for c in cols if c in df.columns)
            if match > 0:
                base_cols = cols
                break

    base = [c for c in base_cols if c in df.columns]

    # S/R features
    sr_extra = [c for c in SR_FEATURE_COLUMNS if c in df.columns and c not in base]

    # Session features
    sess_extra = [c for c in SESSION_FEATURE_COLUMNS if c in df.columns and c not in base]

    # GenAI features
    gai_extra = sorted(c for c in df.columns if str(c).endswith("_gai") and c not in base)

    return tuple(base + sr_extra + sess_extra + gai_extra)


import pandas as pd  # noqa: E402 — needed for yfinance column rename


def save_parquet(df: pl.DataFrame, path: Path) -> None:
    """Save DataFrame as parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def load_parquet(path: Path) -> pl.DataFrame:
    """Load DataFrame from parquet."""
    return pl.read_parquet(path)

"""Tests for data pipeline: TA indicators, returns, feature columns."""

import polars as pl
import pytest
import numpy as np


def test_compute_returns(sample_ohlcv_daily: pl.DataFrame):
    from agent.data.pipeline import compute_returns
    result = compute_returns(sample_ohlcv_daily, periods=(1, 5))
    assert "ret_1" in result.columns
    assert "ret_5" in result.columns
    vals = result["ret_1"].drop_nulls().to_numpy()
    assert len(vals) > 0


def test_compute_ta_indicators(sample_ohlcv_daily: pl.DataFrame):
    from agent.data.pipeline import compute_ta_indicators
    result = compute_ta_indicators(sample_ohlcv_daily)
    for col in ("rsi", "macd", "atr", "adx", "cci", "obv", "stoch_k", "stoch_d"):
        assert col in result.columns, f"Missing column: {col}"


def test_build_feature_table(sample_ohlcv_daily: pl.DataFrame):
    from agent.data.pipeline import build_feature_table
    result = build_feature_table(sample_ohlcv_daily)
    assert "target" in result.columns
    assert "label" in result.columns
    targets = result["target"].to_numpy()
    assert all(t in (0, 1, 2) for t in targets)


def test_feature_columns_present(sample_ohlcv_daily: pl.DataFrame):
    from agent.data.pipeline import build_feature_table, feature_columns_present
    result = build_feature_table(sample_ohlcv_daily)
    cols = feature_columns_present(result)
    assert len(cols) > 0


def test_triple_barrier_labels_distinct():
    """With enough price movement, labels should include 0 and 2."""
    import datetime as dt
    n = 100
    base = dt.date(2024, 1, 1)
    dates = [(base + dt.timedelta(days=i)).isoformat() for i in range(n)]
    ts = pl.Series("timestamp", dates).str.strptime(pl.Datetime("ms"), "%Y-%m-%d")
    prices = [2000 + i * 3 for i in range(n)]  # Strong uptrend
    df = pl.DataFrame({
        "timestamp": ts,
        "open": prices, "high": [p + 5 for p in prices], "low": [p - 5 for p in prices],
        "close": prices, "volume": [100000] * n,
    }, schema={"timestamp": pl.Datetime("ms"), "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64})

    from agent.data.pipeline import build_feature_table
    result = build_feature_table(df, upper_barrier=0.01, lower_barrier=0.01)
    labels = set(result["label"].drop_nulls().to_list())
    assert 2 in labels  # Should detect uptrend labels

"""Tests for regime detection."""

import numpy as np
import polars as pl
import pytest
from agent.models.regime import GMMRegimeDetector


def test_regime_detector_fit_predict():
    n = 200
    np.random.seed(42)
    df = pl.DataFrame({
        "timestamp": pl.Series([f"2024-01-{i%28+1:02d}" for i in range(n)]).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"),
        "close": 2000 + np.cumsum(np.random.randn(n) * 5),
        "ret_1": np.random.randn(n) * 0.01,
        "atr": np.abs(np.random.randn(n)) * 10,
        "adx": np.random.uniform(10, 40, n),
        "rsi": np.random.uniform(20, 80, n),
        "cci": np.random.randn(n) * 50,
        "stoch_k": np.random.uniform(20, 80, n),
        "obv": np.cumsum(np.random.randn(n) * 1000),
        "volume": np.random.uniform(50000, 200000, n),
    })

    det = GMMRegimeDetector(n_components=4)
    det.fit(df)
    results = det.predict(df.tail(1))

    assert len(results) == 1
    assert results[0].regime_name in ("trending_up", "trending_down", "ranging", "volatile")
    assert len(results[0].probabilities) == 4


def test_regime_detector_save_load(tmp_path):
    n = 100
    np.random.seed(42)
    df = pl.DataFrame({
        "timestamp": pl.Series([f"2024-01-{i%28+1:02d}" for i in range(n)]).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"),
        "close": 2000 + np.cumsum(np.random.randn(n) * 5),
        "ret_1": np.random.randn(n) * 0.01,
        "atr": np.abs(np.random.randn(n)) * 10,
        "adx": np.random.uniform(10, 40, n),
        "rsi": np.random.uniform(20, 80, n),
        "cci": np.random.randn(n) * 50,
        "stoch_k": np.random.uniform(20, 80, n),
        "obv": np.cumsum(np.random.randn(n) * 1000),
        "volume": np.random.uniform(50000, 200000, n),
    })

    det = GMMRegimeDetector(n_components=4)
    det.fit(df)
    path = tmp_path / "regime.pkl"
    det.save(path)

    loaded = GMMRegimeDetector.load(path)
    results = loaded.predict(df.tail(1))
    assert len(results) == 1

"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def sample_ohlcv_daily() -> pl.DataFrame:
    """10-day sample of gold OHLCV data."""
    dates = pl.Series(
        "timestamp",
        [
            "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
            "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11",
            "2024-01-12", "2024-01-15",
        ],
    ).str.strptime(pl.Datetime("ms"), "%Y-%m-%d")
    return pl.DataFrame(
        {
            "timestamp": dates,
            "open":  [2060, 2050, 2045, 2055, 2060, 2065, 2070, 2068, 2075, 2080],
            "high":  [2070, 2058, 2055, 2065, 2068, 2075, 2080, 2078, 2085, 2090],
            "low":   [2048, 2042, 2038, 2048, 2055, 2060, 2065, 2060, 2070, 2075],
            "close": [2050, 2045, 2055, 2060, 2065, 2070, 2068, 2075, 2080, 2085],
            "volume": [100000, 120000, 90000, 110000, 105000, 95000, 115000, 100000, 108000, 112000],
        },
        schema={
            "timestamp": pl.Datetime("ms"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
    )


@pytest.fixture
def artifacts_dir(tmp_path: Path) -> Path:
    d = tmp_path / "artifacts"
    d.mkdir()
    return d

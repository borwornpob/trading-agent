"""Tests for session analyzer."""

import datetime as dt
import polars as pl
import pytest
from agent.data.session_analyzer import (
    detect_session, get_session_name, is_high_liquidity_session,
    is_asian_session, compute_session_features,
)


def test_detect_session():
    ts = pl.Series("timestamp", [
        dt.datetime(2024, 1, 1, 3, 0, tzinfo=dt.UTC),   # Asian
        dt.datetime(2024, 1, 1, 10, 0, tzinfo=dt.UTC),   # London
        dt.datetime(2024, 1, 1, 15, 0, tzinfo=dt.UTC),   # London/NY overlap
        dt.datetime(2024, 1, 1, 20, 0, tzinfo=dt.UTC),   # NY
    ]).cast(pl.Datetime("ms"))
    df = pl.DataFrame({"timestamp": ts, "close": [1.0, 2.0, 3.0, 4.0]})
    result = compute_session_features(df)
    sessions = result["session"].to_list()
    assert sessions[0] == "asian"
    assert sessions[1] == "london"
    assert sessions[2] == "london_ny_overlap"
    assert sessions[3] == "ny"


def test_get_session_name():
    assert get_session_name(dt.datetime(2024, 1, 1, 14, 0, tzinfo=dt.UTC)) == "london_ny_overlap"
    assert get_session_name(dt.datetime(2024, 1, 1, 3, 0, tzinfo=dt.UTC)) == "asian"
    assert get_session_name(dt.datetime(2024, 1, 1, 10, 0, tzinfo=dt.UTC)) == "london"
    assert get_session_name(dt.datetime(2024, 1, 1, 20, 0, tzinfo=dt.UTC)) == "ny"


def test_is_high_liquidity():
    assert is_high_liquidity_session(dt.datetime(2024, 1, 1, 14, 0, tzinfo=dt.UTC))
    assert not is_high_liquidity_session(dt.datetime(2024, 1, 1, 3, 0, tzinfo=dt.UTC))


def test_is_asian():
    assert is_asian_session(dt.datetime(2024, 1, 1, 3, 0, tzinfo=dt.UTC))
    assert not is_asian_session(dt.datetime(2024, 1, 1, 14, 0, tzinfo=dt.UTC))


def test_compute_session_features():
    df = pl.DataFrame({
        "timestamp": pl.Series([
            dt.datetime(2024, 1, 1, 14, 0, tzinfo=dt.UTC),
        ]).cast(pl.Datetime("ms")),
        "close": [2050.0],
    })
    result = compute_session_features(df)
    assert "session" in result.columns

"""Session detection and range features for gold.

Gold trades nearly 24/5 but liquidity and behavior vary dramatically by session:
- Asian session (00:00-07:00 UTC): low volatility, range-building
- London session (07:00-16:00 UTC): high volume, directional moves
- NY session (13:00-22:00 UTC): high volume, overlap with London
- London/NY overlap (13:00-17:00 UTC): peak liquidity

Concepts: Ch 02 (market microstructure, session calendars).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class SessionWindow:
    name: str
    start_hour_utc: int
    end_hour_utc: int


SESSIONS = {
    "asian": SessionWindow("asian", 0, 7),
    "london": SessionWindow("london", 7, 16),
    "ny": SessionWindow("ny", 13, 22),
    "london_ny_overlap": SessionWindow("london_ny_overlap", 13, 17),
}


def detect_session(timestamp: pl.Series) -> pl.Series:
    """Classify each timestamp into its trading session."""
    hour = timestamp.dt.hour()
    return (
        pl.when((hour >= 13) & (hour < 17)).then(pl.lit("london_ny_overlap"))
        .when((hour >= 7) & (hour < 16)).then(pl.lit("london"))
        .when((hour >= 13) & (hour < 22)).then(pl.lit("ny"))
        .when((hour >= 0) & (hour < 7)).then(pl.lit("asian"))
        .otherwise(pl.lit("off_hours"))
        .alias("session")
    )


def compute_session_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add session identification and session-range features.

    For daily bars, these are approximations based on typical session behavior.
    For intraday bars, exact session ranges are computed.
    """
    out = df.sort("timestamp")

    ts = out["timestamp"]
    if ts.dtype == pl.Datetime:
        out = out.with_columns(detect_session(ts))
    else:
        out = out.with_columns(pl.lit("unknown").alias("session"))

    return out


def compute_asian_range(df: pl.DataFrame) -> pl.DataFrame:
    """Compute Asian session high/low for intraday data.

    Asian range is a key predictor: London/NY often test these levels.
    Requires intraday (15m or hourly) data with timestamps.
    """
    if "session" not in df.columns:
        df = compute_session_features(df)

    asian = df.filter(pl.col("session") == "asian")

    if asian.is_empty():
        return df.with_columns([
            pl.lit(None).cast(pl.Float64).alias("asian_high"),
            pl.lit(None).cast(pl.Float64).alias("asian_low"),
            pl.lit(None).cast(pl.Float64).alias("asian_range"),
        ])

    date_col = pl.col("timestamp").dt.date().alias("_date")
    asian_levels = asian.group_by(date_col).agg([
        pl.col("high").max().alias("asian_high"),
        pl.col("low").min().alias("asian_low"),
    ]).with_columns(
        (pl.col("asian_high") - pl.col("asian_low")).alias("asian_range")
    ).rename({"_date": "_join_date"})

    return df.with_columns(
        pl.col("timestamp").dt.date().alias("_join_date")
    ).join(asian_levels, on="_join_date", how="left").drop("_join_date")


def compute_session_range_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add derived session features for model input."""
    out = df

    if "asian_high" in out.columns and "asian_low" in out.columns:
        out = out.with_columns([
            (pl.col("close") - pl.col("asian_high")).alias("dist_to_asian_high"),
            (pl.col("close") - pl.col("asian_low")).alias("dist_to_asian_low"),
        ])

    if "prev_high" in out.columns:
        out = out.with_columns([
            pl.when(pl.col("close") > pl.col("prev_high"))
            .then(pl.lit(1))
            .when(pl.col("close") < pl.col("prev_low"))
            .then(pl.lit(-1))
            .otherwise(pl.lit(0))
            .alias("breakout_signal"),
        ])

    return out


def is_high_liquidity_session(timestamp_utc: dt.datetime) -> bool:
    """Check if current time is during London/NY overlap (peak liquidity)."""
    hour = timestamp_utc.hour
    return 13 <= hour < 17


def is_asian_session(timestamp_utc: dt.datetime) -> bool:
    """Check if current time is during Asian session."""
    hour = timestamp_utc.hour
    return 0 <= hour < 7


def get_session_name(timestamp_utc: dt.datetime) -> str:
    """Return the current session name."""
    hour = timestamp_utc.hour
    if 13 <= hour < 17:
        return "london_ny_overlap"
    if 7 <= hour < 16:
        return "london"
    if 13 <= hour < 22:
        return "ny"
    if 0 <= hour < 7:
        return "asian"
    return "off_hours"


# Session-related feature columns
SESSION_FEATURE_COLUMNS: tuple[str, ...] = (
    "dist_to_asian_high",
    "dist_to_asian_low",
    "asian_range",
    "breakout_signal",
)

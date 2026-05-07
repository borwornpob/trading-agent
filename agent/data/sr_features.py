"""Support/Resistance feature engineering.

Concepts: Ch 04 (alpha factors), Ch 09 (time series structure), Ch 24 (factor library).
Predicts key price levels for smart grid placement.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def compute_prev_day_levels(df: pl.DataFrame) -> pl.DataFrame:
    """Previous day high, low, close, midpoint as features."""
    return df.with_columns([
        pl.col("high").shift(1).alias("prev_high"),
        pl.col("low").shift(1).alias("prev_low"),
        pl.col("close").shift(1).alias("prev_close"),
        ((pl.col("high").shift(1) + pl.col("low").shift(1)) / 2).alias("prev_mid"),
        (pl.col("high").shift(1) - pl.col("low").shift(1)).alias("prev_range"),
    ])


def compute_swing_points(df: pl.DataFrame, window: int = 5) -> pl.DataFrame:
    """Detect local swing highs and lows over a rolling window."""
    out = df.sort("timestamp")

    out = out.with_columns([
        pl.col("high").rolling_max(window_size=window).alias("swing_high_raw"),
        pl.col("low").rolling_min(window_size=window).alias("swing_low_raw"),
    ])

    out = out.with_columns([
        pl.when(pl.col("high") == pl.col("swing_high_raw"))
        .then(pl.col("high"))
        .otherwise(None)
        .forward_fill()
        .alias("nearest_swing_high"),

        pl.when(pl.col("low") == pl.col("swing_low_raw"))
        .then(pl.col("low"))
        .otherwise(None)
        .forward_fill()
        .alias("nearest_swing_low"),
    ])

    out = out.with_columns([
        (pl.col("close") - pl.col("nearest_swing_high")).alias("dist_to_swing_high"),
        (pl.col("close") - pl.col("nearest_swing_low")).alias("dist_to_swing_low"),
    ])

    return out.drop(["swing_high_raw", "swing_low_raw"])


def compute_round_number_features(
    df: pl.DataFrame,
    levels: tuple[float, ...] = (5.0, 10.0, 50.0),
) -> pl.DataFrame:
    """Distance to nearest round number at various granularities.

    Gold tends to respect psychological levels: 2650, 2700, 2500, etc.
    """
    exprs = []
    for lvl in levels:
        col_name = f"dist_to_round_{int(lvl)}"
        exprs.append(
            (pl.col("close") - (pl.col("close") / lvl).round() * lvl).alias(col_name)
        )
    return df.with_columns(exprs)


def compute_volume_profile(df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """Volume-weighted price level (VWAP approximation) and volume concentration."""
    out = df.sort("timestamp")

    out = out.with_columns([
        ((pl.col("close") * pl.col("volume")).rolling_sum(window_size=window)
         / (pl.col("volume").rolling_sum(window_size=window) + 1e-10))
        .alias("vwap"),
    ])

    out = out.with_columns([
        (pl.col("close") - pl.col("vwap")).alias("dist_to_vwap"),
        (pl.col("volume").rolling_std(window_size=window)
         / (pl.col("volume").rolling_mean(window_size=window) + 1e-10))
        .alias("volume_cv"),
    ])

    return out


def compute_pivot_levels(df: pl.DataFrame) -> pl.DataFrame:
    """Classic pivot points: PP, R1, R2, S1, S2 from previous day.

    Widely watched by institutional traders.
    """
    out = df.with_columns([
        ((pl.col("prev_high") + pl.col("prev_low") + pl.col("prev_close")) / 3).alias("pivot_pp"),
    ])
    out = out.with_columns([
        (2 * pl.col("pivot_pp") - pl.col("prev_low")).alias("pivot_r1"),
        (pl.col("pivot_pp") + (pl.col("prev_high") - pl.col("prev_low"))).alias("pivot_r2"),
        (2 * pl.col("pivot_pp") - pl.col("prev_high")).alias("pivot_s1"),
        (pl.col("pivot_pp") - (pl.col("prev_high") - pl.col("prev_low"))).alias("pivot_s2"),
    ])
    out = out.with_columns([
        (pl.col("close") - pl.col("pivot_r1")).alias("dist_to_r1"),
        (pl.col("close") - pl.col("pivot_s1")).alias("dist_to_s1"),
        (pl.col("close") - pl.col("pivot_pp")).alias("dist_to_pivot"),
    ])
    return out


def compute_atr_zones(df: pl.DataFrame) -> pl.DataFrame:
    """ATR-based zone width for S/R level fuzziness.

    S/R is never exact — zones are ATR-based bands.
    """
    if "atr" not in df.columns:
        return df
    return df.with_columns([
        (pl.col("atr") * 0.5).alias("sr_zone_width_half"),
        (pl.col("atr") * 2.0).alias("daily_range_estimate"),
    ])


def compute_all_sr_features(df: pl.DataFrame) -> pl.DataFrame:
    """Full S/R feature pipeline."""
    out = compute_prev_day_levels(df)
    out = compute_swing_points(out)
    out = compute_round_number_features(out)
    out = compute_volume_profile(out)
    out = compute_pivot_levels(out)
    out = compute_atr_zones(out)
    return out


# S/R-specific feature columns for the prediction model
SR_FEATURE_COLUMNS: tuple[str, ...] = (
    "prev_high", "prev_low", "prev_close", "prev_mid", "prev_range",
    "nearest_swing_high", "nearest_swing_low",
    "dist_to_swing_high", "dist_to_swing_low",
    "dist_to_round_5", "dist_to_round_10", "dist_to_round_50",
    "vwap", "dist_to_vwap", "volume_cv",
    "pivot_pp", "pivot_r1", "pivot_r2", "pivot_s1", "pivot_s2",
    "dist_to_r1", "dist_to_s1", "dist_to_pivot",
    "atr", "sr_zone_width_half", "daily_range_estimate",
    "rsi", "adx", "cci",
)

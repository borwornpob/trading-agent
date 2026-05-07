"""Fetch gold data for all timeframes and build complete feature tables.

Usage: python -m scripts.fetch_all_timeframes
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import ARTIFACTS
from agent.data.pipeline import (
    fetch_ohlcv_yahoo,
    build_feature_table,
    save_parquet,
    resample_bars,
)
from agent.data.sr_features import compute_all_sr_features
from agent.data.session_analyzer import (
    compute_session_features,
    compute_asian_range,
    compute_session_range_features,
)


def build_enhanced_features(raw: pl.DataFrame, *, bar_frequency: str = "daily") -> pl.DataFrame:
    """Build full feature set: TA + S/R + session + returns + labels."""
    import polars as pl

    out = build_feature_table(raw, bar_frequency=bar_frequency)
    out = compute_all_sr_features(out)
    out = compute_session_features(out)

    if bar_frequency in ("15m", "1h", "hourly"):
        out = compute_asian_range(out)
        out = compute_session_range_features(out)

    return out


def main() -> None:
    import polars as pl

    print("=" * 60)
    print("  Multi-Timeframe Gold Data Pipeline")
    print("=" * 60)

    # --- 1h data (2 years) ---
    print("\n[1/3] Fetching 1h data (730d)...")
    try:
        raw_1h = fetch_ohlcv_yahoo("GC=F", "", "", frequency="1h", period="730d")
        print(f"  Fetched {len(raw_1h)} 1h bars")
        if len(raw_1h) > 100:
            feat_1h = build_enhanced_features(raw_1h, bar_frequency="1h")
            save_parquet(feat_1h, ARTIFACTS / "gold_1h.parquet")
            valid = feat_1h.drop_nulls(subset=["target"])
            print(f"  {len(valid)} rows with valid targets, {len(feat_1h.columns)} columns")
    except Exception as e:
        print(f"  ERROR: {e}")

    # --- 4h from 1h resample ---
    print("\n[2/3] Resampling 1h -> 4h...")
    try:
        if len(raw_1h) > 100:
            raw_4h = resample_bars(raw_1h, "4h")
            feat_4h = build_enhanced_features(raw_4h, bar_frequency="4h")
            save_parquet(feat_4h, ARTIFACTS / "gold_4h.parquet")
            valid = feat_4h.drop_nulls(subset=["target"])
            print(f"  {len(valid)} rows with valid targets, {len(feat_4h.columns)} columns")
    except Exception as e:
        print(f"  ERROR: {e}")

    # --- 15m data (60 days) ---
    print("\n[3/3] Fetching 15m data (60d)...")
    try:
        raw_15m = fetch_ohlcv_yahoo("GC=F", "", "", frequency="15m", period="60d")
        print(f"  Fetched {len(raw_15m)} 15m bars")
        if len(raw_15m) > 100:
            feat_15m = build_enhanced_features(raw_15m, bar_frequency="15m")
            save_parquet(feat_15m, ARTIFACTS / "gold_15m.parquet")
            valid = feat_15m.drop_nulls(subset=["target"])
            print(f"  {len(valid)} rows with valid targets, {len(feat_15m.columns)} columns")
    except Exception as e:
        print(f"  ERROR: {e}")

    # --- Daily data (extended) ---
    print("\n[Bonus] Fetching extended daily data (5 years)...")
    try:
        raw_daily = fetch_ohlcv_yahoo("GC=F", "2021-01-01", "2026-04-21", frequency="daily")
        print(f"  Fetched {len(raw_daily)} daily bars")
        feat_daily = build_enhanced_features(raw_daily, bar_frequency="daily")
        save_parquet(feat_daily, ARTIFACTS / "gold_training.parquet")
        valid = feat_daily.drop_nulls(subset=["target"])
        print(f"  {len(valid)} rows with valid targets, {len(feat_daily.columns)} columns")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("  All timeframes saved to artifacts/")
    print("=" * 60)

    # Summary
    print("\nFeature columns across daily data:")
    if len(feat_daily) > 0:
        for col in sorted(feat_daily.columns):
            print(f"  - {col}")


if __name__ == "__main__":
    import polars as pl
    main()

"""Fetch MT5 broker candles and build feature artifacts.

Usage:
    python -m scripts.fetch_mt5_timeframes --count 20000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import ARTIFACTS, AgentConfig, MT5Settings
from agent.data.pipeline import build_feature_table, resample_bars, save_parquet
from agent.data.session_analyzer import (
    compute_asian_range,
    compute_session_features,
    compute_session_range_features,
)
from agent.data.sr_features import compute_all_sr_features
from agent.mt5_gateway import MT5Gateway


def build_mt5_features(
    raw: pl.DataFrame,
    *,
    bar_frequency: str,
    upper_barrier: float,
    lower_barrier: float,
    max_holding_period: int,
) -> pl.DataFrame:
    out = build_feature_table(
        raw,
        bar_frequency=bar_frequency,
        upper_barrier=upper_barrier,
        lower_barrier=lower_barrier,
        max_holding_period=max_holding_period,
    )
    out = compute_all_sr_features(out)
    out = compute_session_features(out)
    if bar_frequency in ("15m", "1h", "hourly"):
        out = compute_asian_range(out)
        out = compute_session_range_features(out)
    return out


def _write_artifact(
    df: pl.DataFrame,
    path: Path,
    *,
    bar_frequency: str,
    upper_barrier: float,
    lower_barrier: float,
    max_holding_period: int,
) -> dict[str, object]:
    features = build_mt5_features(
        df,
        bar_frequency=bar_frequency,
        upper_barrier=upper_barrier,
        lower_barrier=lower_barrier,
        max_holding_period=max_holding_period,
    )
    save_parquet(features, path)
    ts = features["timestamp"].to_list()
    return {
        "path": str(path),
        "rows": len(features),
        "valid_targets": len(features.drop_nulls(subset=["target"])),
        "start": str(ts[0]) if ts else None,
        "end": str(ts[-1]) if ts else None,
        "columns": len(features.columns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch MT5 OHLCV and build feature artifacts")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--frequency", default="15m")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--prefix", default="gold_mt5")
    parser.add_argument("--upper-barrier", type=float, default=0.01)
    parser.add_argument("--lower-barrier", type=float, default=0.005)
    parser.add_argument("--max-holding-period", type=int, default=10)
    parser.add_argument("--replace-live-15m", action="store_true")
    args = parser.parse_args()

    config = AgentConfig.from_env()
    settings = MT5Settings.from_env()
    symbol = args.symbol or config.broker_symbol_name
    count = args.count or settings.rate_count

    gateway = MT5Gateway(settings)
    gateway.connect()
    try:
        raw_15m = gateway.fetch_ohlcv(symbol=symbol, frequency=args.frequency, count=count)
    finally:
        gateway.disconnect()

    if raw_15m.is_empty():
        raise SystemExit("MT5 returned no bars")

    raw_15m_path = ARTIFACTS / f"{args.prefix}_raw_15m.parquet"
    save_parquet(raw_15m, raw_15m_path)

    summary = {
        "symbol": symbol,
        "requested_frequency": args.frequency,
        "requested_count": count,
        "labeling": {
            "upper_barrier": args.upper_barrier,
            "lower_barrier": args.lower_barrier,
            "max_holding_period": args.max_holding_period,
        },
        "raw_15m": {
            "path": str(raw_15m_path),
            "rows": len(raw_15m),
            "start": str(raw_15m["timestamp"][0]),
            "end": str(raw_15m["timestamp"][-1]),
        },
    }

    summary["features_15m"] = _write_artifact(
        raw_15m,
        ARTIFACTS / f"{args.prefix}_15m.parquet",
        bar_frequency="15m",
        upper_barrier=args.upper_barrier,
        lower_barrier=args.lower_barrier,
        max_holding_period=args.max_holding_period,
    )

    raw_1h = resample_bars(raw_15m, "1h")
    raw_1h_path = ARTIFACTS / f"{args.prefix}_raw_1h.parquet"
    save_parquet(raw_1h, raw_1h_path)
    summary["raw_1h"] = {"path": str(raw_1h_path), "rows": len(raw_1h)}
    summary["features_1h"] = _write_artifact(
        raw_1h,
        ARTIFACTS / f"{args.prefix}_1h.parquet",
        bar_frequency="1h",
        upper_barrier=args.upper_barrier,
        lower_barrier=args.lower_barrier,
        max_holding_period=args.max_holding_period,
    )

    raw_4h = resample_bars(raw_1h, "4h")
    raw_4h_path = ARTIFACTS / f"{args.prefix}_raw_4h.parquet"
    save_parquet(raw_4h, raw_4h_path)
    summary["raw_4h"] = {"path": str(raw_4h_path), "rows": len(raw_4h)}
    summary["features_4h"] = _write_artifact(
        raw_4h,
        ARTIFACTS / f"{args.prefix}_4h.parquet",
        bar_frequency="4h",
        upper_barrier=args.upper_barrier,
        lower_barrier=args.lower_barrier,
        max_holding_period=args.max_holding_period,
    )

    if args.replace_live_15m:
        save_parquet(pl.read_parquet(ARTIFACTS / f"{args.prefix}_15m.parquet"), ARTIFACTS / "gold_15m.parquet")

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

"""Fetch gold OHLCV data and build training dataset.

Usage: python -m scripts.fetch_gold_data [--start 2023-01-01] [--end 2024-12-31] [--frequency daily]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import ARTIFACTS
from agent.data.pipeline import fetch_ohlcv_yahoo, build_feature_table, save_parquet
from agent.data.sr_features import compute_all_sr_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch gold data and build features")
    parser.add_argument("--symbol", default="GC=F", help="Yahoo Finance symbol")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--frequency", default="daily", choices=["daily", "15m", "1h"])
    parser.add_argument("--output", default=str(ARTIFACTS / "gold_training.parquet"))
    args = parser.parse_args()

    print(f"Fetching {args.symbol} {args.frequency} data from {args.start} to {args.end}...")
    raw = fetch_ohlcv_yahoo(args.symbol, args.start, args.end, frequency=args.frequency)
    print(f"  Fetched {len(raw)} bars")

    print("Building features...")
    features = build_feature_table(raw, bar_frequency=args.frequency)
    features = compute_all_sr_features(features)

    valid = features.drop_nulls(subset=["target"])
    print(f"  {len(valid)} rows with valid targets")

    out_path = Path(args.output)
    save_parquet(features, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

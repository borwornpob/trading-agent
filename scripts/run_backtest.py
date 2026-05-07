"""Run backtest on holdout data.

Usage: python -m scripts.run_backtest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import ARTIFACTS
from agent.data.pipeline import load_parquet, feature_columns_present
from agent.models.lgbm_model import load_bundle, predict_table
from agent.backtest.engine import run_vectorized_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--dataset", default=str(ARTIFACTS / "gold_training.parquet"))
    parser.add_argument("--model", default=str(ARTIFACTS / "gold_lgbm.pkl"))
    parser.add_argument("--initial-cash", type=float, default=100_000)
    parser.add_argument("--commission", type=float, default=0.0002)
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset}...")
    df = load_parquet(Path(args.dataset))

    print(f"Loading model from {args.model}...")
    bundle = load_bundle(Path(args.model))
    model = bundle["model"]
    feats = tuple(bundle.get("feature_names", list(feature_columns_present(df))))
    print(f"  {len(feats)} features")

    valid = df.drop_nulls(subset=list(feats) + ["target"]).sort("timestamp")
    print(f"  {len(valid)} valid rows")

    # Split: use last 20% as test
    n = len(valid)
    split = int(n * 0.8)
    train_df = valid[:split]
    test_df = valid[split:]

    print(f"Running backtest on {len(test_df)} test bars...")
    predictions = predict_table(model, test_df, feats)
    prices = test_df.select("timestamp", "close")

    result = run_vectorized_backtest(
        predictions, prices,
        initial_cash=args.initial_cash,
        commission_rate=args.commission,
    )

    print("\n=== Backtest Results ===")
    for key, value in result.metrics.items():
        print(f"  {key}: {value}")
    print(f"  Trades: {len(result.trades)}")


if __name__ == "__main__":
    main()

"""Train the primary LightGBM model with walk-forward validation.

Usage: python -m scripts.train_model [--dataset artifacts/gold_training.parquet]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import ARTIFACTS
from agent.data.pipeline import load_parquet, feature_columns_present
from agent.models.lgbm_model import train_model, save_bundle, save_model_card


def main() -> None:
    parser = argparse.ArgumentParser(description="Train gold trading model")
    parser.add_argument("--dataset", default=str(ARTIFACTS / "gold_training.parquet"))
    parser.add_argument("--output", default=str(ARTIFACTS / "gold_lgbm.pkl"))
    parser.add_argument("--card", default=str(ARTIFACTS / "model_card.json"))
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset}...")
    df = load_parquet(Path(args.dataset))
    print(f"  {len(df)} rows, columns: {df.columns}")

    feature_names = feature_columns_present(df)
    print(f"  {len(feature_names)} features: {feature_names}")

    valid = df.drop_nulls(subset=list(feature_names) + ["target"])
    print(f"  {len(valid)} valid rows after drop_nulls")

    print("Training model...")
    model, feats, meta = train_model(valid, feature_names=feature_names, random_state=args.random_state)
    print(f"  Backend: {meta['backend']}, Classes: {meta['classes']}")

    # Also create a momentum secondary voter
    from agent.models.ensemble import MomentumQuantileClassifier
    import numpy as np

    mom_col = "ret_21" if "ret_21" in feats else ("ret_b26" if "ret_b26" in feats else None)
    secondary = None
    if mom_col:
        mom_idx = list(feats).index(mom_col)
        ret_vals = valid[mom_col].drop_nulls().to_numpy()
        lo, hi = np.quantile(ret_vals[np.isfinite(ret_vals)], [0.33, 0.67])
        secondary = MomentumQuantileClassifier(col_index=mom_idx, low=float(lo), high=float(hi))
        print(f"  Momentum secondary voter on {mom_col}")

    output_path = Path(args.output)
    save_bundle(output_path, model, feats, meta, ensemble_secondary=secondary)
    print(f"Saved model bundle to {output_path}")

    card = {**meta, "feature_names": list(feats), "data_path": args.dataset}
    save_model_card(Path(args.card), card)
    print(f"Saved model card to {args.card}")


if __name__ == "__main__":
    main()

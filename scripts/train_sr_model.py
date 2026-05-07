"""Train the S/R prediction model.

Usage: python -m scripts.train_sr_model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import ARTIFACTS
from agent.data.pipeline import load_parquet
from agent.models.sr_predictor import SRPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Train S/R prediction model")
    parser.add_argument("--dataset", default=str(ARTIFACTS / "gold_training.parquet"))
    parser.add_argument("--output", default=str(ARTIFACTS / "sr_predictor.pkl"))
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset}...")
    df = load_parquet(Path(args.dataset))
    print(f"  {len(df)} rows")

    print("Training S/R predictor...")
    model = SRPredictor()
    model.fit(df)
    model.save(Path(args.output))
    print(f"Saved S/R model to {args.output}")

    # Quick validation
    pred = model.predict(df)
    print(f"  Predicted range: {pred.predicted_range:.2f}")
    print(f"  Bounce probability: {pred.bounce_probability:.2f}")


if __name__ == "__main__":
    main()

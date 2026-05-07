"""Train all models: LightGBM (daily, 15m, 1h, 4h), S/R predictor, regime detector, GARCH volatility.

Usage: python -m scripts.train_all_models
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import ARTIFACTS
from agent.data.pipeline import load_parquet, feature_columns_present
from agent.data.sr_features import SR_FEATURE_COLUMNS
from agent.models.lgbm_model import train_model, save_bundle, save_model_card
from agent.models.ensemble import MomentumQuantileClassifier
from agent.models.regime import GMMRegimeDetector
from agent.models.volatility import GARCHVolatility
from agent.models.sr_predictor import SRPredictor


def train_timeframe_model(
    dataset_path: Path,
    output_path: Path,
    card_path: Path,
    *,
    random_state: int = 42,
) -> bool:
    """Train a model for a single timeframe. Returns True if successful."""
    if not dataset_path.exists():
        print(f"  SKIP: {dataset_path.name} not found")
        return False

    df = load_parquet(dataset_path)
    print(f"  Loaded {len(df)} rows from {dataset_path.name}")

    feature_names = feature_columns_present(df)
    print(f"  {len(feature_names)} features detected")

    valid = df.drop_nulls(subset=list(feature_names) + ["target"])
    print(f"  {len(valid)} valid rows after drop_nulls")

    if len(valid) < 100:
        print(f"  SKIP: too few valid rows ({len(valid)})")
        return False

    model, feats, meta = train_model(valid, feature_names=feature_names, random_state=random_state)
    print(f"  Trained: {meta['backend']}, classes={meta['classes']}")

    # Momentum secondary voter
    mom_col = None
    for candidate in ("ret_21", "ret_b26", "ret_5"):
        if candidate in feats:
            mom_col = candidate
            break

    secondary = None
    if mom_col:
        mom_idx = list(feats).index(mom_col)
        ret_vals = valid[mom_col].drop_nulls().to_numpy()
        finite = ret_vals[np.isfinite(ret_vals)]
        if len(finite) > 10:
            lo, hi = np.quantile(finite, [0.33, 0.67])
            secondary = MomentumQuantileClassifier(col_index=mom_idx, low=float(lo), high=float(hi))
            print(f"  Momentum secondary voter on {mom_col}")

    save_bundle(output_path, model, feats, meta, ensemble_secondary=secondary)
    print(f"  Saved to {output_path.name}")

    card = {**meta, "feature_names": list(feats), "data_path": str(dataset_path)}
    save_model_card(card_path, card)
    return True


def train_sr_model(dataset_path: Path, output_path: Path = ARTIFACTS / "sr_predictor.pkl") -> bool:
    """Train S/R predictor. Returns True if successful."""
    if not dataset_path.exists():
        print(f"  SKIP: {dataset_path.name} not found")
        return False

    df = load_parquet(dataset_path)
    print(f"  Loaded {len(df)} rows from {dataset_path.name}")

    if len(df) < 100:
        print(f"  SKIP: too few rows ({len(df)})")
        return False

    sr = SRPredictor()
    sr.fit(df)
    sr.save(output_path)

    pred = sr.predict(df)
    print(f"  S/R range: {pred.predicted_range:.2f}, bounce prob: {pred.bounce_probability:.2f}")
    print(f"  Saved to {output_path.name}")
    return True


def train_regime_model(dataset_path: Path, output_path: Path = ARTIFACTS / "regime_gmm.pkl") -> bool:
    """Train GMM regime detector. Returns True if successful."""
    if not dataset_path.exists():
        print(f"  SKIP: {dataset_path.name} not found")
        return False

    df = load_parquet(dataset_path)
    if len(df) < 100:
        print(f"  SKIP: too few rows")
        return False

    det = GMMRegimeDetector(n_components=4)
    det.fit(df)
    det.save(output_path)

    result = det.predict_latest(df)
    print(f"  Current regime: {result.regime_name}")
    print(f"  Probs: {result.probabilities}")
    print(f"  Saved to {output_path.name}")
    return True


def train_garch_model(dataset_path: Path) -> bool:
    """Train GARCH(1,1) volatility model. Returns True if successful."""
    if not dataset_path.exists():
        print(f"  SKIP: {dataset_path.name} not found")
        return False

    df = load_parquet(dataset_path)
    if len(df) < 100:
        print(f"  SKIP: too few rows")
        return False

    returns = df["close"].pct_change().drop_nulls().to_numpy()
    current_price = float(df["close"].tail(1).item())

    vol = GARCHVolatility()
    vol.fit(returns, price_mean=current_price)
    vol.save(ARTIFACTS / "garch_vol.pkl")

    forecast = vol.forecast(horizon=1, current_price=current_price)
    print(f"  Vol regime: {forecast.vol_regime}")
    print(f"  1d forecast: {forecast.forecast_1d:.4f}")
    print(f"  Trailing stop distance: ${forecast.trailing_stop_distance:.2f}")
    print(f"  Saved to garch_vol.pkl")
    return True


def main() -> None:
    print("=" * 60)
    print("  Training All Models")
    print("=" * 60)

    # --- Daily model (primary) ---
    print("\n[1/6] Daily LightGBM model...")
    train_timeframe_model(
        ARTIFACTS / "gold_training.parquet",
        ARTIFACTS / "gold_lgbm.pkl",
        ARTIFACTS / "model_card.json",
    )

    # --- 15m model ---
    print("\n[2/7] 15M LightGBM model...")
    train_timeframe_model(
        ARTIFACTS / "gold_15m.parquet",
        ARTIFACTS / "gold_lgbm_15m.pkl",
        ARTIFACTS / "model_card_15m.json",
    )

    # --- 1h model ---
    print("\n[3/7] 1H LightGBM model...")
    train_timeframe_model(
        ARTIFACTS / "gold_1h.parquet",
        ARTIFACTS / "gold_lgbm_1h.pkl",
        ARTIFACTS / "model_card_1h.json",
    )

    # --- 4h model ---
    print("\n[4/7] 4H LightGBM model...")
    train_timeframe_model(
        ARTIFACTS / "gold_4h.parquet",
        ARTIFACTS / "gold_lgbm_4h.pkl",
        ARTIFACTS / "model_card_4h.json",
    )

    # --- S/R predictors ---
    print("\n[5/7] S/R predictors...")
    train_sr_model(ARTIFACTS / "gold_training.parquet", ARTIFACTS / "sr_predictor.pkl")
    train_sr_model(ARTIFACTS / "gold_15m.parquet", ARTIFACTS / "sr_predictor_15m.pkl")

    # --- Regime detectors ---
    print("\n[6/7] GMM regime detectors...")
    train_regime_model(ARTIFACTS / "gold_training.parquet", ARTIFACTS / "regime_gmm.pkl")
    train_regime_model(ARTIFACTS / "gold_15m.parquet", ARTIFACTS / "regime_gmm_15m.pkl")
    train_regime_model(ARTIFACTS / "gold_1h.parquet", ARTIFACTS / "regime_gmm_1h.pkl")
    train_regime_model(ARTIFACTS / "gold_4h.parquet", ARTIFACTS / "regime_gmm_4h.pkl")

    # --- GARCH volatility ---
    print("\n[7/7] GARCH volatility model...")
    train_garch_model(ARTIFACTS / "gold_training.parquet")

    print("\n" + "=" * 60)
    print("  All models trained!")
    print("=" * 60)


if __name__ == "__main__":
    main()

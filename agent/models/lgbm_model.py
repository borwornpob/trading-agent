"""LightGBM model training, prediction, and persistence.

Concepts: Ch 12 (gradient boosting, trading signals), Ch 06 (walk-forward CV).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl

from agent.data.pipeline import feature_columns_present

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None  # type: ignore[misc, assignment]

from sklearn.ensemble import HistGradientBoostingClassifier

Classifier = Any


def make_classifier(
    *,
    random_state: int = 42,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
) -> tuple[str, Classifier]:
    """Return (backend_name, classifier) — prefers LightGBM, falls back to sklearn."""
    if LGBMClassifier is not None:
        return (
            "lightgbm",
            LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=-1,
                num_leaves=31,
                min_child_samples=15,
                class_weight="balanced",
                random_state=random_state,
                verbose=-1,
            ),
        )
    return (
        "sklearn_hist_gbm",
        HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=learning_rate,
            max_iter=n_estimators,
            random_state=random_state,
        ),
    )


def train_model(
    df: pl.DataFrame,
    *,
    feature_names: tuple[str, ...] | None = None,
    random_state: int = 42,
) -> tuple[Classifier, tuple[str, ...], dict[str, Any]]:
    """Train a classifier on the full dataframe. Returns (model, feature_names, meta)."""
    df = df.sort("timestamp").drop_nulls(subset=["target"])
    if feature_names is None:
        feature_names = feature_columns_present(df)

    X = _to_X(df, feature_names)
    y = df["target"].to_numpy()

    backend, clf = make_classifier(random_state=random_state)
    clf.fit(X, y)

    meta = {
        "backend": backend,
        "n_features": len(feature_names),
        "n_samples": len(df),
        "classes": sorted(np.unique(y).tolist()),
        "random_state": random_state,
    }
    return clf, feature_names, meta


def predict_table(
    model: Classifier,
    df: pl.DataFrame,
    feature_names: tuple[str, ...],
) -> pl.DataFrame:
    """Run prediction on a DataFrame, return columns: timestamp, symbol, pred_class, p_down, p_neutral, p_up, score."""
    X = _to_X(df, feature_names)
    pred = np.asarray(model.predict(X))
    proba = np.asarray(model.predict_proba(X), dtype=np.float64)
    proba = semantic_probability_columns(model, proba)
    pred = normalize_class_predictions(model, pred)

    out = df.select("timestamp")
    if "symbol" in df.columns:
        out = out.with_columns(df["symbol"])
    else:
        out = out.with_columns(pl.lit("GC=F").alias("symbol"))

    out = out.with_columns([
        pl.Series("pred_class", pred.astype(int)),
        pl.Series("p_down", proba[:, 0]),
        pl.Series("p_neutral", proba[:, 1]),
        pl.Series("p_up", proba[:, 2]),
        pl.Series("score", proba[:, 2] - proba[:, 0]),
    ])
    return out


def save_bundle(
    path: Path,
    model: Classifier,
    feature_names: tuple[str, ...],
    meta: dict[str, Any],
    *,
    ensemble_secondary: Classifier | None = None,
) -> None:
    """Save model bundle as joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_names": list(feature_names),
        "meta": meta,
        "ensemble_secondary": ensemble_secondary,
    }
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> dict[str, Any]:
    """Load full model bundle."""
    raw = joblib.load(path)
    if isinstance(raw, dict) and "model" in raw:
        return raw
    return {"model": raw, "feature_names": [], "meta": {}, "ensemble_secondary": None}


def normalize_class_predictions(model: Classifier, pred: np.ndarray) -> np.ndarray:
    """Normalize model labels to 0=down, 1=neutral, 2=up."""
    classes = _model_classes(model, len(np.unique(pred)))
    mapping = _semantic_class_mapping(classes)
    return np.asarray([mapping.get(int(label), int(label)) for label in pred], dtype=np.int64)


def semantic_probability_columns(model: Classifier, proba: np.ndarray) -> np.ndarray:
    """Return probability columns ordered as down, neutral, up."""
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2:
        raise ValueError("predict_proba output must be a 2D array")

    classes = _model_classes(model, proba.shape[1])
    mapping = _semantic_class_mapping(classes)
    out = np.zeros((proba.shape[0], 3), dtype=np.float64)

    for idx, class_label in enumerate(classes[: proba.shape[1]]):
        semantic_idx = mapping.get(int(class_label))
        if semantic_idx is not None and 0 <= semantic_idx <= 2:
            out[:, semantic_idx] = proba[:, idx]

    if not out.any() and proba.shape[1] >= 3:
        out = proba[:, :3]
    elif not out.any() and proba.shape[1] == 2:
        out = np.column_stack([proba[:, 0], np.zeros(proba.shape[0]), proba[:, 1]])

    return out


def _model_classes(model: Classifier, n_columns: int) -> list[int]:
    raw_classes = getattr(model, "classes_", None)
    if raw_classes is None:
        return list(range(n_columns))
    return [int(c) for c in list(raw_classes)]


def _semantic_class_mapping(classes: list[int]) -> dict[int, int]:
    class_set = set(classes)
    if class_set == {1, 2, 3}:
        return {1: 0, 2: 1, 3: 2}
    if class_set == {0, 1, 2}:
        return {0: 0, 1: 1, 2: 2}
    return {label: idx for idx, label in enumerate(sorted(classes)) if idx <= 2}


def save_model_card(path: Path, card: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")


def _to_X(df: pl.DataFrame, feature_names: tuple[str, ...]) -> pd.DataFrame | np.ndarray:
    """Convert feature columns to model input. Uses pandas for LGBM, numpy otherwise."""
    sub = _aligned_feature_frame(df, feature_names)

    try:
        if LGBMClassifier is not None:
            X = sub.to_pandas()
            X.columns = list(feature_names)
            return X
    except Exception:
        pass

    return sub.to_numpy()


def _aligned_feature_frame(df: pl.DataFrame, feature_names: tuple[str, ...]) -> pl.DataFrame:
    """Return exactly the trained feature columns, filling live-missing features with zero."""
    return df.select([
        pl.col(c).fill_null(0).cast(pl.Float64, strict=False).alias(c)
        if c in df.columns
        else pl.lit(0.0).alias(c)
        for c in feature_names
    ])

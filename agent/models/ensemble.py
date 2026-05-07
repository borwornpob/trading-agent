"""Ensemble prediction: blend primary classifier with secondary voter and multi-timeframe signals.

Concepts: Ch 11 (ensemble methods), Ch 12 (boosting baselines).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from agent.models.lgbm_model import _aligned_feature_frame, predict_table

Classifier = Any


class MomentumQuantileClassifier:
    """3-class rule: low / mid / high vs train-set quantiles of a momentum column."""

    def __init__(self, *, col_index: int, low: float, high: float) -> None:
        self.col_index = col_index
        self.low = low
        self.high = high

    def predict(self, X: np.ndarray) -> np.ndarray:
        r = X[:, self.col_index]
        out = np.ones(len(r), dtype=np.int64)
        out[r < self.low] = 0
        out[r > self.high] = 2
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.full((X.shape[0], 3), 0.075)
        pred = self.predict(X)
        p[np.arange(len(pred)), pred] = 0.85
        p /= p.sum(axis=1, keepdims=True)
        return p


def predict_ensemble(
    primary: Classifier,
    df: pl.DataFrame,
    feature_names: tuple[str, ...],
    secondary: Classifier | None = None,
    *,
    primary_weight: float = 0.65,
) -> pl.DataFrame:
    """Blend primary and secondary classifier probabilities."""
    p1 = _get_proba(primary, df, feature_names)

    if secondary is None or primary_weight >= 1.0:
        return predict_table(primary, df, feature_names)

    w = max(0.0, min(1.0, primary_weight))
    p2 = _get_proba(secondary, df, feature_names)

    proba = w * p1 + (1.0 - w) * p2
    row_sums = proba.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    proba = proba / row_sums

    pred = np.argmax(proba, axis=1)

    out = df.select("timestamp")
    if "symbol" in df.columns:
        out = out.with_columns(df["symbol"])
    else:
        out = out.with_columns(pl.lit("GC=F").alias("symbol"))

    return out.with_columns([
        pl.Series("pred_class", pred.astype(int)),
        pl.Series("p_down", proba[:, 0]),
        pl.Series("p_neutral", proba[:, 1]),
        pl.Series("p_up", proba[:, 2]),
        pl.Series("score", proba[:, 2] - proba[:, 0]),
    ])


def predict_mtf(
    predictions: list[pl.DataFrame],
    *,
    weights: list[float] | None = None,
) -> pl.DataFrame:
    """Multi-timeframe ensemble: weighted average of predictions from different bar frequencies.

    Each prediction DataFrame must have columns: timestamp, pred_class, p_up, p_down, p_neutral, score.
    Returns a single consensus prediction.
    """
    if not predictions:
        raise ValueError("Need at least one prediction DataFrame")

    if len(predictions) == 1:
        return predictions[0]

    n = len(predictions)
    weights = weights or [1.0 / n] * n
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    # Stack probabilities
    probas = []
    for pdf in predictions:
        p_down = pdf["p_down"].to_numpy()
        p_neutral = pdf["p_neutral"].to_numpy()
        p_up = pdf["p_up"].to_numpy()
        probas.append(np.column_stack([p_down, p_neutral, p_up]))

    blended = sum(w * p for w, p in zip(weights, probas))
    pred = np.argmax(blended, axis=1)

    base = predictions[0]
    return base.with_columns([
        pl.Series("pred_class", pred.astype(int)),
        pl.Series("p_down", blended[:, 0]),
        pl.Series("p_neutral", blended[:, 1]),
        pl.Series("p_up", blended[:, 2]),
        pl.Series("score", blended[:, 2] - blended[:, 0]),
    ])


def _get_proba(model: Classifier, df: pl.DataFrame, feature_names: tuple[str, ...]) -> np.ndarray:
    """Extract class probabilities from a model."""
    sub = _aligned_feature_frame(df, feature_names)

    if isinstance(model, MomentumQuantileClassifier):
        X = sub.to_numpy()
    elif type(model).__name__ in ("LGBMClassifier", "LogisticRegression"):
        X = sub.to_pandas()
        X.columns = list(feature_names)
    else:
        X = sub.to_numpy()

    return np.asarray(model.predict_proba(X), dtype=np.float64)

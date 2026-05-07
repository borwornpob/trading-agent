"""Support/Resistance level and daily range prediction model.

Concepts: Ch 04 (alpha factors), Ch 12 (gradient boosting), Ch 09 (time series).
Predicts: daily_high, daily_low, bounce_probability at key levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from agent.data.sr_features import SR_FEATURE_COLUMNS

Regressor = Any


@dataclass(frozen=True)
class SRPrediction:
    predicted_high: float
    predicted_low: float
    predicted_range: float
    pivot_pp: float
    support_levels: list[float]
    resistance_levels: list[float]
    bounce_probability: float  # 0-1 at nearest support/resistance


class SRPredictor:
    """LightGBM regressors for daily high/low prediction + bounce classifier."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self._high_model: Regressor | None = None
        self._low_model: Regressor | None = None
        self._bounce_model: Any = None
        self._feature_names: tuple[str, ...] = ()
        self._fitted: bool = False

    def fit(self, df: pl.DataFrame, *, feature_names: tuple[str, ...] | None = None) -> None:
        """Train high/low regressors and bounce classifier."""
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

        if feature_names is not None:
            self._feature_names = feature_names
        else:
            self._feature_names = tuple(c for c in SR_FEATURE_COLUMNS if c in df.columns)

        present = [c for c in self._feature_names if c in df.columns]
        self._feature_names = tuple(present)

        X = df.select(self._feature_names).fill_null(0).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Targets: next bar's high and low
        y_high = df["high"].shift(-1).drop_nulls().to_numpy()
        y_low = df["low"].shift(-1).drop_nulls().to_numpy()
        n = min(len(X) - 1, len(y_high), len(y_low))
        X_train = X[:n]
        y_high_train = y_high[:n]
        y_low_train = y_low[:n]

        self._high_model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=self.random_state,
        )
        self._high_model.fit(X_train, y_high_train)

        self._low_model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=self.random_state,
        )
        self._low_model.fit(X_train, y_low_train)

        # Bounce target: did next bar's low hold above current support?
        close = df["close"].to_numpy()[:n]
        prev_low = df["low"].to_numpy()[:n]
        next_low = y_low_train
        bounce = (next_low >= prev_low).astype(int)

        self._bounce_model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=self.random_state,
        )
        self._bounce_model.fit(X_train, bounce)

        self._fitted = True

    def predict(self, df: pl.DataFrame, current_price: float | None = None) -> SRPrediction:
        """Predict S/R levels for the next bar."""
        if not self._fitted:
            raise RuntimeError("Model not fitted — call fit() first")

        last = df.tail(1)
        X = last.select(self._feature_names).fill_null(0).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        pred_high = float(self._high_model.predict(X)[0])
        pred_low = float(self._low_model.predict(X)[0])
        bounce_prob = float(self._bounce_model.predict_proba(X)[0, 1])

        price = current_price or float(last["close"].item())
        pivot_pp = float((pred_high + pred_low + price) / 3)

        support_levels = sorted([
            pred_low,
            2 * pivot_pp - pred_high,
            pivot_pp - (pred_high - pred_low),
        ])

        resistance_levels = sorted([
            pred_high,
            2 * pivot_pp - pred_low,
            pivot_pp + (pred_high - pred_low),
        ])

        return SRPrediction(
            predicted_high=pred_high,
            predicted_low=pred_low,
            predicted_range=pred_high - pred_low,
            pivot_pp=pivot_pp,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            bounce_probability=bounce_prob,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "high_model": self._high_model,
            "low_model": self._low_model,
            "bounce_model": self._bounce_model,
            "feature_names": list(self._feature_names),
            "random_state": self.random_state,
        }, path)

    @classmethod
    def load(cls, path: Path) -> SRPredictor:
        data = joblib.load(path)
        pred = cls(random_state=data.get("random_state", 42))
        pred._high_model = data["high_model"]
        pred._low_model = data["low_model"]
        pred._bounce_model = data["bounce_model"]
        pred._feature_names = tuple(data.get("feature_names", []))
        pred._fitted = True
        return pred

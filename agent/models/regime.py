"""Market regime detection via Gaussian Mixture Model or Hidden Markov Model.

Concepts: Ch 13 (unsupervised learning, clustering), Ch 09 (time series structure).
Regimes: trending_up, trending_down, ranging, volatile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl


REGIME_LABELS = {
    0: "ranging",
    1: "trending_up",
    2: "trending_down",
    3: "volatile",
}

REGIME_FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_1", "ret_b1", "ret_b4", "ret_b26", "atr", "adx", "rsi", "cci",
    "stoch_k", "obv", "volume",
)


@dataclass(frozen=True)
class RegimeResult:
    regime_id: int
    regime_name: str
    probabilities: dict[str, float]


class GMMRegimeDetector:
    """Gaussian Mixture Model for regime classification.

    Trained on TA features to classify market into regimes.
    """

    def __init__(self, n_components: int = 4, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self._model: Any = None
        self._feature_names: tuple[str, ...] = REGIME_FEATURE_COLUMNS
        self._label_map: dict[int, str] = {}

    def fit(self, df: pl.DataFrame, *, feature_names: tuple[str, ...] | None = None) -> None:
        from sklearn.mixture import GaussianMixture

        if feature_names is not None:
            self._feature_names = feature_names
        present = [c for c in self._feature_names if c in df.columns]
        self._feature_names = tuple(present)

        X = df.select(self._feature_names).fill_null(0).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        self._model = GaussianMixture(
            n_components=self.n_components,
            random_state=self.random_state,
            covariance_type="full",
        )
        self._model.fit(X)
        self._infer_labels(df, X)

    def _infer_labels(self, df: pl.DataFrame, X: np.ndarray) -> None:
        """Map GMM cluster IDs to semantic regime names based on cluster statistics."""
        labels = self._model.predict(X)
        returns = df["close"].pct_change().fill_null(0).to_numpy()

        stats = {}
        for i in range(self.n_components):
            mask = labels == i
            if mask.sum() == 0:
                stats[i] = {"mean_ret": 0.0, "vol": 0.0, "adx_mean": 0.0}
                continue
            cluster_returns = returns[mask]
            adx_col = "adx" if "adx" in df.columns else None
            adx_vals = df["adx"].to_numpy()[mask] if adx_col else np.ones(mask.sum()) * 20
            stats[i] = {
                "mean_ret": float(np.mean(cluster_returns)),
                "vol": float(np.std(cluster_returns)),
                "adx_mean": float(np.mean(adx_vals)),
            }

        ranked_by_ret = sorted(stats.items(), key=lambda x: x[1]["mean_ret"])
        ranked_by_vol = sorted(stats.items(), key=lambda x: x[1]["vol"], reverse=True)

        self._label_map = {}
        assigned = set()

        # Highest mean return = trending_up
        for idx, _ in reversed(ranked_by_ret):
            if idx not in assigned:
                self._label_map[idx] = "trending_up"
                assigned.add(idx)
                break

        # Lowest mean return = trending_down
        for idx, _ in ranked_by_ret:
            if idx not in assigned:
                self._label_map[idx] = "trending_down"
                assigned.add(idx)
                break

        # Highest volatility = volatile
        for idx, _ in ranked_by_vol:
            if idx not in assigned:
                self._label_map[idx] = "volatile"
                assigned.add(idx)
                break

        # Remaining = ranging
        for i in range(self.n_components):
            if i not in assigned:
                self._label_map[i] = "ranging"

    def predict(self, df: pl.DataFrame) -> list[RegimeResult]:
        if self._model is None:
            raise RuntimeError("Model not fitted — call fit() first")

        present = [c for c in self._feature_names if c in df.columns]
        X = df.select(present).fill_null(0).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        labels = self._model.predict(X)
        probs = self._model.predict_proba(X)

        results = []
        for i in range(len(df)):
            label = int(labels[i])
            regime_name = self._label_map.get(label, "unknown")
            prob_dict = {
                self._label_map.get(j, f"cluster_{j}"): float(probs[i, j])
                for j in range(self.n_components)
            }
            results.append(RegimeResult(
                regime_id=label,
                regime_name=regime_name,
                probabilities=prob_dict,
            ))
        return results

    def predict_latest(self, df: pl.DataFrame) -> RegimeResult:
        results = self.predict(df.tail(1))
        return results[0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self._model,
            "n_components": self.n_components,
            "random_state": self.random_state,
            "feature_names": list(self._feature_names),
            "label_map": self._label_map,
        }, path)

    @classmethod
    def load(cls, path: Path) -> GMMRegimeDetector:
        data = joblib.load(path)
        det = cls(
            n_components=data.get("n_components", 4),
            random_state=data.get("random_state", 42),
        )
        det._model = data["model"]
        det._feature_names = tuple(data.get("feature_names", list(REGIME_FEATURE_COLUMNS)))
        det._label_map = data.get("label_map", {})
        return det

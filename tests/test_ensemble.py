"""Tests for ensemble prediction."""

import numpy as np
import polars as pl
import pytest
from agent.models.ensemble import MomentumQuantileClassifier, predict_ensemble
from agent.models.lgbm_model import predict_table


class OneTwoThreeModel:
    classes_ = np.array([1, 2, 3])

    def predict(self, X):
        return np.array([3, 2, 1])[: len(X)]

    def predict_proba(self, X):
        return np.array([
            [0.05, 0.10, 0.85],
            [0.10, 0.80, 0.10],
            [0.70, 0.20, 0.10],
        ])[: len(X)]


def test_momentum_classifier():
    clf = MomentumQuantileClassifier(col_index=0, low=-0.01, high=0.01)
    X = np.array([[-0.02], [0.0], [0.03]])
    preds = clf.predict(X)
    assert preds[0] == 0  # Below low
    assert preds[1] == 1  # In range
    assert preds[2] == 2  # Above high
    proba = clf.predict_proba(X)
    assert proba.shape == (3, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_predict_ensemble_no_secondary():
    """When secondary is None, should just return primary predictions."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.choice([0, 1, 2], size=100)
    model = HistGradientBoostingClassifier(max_iter=10, random_state=42)
    model.fit(X, y)

    df = pl.DataFrame({
        "timestamp": pl.Series(["2024-01-01"]).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"),
        "f0": [0.1], "f1": [0.2], "f2": [0.3], "f3": [0.4], "f4": [0.5],
    })
    result = predict_ensemble(model, df, ("f0", "f1", "f2", "f3", "f4"), None)
    assert "pred_class" in result.columns
    assert "p_up" in result.columns


def test_predict_ensemble_fills_missing_live_features():
    """Live inference should preserve trained feature shape when some columns are absent."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    np.random.seed(7)
    X = np.random.randn(100, 5)
    y = np.random.choice([0, 1, 2], size=100)
    model = HistGradientBoostingClassifier(max_iter=10, random_state=7)
    model.fit(X, y)

    df = pl.DataFrame({
        "timestamp": pl.Series(["2024-01-01"]).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"),
        "f0": [0.1], "f1": [0.2], "f2": [0.3], "f3": [0.4],
    })
    result = predict_ensemble(model, df, ("f0", "f1", "f2", "f3", "f4"), None)

    assert "pred_class" in result.columns
    assert result.height == 1


def test_predict_table_normalizes_one_two_three_labels():
    df = pl.DataFrame({
        "timestamp": pl.Series(["2024-01-01", "2024-01-02", "2024-01-03"]).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"),
        "f0": [0.1, 0.2, 0.3],
    })

    result = predict_table(OneTwoThreeModel(), df, ("f0",))

    assert result["pred_class"].to_list() == [2, 1, 0]
    assert result["p_down"].to_list() == [0.05, 0.10, 0.70]
    assert result["p_neutral"].to_list() == [0.10, 0.80, 0.20]
    assert result["p_up"].to_list() == [0.85, 0.10, 0.10]

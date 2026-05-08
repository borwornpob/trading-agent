"""Train and audit MT5-specific models from broker candle artifacts.

Usage:
    python -m scripts.train_mt5_models
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import ARTIFACTS
from agent.data.pipeline import feature_columns_present, load_parquet
from agent.models.ensemble import MomentumQuantileClassifier, predict_ensemble
from agent.models.lgbm_model import save_bundle, save_model_card, train_model
from agent.models.regime import GMMRegimeDetector
from agent.models.sr_predictor import SRPredictor
from agent.strategy.gold_strategy import signal_from_prediction
from agent.strategy.signal_gates import run_all_gates


def _secondary_voter(df: pl.DataFrame, feats: tuple[str, ...]) -> MomentumQuantileClassifier | None:
    for candidate in ("ret_21", "ret_b26", "ret_5"):
        if candidate not in feats:
            continue
        values = df[candidate].drop_nulls().to_numpy()
        finite = values[np.isfinite(values)]
        if len(finite) <= 10:
            continue
        low, high = np.quantile(finite, [0.33, 0.67])
        return MomentumQuantileClassifier(
            col_index=list(feats).index(candidate),
            low=float(low),
            high=float(high),
        )
    return None


def _train_classifier(
    df: pl.DataFrame,
    *,
    output_path: Path,
    card_path: Path,
    data_path: Path,
) -> dict[str, Any]:
    feature_names = feature_columns_present(df)
    valid = df.drop_nulls(subset=list(feature_names) + ["target"])
    model, feats, meta = train_model(valid, feature_names=feature_names)
    secondary = _secondary_voter(valid, feats)
    save_bundle(output_path, model, feats, meta, ensemble_secondary=secondary)
    save_model_card(card_path, {**meta, "feature_names": list(feats), "data_path": str(data_path)})
    return {"model": model, "feature_names": feats, "secondary": secondary, "meta": meta}


def _signal_audit(
    df: pl.DataFrame,
    *,
    model: Any,
    feature_names: tuple[str, ...],
    secondary: Any,
    regime_model: GMMRegimeDetector | None,
) -> dict[str, Any]:
    preds = predict_ensemble(model, df, feature_names, secondary, primary_weight=0.65)
    if regime_model is not None:
        regimes = [r.regime_name for r in regime_model.predict(df)]
    else:
        regimes = ["unknown"] * len(df)

    raw_classes = Counter(preds["pred_class"].to_list())
    dirs = Counter()
    gate_pass = Counter()
    gate_fail = Counter()
    pmax = []
    scores = []
    for row, regime in zip(preds.iter_rows(named=True), regimes):
        signal = signal_from_prediction(row, regime=regime, sentiment_score=0.0, event_risk=False)
        dirs[signal.direction] += 1
        pmax.append(max(signal.p_up, signal.p_down))
        scores.append(abs(signal.score))
        passed, gates = run_all_gates(
            signal.direction,
            signal.p_up,
            signal.p_down,
            signal.score,
            0.0,
            regime,
        )
        if signal.direction != "flat" and passed:
            gate_pass[signal.direction] += 1
        elif signal.direction != "flat":
            gate_pass["blocked"] += 1
            for gate in gates:
                if not gate.passed:
                    gate_fail[gate.reason] += 1

    def pct(count: int) -> float:
        return round(100.0 * count / max(len(df), 1), 2)

    return {
        "rows": len(df),
        "raw_model_classes": dict(raw_classes),
        "raw_trade_pct": pct(raw_classes.get(0, 0) + raw_classes.get(2, 0)),
        "strategy_signal_counts": dict(dirs),
        "strategy_trade_pct": pct(dirs["long"] + dirs["short"]),
        "strategy_flat_pct": pct(dirs["flat"]),
        "gate_pass_counts": dict(gate_pass),
        "gate_pass_trade_pct": pct(gate_pass["long"] + gate_pass["short"]),
        "top_gate_fail_reasons": gate_fail.most_common(10),
        "regime_counts": dict(Counter(regimes)),
        "pmax_quantiles": {
            str(q): round(float(pl.Series(pmax).quantile(q)), 4)
            for q in (0.5, 0.75, 0.9, 0.95, 0.99)
        },
        "abs_score_quantiles": {
            str(q): round(float(pl.Series(scores).quantile(q)), 4)
            for q in (0.5, 0.75, 0.9, 0.95, 0.99)
        },
        "last_prediction": preds.tail(1).to_dicts()[0] if len(preds) else None,
    }


def main() -> None:
    data_path = ARTIFACTS / "gold_mt5_15m.parquet"
    df = load_parquet(data_path).sort("timestamp")
    if len(df) < 500:
        raise SystemExit(f"Need at least 500 MT5 rows, got {len(df)}")

    split_idx = max(300, int(len(df) * 0.70))
    train_df = df.slice(0, split_idx)
    holdout_df = df.slice(split_idx)

    regime = GMMRegimeDetector(n_components=4)
    regime.fit(train_df)
    regime.save(ARTIFACTS / "regime_gmm_mt5_15m.pkl")

    sr = SRPredictor()
    sr.fit(train_df)
    sr.save(ARTIFACTS / "sr_predictor_mt5_15m.pkl")

    holdout_model = _train_classifier(
        train_df,
        output_path=ARTIFACTS / "gold_lgbm_mt5_15m_holdout.pkl",
        card_path=ARTIFACTS / "model_card_mt5_15m_holdout.json",
        data_path=data_path,
    )
    full_model = _train_classifier(
        df,
        output_path=ARTIFACTS / "gold_lgbm_mt5_15m.pkl",
        card_path=ARTIFACTS / "model_card_mt5_15m.json",
        data_path=data_path,
    )

    summary = {
        "data_path": str(data_path),
        "rows": len(df),
        "start": str(df["timestamp"][0]),
        "end": str(df["timestamp"][-1]),
        "target_distribution": dict(Counter(df["target"].to_list())),
        "holdout": _signal_audit(
            holdout_df,
            model=holdout_model["model"],
            feature_names=holdout_model["feature_names"],
            secondary=holdout_model["secondary"],
            regime_model=regime,
        ),
        "full_in_sample": _signal_audit(
            df,
            model=full_model["model"],
            feature_names=full_model["feature_names"],
            secondary=full_model["secondary"],
            regime_model=regime,
        ),
        "artifacts": {
            "model": str(ARTIFACTS / "gold_lgbm_mt5_15m.pkl"),
            "sr": str(ARTIFACTS / "sr_predictor_mt5_15m.pkl"),
            "regime": str(ARTIFACTS / "regime_gmm_mt5_15m.pkl"),
        },
    }
    out_path = ARTIFACTS / "mt5_model_audit.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

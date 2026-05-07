"""Backtest API endpoints."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

import polars as pl
from agent.backtest.engine import run_grid_backtest, run_sl_tp_backtest
from agent.config import AgentConfig
from agent.data.pipeline import (
    build_feature_table,
    feature_columns_present,
    fetch_ohlcv_yahoo,
)
from agent.data.sr_features import compute_all_sr_features
from agent.models.lgbm_model import load_bundle, predict_table
from agent.models.sr_predictor import SRPredictor
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["backtest"])

_backtest_results: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _to_json_val(v: Any) -> Any:
    """Convert a single value to a JSON-safe representation."""
    if v is None:
        return None
    if isinstance(v, (int, float, str, bool)):
        if isinstance(v, float) and (v != v):  # NaN check
            return None
        return v
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, date):
        return v.isoformat()
    return str(v)


def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Serialize a single row dict to JSON-safe values."""
    return {k: _to_json_val(v) for k, v in row.items()}


def _build_equity_points(
    equity_curve: list[float], timestamps: list[Any]
) -> list[dict[str, Any]]:
    """Align equity values to the bars actually used by the backtest."""
    if not equity_curve or not timestamps:
        return []

    values = equity_curve
    if len(equity_curve) == len(timestamps) + 1:
        values = equity_curve[1:]

    n = min(len(values), len(timestamps))
    return [
        {"timestamp": _to_json_val(timestamps[i]), "equity": float(values[i])}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class BacktestRequest(BaseModel):
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    bar_frequency: str = "daily"
    initial_cash: float = 100_000.0
    commission_rate: float = 0.0002
    units: float = 1.0
    grid_enabled: bool = True
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 1.5
    sr_cap_tp: bool = True
    grid_max_levels: int = 3
    grid_sizing_decay: float = 0.7
    grid_atr_stop_mult: float = 1.0


# ---------------------------------------------------------------------------
# POST /backtest/run — run full backtest
# ---------------------------------------------------------------------------


@router.post("/backtest/run")
async def run_backtest(req: BacktestRequest) -> dict[str, Any]:
    """Trigger a backtest with given parameters using SL/TP engine."""
    try:
        config = AgentConfig.from_env()

        # ------------------------------------------------------------------
        # 1. Fetch OHLCV data
        # ------------------------------------------------------------------
        raw = fetch_ohlcv_yahoo(
            config.yahoo_symbol,
            req.start_date,
            req.end_date,
            frequency=req.bar_frequency,
        )

        if raw.is_empty():
            return {"error": "No OHLCV data returned for the given date range."}

        # ------------------------------------------------------------------
        # 2. Build features + S/R features
        # ------------------------------------------------------------------
        features = compute_all_sr_features(
            build_feature_table(raw, bar_frequency=req.bar_frequency)
        )

        # ------------------------------------------------------------------
        # 3. Load model & predict
        # ------------------------------------------------------------------
        try:
            bundle = load_bundle(config.model_path)
        except Exception:
            return {"error": "No trained model found. Run train_model.py first."}

        model = bundle["model"]
        feat_names = tuple(
            bundle.get("feature_names", list(feature_columns_present(features)))
        )

        valid = features.drop_nulls(subset=list(feat_names) + ["target"])
        if valid.is_empty():
            return {"error": "No valid rows after feature engineering"}

        signals = predict_table(model, valid, feat_names)

        # ------------------------------------------------------------------
        # 4. Load S/R predictor and add predicted_high / predicted_low
        # ------------------------------------------------------------------
        sr_predictor: SRPredictor | None = None
        try:
            sr_predictor = SRPredictor.load(config.sr_model_path)
        except Exception:
            # S/R predictor is optional — continue without it
            pass

        if sr_predictor is not None:
            sr_feature_names = sr_predictor._feature_names
            sr_features_df = features.select(
                ["timestamp"] + [c for c in sr_feature_names if c in features.columns]
            )
            sr_features_df = sr_features_df.fill_null(0)

            # Build predictions for each row by running the regressor over the
            # feature matrix directly (vectorised for speed).
            import numpy as np

            X_sr = sr_features_df.select(
                [c for c in sr_feature_names if c in sr_features_df.columns]
            ).to_numpy()
            X_sr = np.nan_to_num(X_sr, nan=0.0, posinf=0.0, neginf=0.0)

            pred_highs = sr_predictor._high_model.predict(X_sr)
            pred_lows = sr_predictor._low_model.predict(X_sr)

            sr_pred_df = sr_features_df.select("timestamp").with_columns(
                [
                    pl.Series("predicted_high", pred_highs),
                    pl.Series("predicted_low", pred_lows),
                ]
            )
            signals = signals.join(sr_pred_df, on="timestamp", how="left")

        # ------------------------------------------------------------------
        # 5. Run backtest (grid or simple SL/TP)
        # ------------------------------------------------------------------
        if req.grid_enabled:
            result = run_grid_backtest(
                signals,
                raw,
                initial_cash=req.initial_cash,
                commission_rate=req.commission_rate,
                units=req.units,
                sl_atr_mult=req.sl_atr_mult,
                tp_atr_mult=req.tp_atr_mult,
                sr_cap_tp=req.sr_cap_tp,
                grid_max_levels=req.grid_max_levels,
                grid_sizing_decay=req.grid_sizing_decay,
                grid_atr_stop_mult=req.grid_atr_stop_mult,
            )
        else:
            result = run_sl_tp_backtest(
                signals,
                raw,
                initial_cash=req.initial_cash,
                commission_rate=req.commission_rate,
                units=req.units,
                sl_atr_mult=req.sl_atr_mult,
                tp_atr_mult=req.tp_atr_mult,
                sr_cap_tp=req.sr_cap_tp,
            )

        # ------------------------------------------------------------------
        # 6. Serialize OHLCV
        # ------------------------------------------------------------------
        ohlcv_rows = raw.select(
            ["timestamp", "open", "high", "low", "close"]
        ).to_dicts()
        ohlcv_out = [_serialize_row(r) for r in ohlcv_rows]

        # ------------------------------------------------------------------
        # 7. Serialize signals
        # ------------------------------------------------------------------
        signal_cols = [
            c
            for c in ["timestamp", "pred_class", "score", "p_up", "p_down"]
            if c in signals.columns
        ]
        signal_rows = signals.select(signal_cols).to_dicts()
        signals_out = [_serialize_row(r) for r in signal_rows]

        # ------------------------------------------------------------------
        # 8. Serialize trades
        # ------------------------------------------------------------------
        trades_out = [_serialize_row(t) for t in result.trades]

        # ------------------------------------------------------------------
        # 9. Equity curve → list[float]
        # ------------------------------------------------------------------
        equity_curve_out = [float(v) for v in result.equity_curve]

        # Timestamped equity for charting. The backtest engine runs on the
        # signal/OHLCV inner join, so raw OHLCV row indexes are not reliable.
        equity_timestamps = (
            signals.select("timestamp")
            .join(raw.select("timestamp"), on="timestamp", how="inner")
            .sort("timestamp")["timestamp"]
            .to_list()
        )
        equity_points_out = _build_equity_points(equity_curve_out, equity_timestamps)

        # ------------------------------------------------------------------
        # 10. Build response & cache
        # ------------------------------------------------------------------
        bt_id = f"bt_{req.start_date}_{req.end_date}_{uuid.uuid4().hex[:8]}"

        payload: dict[str, Any] = {
            "id": bt_id,
            "params": req.model_dump(),
            "metrics": {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in result.metrics.items()
            },
            "n_trades": len(result.trades),
            "trades": trades_out,
            "equity_curve": equity_curve_out,
            "equity_points": equity_points_out,
            "ohlcv": ohlcv_out,
            "signals": signals_out,
        }

        # Store full results plus chart subset for the chart endpoint
        _backtest_results[bt_id] = payload

        return payload

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# GET /backtest/{bt_id}/results — retrieve cached full results
# ---------------------------------------------------------------------------


@router.get("/backtest/{bt_id}/results")
async def get_backtest_results(bt_id: str) -> dict[str, Any]:
    """Retrieve cached backtest results by ID."""
    return _backtest_results.get(bt_id, {"error": "backtest not found"})


# ---------------------------------------------------------------------------
# GET /backtest/{bt_id}/chart — lighter payload for charting
# ---------------------------------------------------------------------------


@router.get("/backtest/{bt_id}/chart")
async def get_backtest_chart(bt_id: str) -> dict[str, Any]:
    """Return chart-ready data (equity curve, OHLCV, trades) for a backtest."""
    cached = _backtest_results.get(bt_id)
    if cached is None:
        return {"error": "backtest not found"}

    return {
        "id": cached["id"],
        "equity_curve": cached.get("equity_curve", []),
        "equity_points": cached.get("equity_points", []),
        "ohlcv": cached.get("ohlcv", []),
        "trades": cached.get("trades", []),
    }

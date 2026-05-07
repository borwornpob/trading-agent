"""Grid parameter sweep: find optimal smart grid configuration.

Runs a parameter sweep over grid settings using the backtest engine
with S/R predicted levels. Compares PnL, max drawdown, and profit factor.

Usage: python -m scripts.tune_grid [--dataset artifacts/gold_training.parquet]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import ARTIFACTS
from agent.data.pipeline import load_parquet
from agent.backtest.grid_simulator import simulate_grid
from agent.backtest.metrics import compute_metrics
from agent.models.sr_predictor import SRPredictor
from agent.models.lgbm_model import load_bundle
from agent.data.pipeline import feature_columns_present


@dataclass
class GridConfig:
    max_levels: int
    sizing_decay: float
    atr_stop_multiplier: float
    commission_rate: float


@dataclass
class SweepResult:
    config: GridConfig
    total_pnl: float
    max_drawdown: float
    n_grids: int
    n_tp: int
    n_sl: int
    win_rate: float
    profit_factor: float
    avg_pnl_per_grid: float


def run_grid_backtest(
    df: pl.DataFrame,
    sr_model: SRPredictor,
    lgbm_bundle: dict,
    config: GridConfig,
    *,
    min_confidence: float = 0.3,
    regime_col: str | None = None,
) -> SweepResult:
    """Run grid simulation with given parameters over the dataset."""
    prices = df["close"].to_numpy()
    feature_names = tuple(lgbm_bundle.get("feature_names", []))
    model = lgbm_bundle["model"]

    from agent.models.lgbm_model import predict_table
    signals = predict_table(model, df, feature_names)

    # Merge signals with prices
    merged = df.with_columns([
        signals["pred_class"],
        signals["score"],
        signals["p_up"],
        signals["p_down"],
    ]).sort("timestamp")

    n = len(merged)
    close = merged["close"].to_numpy()
    atr_col = merged["atr"].to_numpy() if "atr" in merged.columns else None

    # Find grid entry points: where model has conviction and direction
    results: list[dict] = []
    grid_active = False
    grid_end = 0

    for i in range(50, n - 20):  # Need lookback and lookahead
        if i < grid_end:
            continue

        score = float(merged["score"][i])
        atr_val = float(atr_col[i]) if atr_col is not None else close[i] * 0.01

        # Entry signal: strong directional conviction
        if abs(score) < min_confidence:
            continue

        direction = "long" if score > 0 else "short"

        # Predict S/R levels for this bar
        window = merged.slice(max(0, i - 50), 51)
        try:
            sr_pred = sr_model.predict(window, current_price=float(close[i]))
        except Exception:
            continue

        # Run grid simulation
        sim = simulate_grid(
            prices=close,
            direction=direction,
            entry_index=i,
            predicted_high=sr_pred.predicted_high,
            predicted_low=sr_pred.predicted_low,
            base_size=1.0,
            max_levels=config.max_levels,
            sizing_decay=config.sizing_decay,
            commission_rate=config.commission_rate,
            atr=atr_val * config.atr_stop_multiplier,
        )

        if sim.n_levels_filled > 0:
            results.append({
                "pnl": sim.total_pnl,
                "n_filled": sim.n_levels_filled,
                "closed_tp": sim.n_grids_closed_tp,
                "closed_sl": sim.n_grids_closed_sl,
                "max_exposure": sim.max_exposure,
            })
            # Skip ahead past this grid
            # Find where grid ended
            grid_end = min(i + 30, n)  # Max 30 bars per grid

    # Aggregate results
    if not results:
        return SweepResult(
            config=config, total_pnl=0.0, max_drawdown=0.0,
            n_grids=0, n_tp=0, n_sl=0, win_rate=0.0,
            profit_factor=0.0, avg_pnl_per_grid=0.0,
        )

    pnls = np.array([r["pnl"] for r in results])
    total_pnl = float(np.sum(pnls))
    n_grids = len(results)
    n_tp = sum(r["closed_tp"] for r in results)
    n_sl = sum(r["closed_sl"] for r in results)
    wins = float(np.sum(pnls > 0))
    win_rate = wins / n_grids if n_grids > 0 else 0.0

    gross_profit = float(np.sum(pnls[pnls > 0]))
    gross_loss = abs(float(np.sum(pnls[pnls < 0])))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown from cumulative PnL
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    dd = (cum_pnl - peak)
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

    return SweepResult(
        config=config,
        total_pnl=total_pnl,
        max_drawdown=max_dd,
        n_grids=n_grids,
        n_tp=n_tp,
        n_sl=n_sl,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_pnl_per_grid=total_pnl / n_grids if n_grids > 0 else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid parameter sweep")
    parser.add_argument("--dataset", default=str(ARTIFACTS / "gold_training.parquet"))
    parser.add_argument("--model", default=str(ARTIFACTS / "gold_lgbm.pkl"))
    parser.add_argument("--sr-model", default=str(ARTIFACTS / "sr_predictor.pkl"))
    parser.add_argument("--output", default=str(ARTIFACTS / "grid_tuning_results.json"))
    args = parser.parse_args()

    print("=" * 60)
    print("  Grid Parameter Tuning")
    print("=" * 60)

    # Load data
    df = load_parquet(Path(args.dataset))
    print(f"\nDataset: {len(df)} rows")

    # Load models
    lgbm_bundle = load_bundle(Path(args.model))
    sr_model = SRPredictor.load(Path(args.sr_model))
    print(f"Model loaded: {len(lgbm_bundle.get('feature_names', []))} features")

    # Parameter grid
    param_grid = [
        GridConfig(max_levels=ml, sizing_decay=sd, atr_stop_multiplier=am, commission_rate=0.0002)
        for ml in [2, 3, 4, 5]
        for sd in [0.5, 0.6, 0.7, 0.8, 0.9]
        for am in [0.5, 1.0, 1.5, 2.0]
    ]

    print(f"\nSweeping {len(param_grid)} grid configurations...")

    results: list[SweepResult] = []
    for i, cfg in enumerate(param_grid):
        result = run_grid_backtest(df, sr_model, lgbm_bundle, cfg)
        results.append(result)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(param_grid)} done...")

    # Sort by total PnL
    results.sort(key=lambda r: r.total_pnl, reverse=True)

    # Print top 10
    print("\n" + "=" * 80)
    print(f"  {'Rank':<5} {'Levels':<8} {'Decay':<8} {'ATRStop':<8} {'PnL':>10} {'MaxDD':>10} "
          f"{'Grids':>7} {'TP':>5} {'SL':>5} {'WinR':>7} {'PF':>7}")
    print("-" * 80)

    top_results = []
    for rank, r in enumerate(results[:20], 1):
        print(
            f"  {rank:<5} {r.config.max_levels:<8} {r.config.sizing_decay:<8.1f} "
            f"{r.config.atr_stop_multiplier:<8.1f} "
            f"{r.total_pnl:>10.2f} {r.max_drawdown:>10.2f} "
            f"{r.n_grids:>7} {r.n_tp:>5} {r.n_sl:>5} "
            f"{r.win_rate:>7.2f} {r.profit_factor:>7.2f}"
        )
        top_results.append({
            "rank": rank,
            "max_levels": r.config.max_levels,
            "sizing_decay": r.config.sizing_decay,
            "atr_stop_multiplier": r.config.atr_stop_multiplier,
            "total_pnl": round(r.total_pnl, 2),
            "max_drawdown": round(r.max_drawdown, 2),
            "n_grids": r.n_grids,
            "n_tp": r.n_tp,
            "n_sl": r.n_sl,
            "win_rate": round(r.win_rate, 3),
            "profit_factor": round(r.profit_factor, 3),
            "avg_pnl_per_grid": round(r.avg_pnl_per_grid, 2),
        })

    # Save results
    output = Path(args.output)
    output.write_text(json.dumps(top_results, indent=2))
    print(f"\nResults saved to {output}")

    # Recommend best config
    if top_results:
        best = top_results[0]
        print("\n" + "=" * 60)
        print("  RECOMMENDED GRID CONFIG:")
        print(f"    GRID_MAX_LEVELS={best['max_levels']}")
        print(f"    GRID_SIZING_DECAY={best['sizing_decay']}")
        print(f"    GRID_ATR_STOP_MULT={best['atr_stop_multiplier']}")
        print(f"    Expected PnL per grid: {best['avg_pnl_per_grid']}")
        print(f"    Win rate: {best['win_rate']}")
        print(f"    Profit factor: {best['profit_factor']}")
        print("=" * 60)


if __name__ == "__main__":
    main()

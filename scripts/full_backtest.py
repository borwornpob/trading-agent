"""Full integrated backtest: multi-timeframe ensemble + regime + grid + risk.

Walk-forward evaluation of the complete trading system.

Usage: python -m scripts.full_backtest [--dataset artifacts/gold_training.parquet]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import ARTIFACTS
from agent.data.pipeline import load_parquet, feature_columns_present
from agent.models.lgbm_model import load_bundle, predict_table
from agent.models.ensemble import predict_ensemble, predict_mtf
from agent.models.regime import GMMRegimeDetector
from agent.models.sr_predictor import SRPredictor
from agent.models.volatility import GARCHVolatility
from agent.backtest.engine import (
    walk_forward_ranges,
    final_holdout_ranges,
    run_vectorized_backtest,
)
from agent.backtest.grid_simulator import simulate_grid
from agent.backtest.metrics import compute_metrics, holdout_pnl_gate
from agent.strategy.gold_strategy import signal_from_prediction
from agent.strategy.signal_gates import run_all_gates
from agent.risk.risk_guard import RiskGuard, RiskConfig


def run_full_backtest(
    df: pl.DataFrame,
    *,
    lgbm_bundle: dict,
    lgbm_1h_bundle: dict | None = None,
    sr_model: SRPredictor,
    regime_model: GMMRegimeDetector | None = None,
    vol_model: GARCHVolatility | None = None,
    use_grid: bool = True,
    grid_max_levels: int = 3,
    grid_sizing_decay: float = 0.7,
    grid_atr_stop_mult: float = 1.0,
    initial_cash: float = 100_000.0,
    commission_rate: float = 0.0002,
) -> dict:
    """Run full walk-forward backtest with all components."""
    feature_names = tuple(lgbm_bundle.get("feature_names", []))
    model = lgbm_bundle["model"]
    secondary = lgbm_bundle.get("ensemble_secondary")
    min_train = 200

    ranges = walk_forward_ranges(len(df), min_train_size=min_train, test_size=50, embargo=10)
    if not ranges:
        # Fall back to simple holdout
        hr = final_holdout_ranges(len(df), holdout_size=100, embargo=10)
        if hr:
            ranges = [hr]
        else:
            print("  Not enough data for walk-forward")
            return {}

    all_returns = []
    all_trades = []
    grid_results = []
    fold_metrics = []
    risk_guard = RiskGuard(RiskConfig(
        shadow_mode=False,
        max_daily_loss_units=initial_cash * 0.03,
    ))

    print(f"\n  Running {len(ranges)} walk-forward folds...")

    for fold_idx, split in enumerate(ranges):
        train_df = df.slice(split.train_start, split.train_end - split.train_start)
        test_df = df.slice(split.test_start, split.test_end - split.test_start)

        if len(train_df) < min_train or len(test_df) < 10:
            continue

        # --- Predictions ---
        # Primary ensemble
        preds = predict_ensemble(model, test_df, feature_names, secondary)

        # --- Regime filtering ---
        if regime_model is not None:
            try:
                regime_result = regime_model.predict_latest(train_df)
                regime_name = regime_result.regime_name
            except Exception:
                regime_name = "unknown"
        else:
            regime_name = "unknown"

        # --- Signal gates ---
        prices = test_df["close"].to_numpy()

        # Simple vectorized trading
        pred_classes = preds["pred_class"].to_numpy()
        scores = preds["score"].to_numpy()

        equity = initial_cash
        position = 0.0
        entry_price = 0.0
        returns_list = []
        trades = []

        for i in range(1, len(pred_classes)):
            prev_pred = pred_classes[i - 1]
            score = float(scores[i - 1])
            price = float(prices[i])
            prev_equity = equity

            # PnL from existing position
            if position != 0:
                prev_price = float(prices[i - 1])
                pnl = position * (price - prev_price) / prev_price * equity
                equity += pnl

            # Signal gating: skip low-conviction signals
            if abs(score) < 0.05:
                # Close position if any
                if position != 0:
                    pnl = position * (price - entry_price) / entry_price * equity
                    cost = abs(position) * equity * commission_rate
                    equity += pnl - cost
                    trades.append({"pnl": pnl - cost, "action": "close_flat"})
                    position = 0.0
                ret = (equity - prev_equity) / max(prev_equity, 1.0)
                returns_list.append(ret)
                continue

            # Determine target position
            # Classes are 0=down, 1=neutral, 2=up
            target = 0.0
            if prev_pred == 2:  # UP
                target = 1.0
            elif prev_pred == 0:  # DOWN
                target = -1.0

            # Risk guard check
            if not risk_guard.allow_order(volume=1, signed_direction=int(target))[0]:
                target = 0.0

            # Regime filter: reduce position in volatile regime
            if regime_name == "volatile":
                target *= 0.5

            if target != position:
                # Close existing
                if position != 0:
                    pnl = position * (price - entry_price) / entry_price * equity
                    cost = abs(position) * equity * commission_rate
                    equity += pnl - cost
                    trades.append({"pnl": pnl - cost, "action": "close"})
                    risk_guard.record_pnl(pnl)

                # Open new
                if target != 0:
                    entry_price = price
                    cost = abs(target) * equity * commission_rate
                    equity -= cost
                position = target

            ret = (equity - prev_equity) / max(prev_equity, 1.0)
            returns_list.append(ret)

            # Grid simulation for ranging regime
            if use_grid and regime_name == "ranging" and abs(score) > 0.3 and i < len(prices) - 20:
                try:
                    window = test_df.slice(max(0, i - 50), min(51, i + 1))
                    sr_pred = sr_model.predict(window, current_price=price)
                    atr_val = float(test_df["atr"][i]) if "atr" in test_df.columns else price * 0.01

                    grid_sim = simulate_grid(
                        prices=prices,
                        direction="long" if score > 0 else "short",
                        entry_index=i,
                        predicted_high=sr_pred.predicted_high,
                        predicted_low=sr_pred.predicted_low,
                        base_size=0.5,
                        max_levels=grid_max_levels,
                        sizing_decay=grid_sizing_decay,
                        commission_rate=commission_rate,
                        atr=atr_val * grid_atr_stop_mult,
                    )
                    if grid_sim.n_grids_opened > 0:
                        grid_results.append(grid_sim)
                except Exception:
                    pass

        fold_ret = np.array(returns_list[1:]) if len(returns_list) > 1 else np.array([0.0])
        metrics = compute_metrics(fold_ret)
        metrics["regime"] = regime_name
        metrics["n_trades"] = len(trades)
        metrics["grid_pnl"] = sum(g.total_pnl for g in grid_results)
        fold_metrics.append(metrics)

        if (fold_idx + 1) % 5 == 0:
            print(f"    Fold {fold_idx + 1}/{len(ranges)}: "
                  f"ret={metrics['total_return_pct']:.2f}%, "
                  f"sharpe={metrics['sharpe']:.3f}, "
                  f"regime={regime_name}")

    # Aggregate
    if not fold_metrics:
        return {}

    avg_metrics = {}
    for key in fold_metrics[0]:
        if isinstance(fold_metrics[0][key], (int, float)):
            avg_metrics[key] = float(np.mean([m[key] for m in fold_metrics]))
        else:
            avg_metrics[key] = fold_metrics[-1].get(key, "")

    # Grid summary
    total_grid_pnl = sum(g.total_pnl for g in grid_results)
    total_grids = len(grid_results)
    grid_tp = sum(g.n_grids_closed_tp for g in grid_results)
    grid_sl = sum(g.n_grids_closed_sl for g in grid_results)

    return {
        "directional": avg_metrics,
        "grid": {
            "total_pnl": round(total_grid_pnl, 2),
            "n_grids": total_grids,
            "n_tp": grid_tp,
            "n_sl": grid_sl,
            "avg_pnl_per_grid": round(total_grid_pnl / total_grids, 2) if total_grids > 0 else 0,
        },
        "n_folds": len(fold_metrics),
        "combined_pnl": round(avg_metrics.get("total_return_pct", 0) + total_grid_pnl, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full integrated backtest")
    parser.add_argument("--dataset", default=str(ARTIFACTS / "gold_training.parquet"))
    parser.add_argument("--model", default=str(ARTIFACTS / "gold_lgbm.pkl"))
    parser.add_argument("--sr-model", default=str(ARTIFACTS / "sr_predictor.pkl"))
    parser.add_argument("--regime-model", default=str(ARTIFACTS / "regime_gmm.pkl"))
    parser.add_argument("--vol-model", default=str(ARTIFACTS / "garch_vol.pkl"))
    parser.add_argument("--output", default=str(ARTIFACTS / "full_backtest_results.json"))
    parser.add_argument("--grid-levels", type=int, default=3)
    parser.add_argument("--grid-decay", type=float, default=0.7)
    parser.add_argument("--grid-atr-stop", type=float, default=1.0)
    parser.add_argument("--no-grid", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Full Integrated Backtest")
    print("=" * 60)

    # Load data
    df = load_parquet(Path(args.dataset))
    feature_names = feature_columns_present(df)
    print(f"\nDataset: {len(df)} rows, {len(feature_names)} features")
    print(f"Features: {feature_names}")

    # Load models
    lgbm_bundle = load_bundle(Path(args.model))
    print(f"\nModel: {len(lgbm_bundle.get('feature_names', []))} features loaded")

    sr_model = SRPredictor.load(Path(args.sr_model))
    print("S/R predictor loaded")

    regime_model = None
    regime_path = Path(args.regime_model)
    if regime_path.exists():
        regime_model = GMMRegimeDetector.load(regime_path)
        print("Regime detector loaded")

    vol_model = None
    vol_path = Path(args.vol_model)
    if vol_path.exists():
        vol_model = GARCHVolatility.load(vol_path)
        print("GARCH volatility model loaded")

    # Run backtest
    results = run_full_backtest(
        df,
        lgbm_bundle=lgbm_bundle,
        sr_model=sr_model,
        regime_model=regime_model,
        vol_model=vol_model,
        use_grid=not args.no_grid,
        grid_max_levels=args.grid_levels,
        grid_sizing_decay=args.grid_decay,
        grid_atr_stop_mult=args.grid_atr_stop,
    )

    if not results:
        print("\nNo results — insufficient data")
        return

    # Print results
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)

    d = results["directional"]
    g = results["grid"]

    print(f"\n  Directional (Walk-Forward, {results['n_folds']} folds):")
    print(f"    Total Return:     {d.get('total_return_pct', 0):.2f}%")
    print(f"    Annualized Return:{d.get('annualized_return', 0):.2f}%")
    print(f"    Sharpe Ratio:     {d.get('sharpe', 0):.3f}")
    print(f"    Sortino Ratio:    {d.get('sortino', 0):.3f}")
    print(f"    Max Drawdown:     {d.get('max_drawdown_pct', 0):.2f}%")
    print(f"    Win Rate:         {d.get('win_rate', 0):.3f}")
    print(f"    Profit Factor:    {d.get('profit_factor', 0):.3f}")

    print(f"\n  Smart Grid (S/R-based recovery):")
    print(f"    Total PnL:        {g['total_pnl']:.2f}")
    print(f"    Grids Opened:     {g['n_grids']}")
    print(f"    Closed TP:        {g['n_tp']}")
    print(f"    Closed SL:        {g['n_sl']}")
    print(f"    Avg PnL/Grid:     {g['avg_pnl_per_grid']:.2f}")

    print(f"\n  Combined PnL:       {results['combined_pnl']:.2f}")
    print("=" * 60)

    # Save
    output = Path(args.output)
    output.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()

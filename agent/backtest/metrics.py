"""Backtest metrics: Sharpe, drawdown, bootstrap gates.

Concepts: Ch 05 (strategy evaluation), Ch 08 (multiple testing).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_metrics(returns: np.ndarray) -> dict[str, float]:
    """Compute standard backtest metrics from a returns array."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]

    if len(r) == 0:
        return {
            "total_return_pct": 0.0,
            "annualized_return": 0.0,
            "annualized_vol": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown_pct": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
            "num_trades": 0,
            "profit_factor": 0.0,
        }

    cum = np.cumprod(1 + r)
    total_return = float(cum[-1] / cum[0] - 1)

    ann_ret = total_return * (252 / max(len(r), 1))
    ann_vol = float(np.std(r) * np.sqrt(252))

    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    downside = r[r < 0]
    downside_vol = float(np.std(downside) * np.sqrt(252)) if len(downside) > 0 else ann_vol
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(cum)
    drawdown = (cum - peak) / peak
    max_dd = float(np.min(drawdown))

    # Win rate
    wins = float(np.sum(r > 0))
    total = float(len(r))
    win_rate = wins / total if total > 0 else 0.0

    # Profit factor
    gross_profit = float(np.sum(r[r > 0]))
    gross_loss = abs(float(np.sum(r[r < 0])))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "total_return_pct": round(total_return * 100, 2),
        "annualized_return": round(ann_ret * 100, 2),
        "annualized_vol": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "win_rate": round(win_rate, 3),
        "num_trades": int(total),
        "profit_factor": round(profit_factor, 3),
    }


def block_bootstrap_mean(
    returns: np.ndarray,
    *,
    block_len: int = 5,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Circular block bootstrap for mean daily return."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)

    if n < block_len or n < 10:
        return {
            "n_days": int(n),
            "mean_daily": float(np.nan),
            "ci_low_pct5": float(np.nan),
            "ci_high_pct95": float(np.nan),
            "p_mean_le_zero": float("nan"),
            "ok_sample": False,
        }

    rng = np.random.default_rng(seed)
    obs_mean = float(np.mean(r))

    def one_sample() -> float:
        n_blocks = int(np.ceil(n / block_len))
        draws: list[np.ndarray] = []
        for _ in range(n_blocks):
            start = int(rng.integers(0, n))
            idx = (start + np.arange(block_len)) % n
            draws.append(r[idx])
        return float(np.mean(np.concatenate(draws)[:n]))

    boot = np.array([one_sample() for _ in range(n_boot)], dtype=np.float64)
    p_le0 = float((np.sum(boot <= 0) + 1.0) / (n_boot + 1.0))

    return {
        "n_days": int(n),
        "mean_daily": obs_mean,
        "ci_low_pct5": float(np.percentile(boot, 5)),
        "ci_high_pct95": float(np.percentile(boot, 95)),
        "p_mean_le_zero": p_le0,
        "ok_sample": True,
    }


def holdout_pnl_gate(
    returns: np.ndarray,
    *,
    alpha: float = 0.05,
    min_trading_days: int = 30,
) -> dict[str, Any]:
    """Gate: holdout shows positive mean return with bootstrap support."""
    stats = block_bootstrap_mean(returns)
    passed = (
        stats["ok_sample"]
        and stats["n_days"] >= min_trading_days
        and stats["mean_daily"] > 0
        and stats["ci_low_pct5"] > 0
        and stats["p_mean_le_zero"] < alpha
    )
    return {"passed": passed, **stats}

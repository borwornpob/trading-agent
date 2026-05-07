"""Smart grid simulator for backtesting.

Concepts: Ch 08 (backtesting), Ch 23 (risk management).
Simulates the smart grid behavior within a backtest context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agent.strategy.adaptive_executor import route_execution
from agent.strategy.gold_strategy import TradeSignal


@dataclass
class GridSimulationResult:
    total_pnl: float
    max_exposure: float
    n_levels_filled: int
    n_grids_opened: int
    n_grids_closed_tp: int
    n_grids_closed_sl: int
    equity_curve: list[float]
    trades: list[dict[str, Any]]


def simulate_grid(
    prices: np.ndarray,
    direction: str,
    entry_index: int,
    *,
    predicted_high: float,
    predicted_low: float,
    base_size: float = 1.0,
    max_levels: int = 3,
    sizing_decay: float = 0.7,
    commission_rate: float = 0.0002,
    atr: float | None = None,
) -> GridSimulationResult:
    """Simulate a single grid cycle from entry to close.

    Args:
        prices: array of close prices
        direction: "long" or "short"
        entry_index: bar index where primary signal fires
        predicted_high/low: S/R predicted daily range
        base_size: position size multiplier
        max_levels: max grid levels including primary, matching live execution
        sizing_decay: anti-martingale decay
    """
    if entry_index >= len(prices) - 1:
        return GridSimulationResult(0.0, 0.0, 0, 0, 0, 0, [], [])

    entry_price = prices[entry_index]
    is_long = direction == "long"

    signal = TradeSignal(
        direction=direction,
        pred_class=3 if is_long else 1,
        score=1.0 if is_long else -1.0,
        p_up=0.8 if is_long else 0.2,
        p_down=0.2 if is_long else 0.8,
        regime="ranging",
        sentiment_score=0.0,
        event_risk=False,
        conviction="high",
    )
    plan = route_execution(
        signal,
        current_price=float(entry_price),
        predicted_high=predicted_high,
        predicted_low=predicted_low,
        bounce_probability=1.0,
        grid_enabled=True,
        max_grid_levels=max_levels,
        grid_sizing_decay=sizing_decay,
        base_size=base_size,
        atr=atr,
    )

    levels = [
        {
            "price": float(level["price"]),
            "size": float(level["size"]),
            "filled": not level.get("is_limit", False),
        }
        for level in plan.grid_levels
    ]
    hard_stop = plan.stop_loss_price or float(entry_price)
    take_profit = plan.take_profit_price or float(entry_price)

    total_pnl = 0.0
    max_exposure = sum(l["size"] for l in levels if l["filled"])
    n_filled = 1
    equity_curve = []
    trades = []
    avg_entry = entry_price
    total_size = base_size

    for bar_idx in range(entry_index + 1, len(prices)):
        price = prices[bar_idx]

        # Fill pending grid levels
        for lvl in levels:
            if lvl["filled"]:
                continue
            if is_long and price <= lvl["price"]:
                lvl["filled"] = True
                n_filled += 1
                # Update average entry
                total_size += lvl["size"]
                avg_entry = (avg_entry * (total_size - lvl["size"]) + lvl["price"] * lvl["size"]) / total_size
                cost = lvl["size"] * price * commission_rate
                total_pnl -= cost
                max_exposure = max(max_exposure, sum(l["size"] for l in levels if l["filled"]))
            elif not is_long and price >= lvl["price"]:
                lvl["filled"] = True
                n_filled += 1
                total_size += lvl["size"]
                avg_entry = (avg_entry * (total_size - lvl["size"]) + lvl["price"] * lvl["size"]) / total_size
                cost = lvl["size"] * price * commission_rate
                total_pnl -= cost
                max_exposure = max(max_exposure, sum(l["size"] for l in levels if l["filled"]))

        # Check hard stop
        if is_long and price <= hard_stop:
            total_pnl += total_size * (price - avg_entry)
            trades.append({"bar": bar_idx, "action": "hard_stop", "price": price})
            return GridSimulationResult(
                total_pnl=total_pnl, max_exposure=max_exposure,
                n_levels_filled=n_filled, n_grids_opened=1,
                n_grids_closed_sl=1, n_grids_closed_tp=0,
                equity_curve=equity_curve, trades=trades,
            )
        elif not is_long and price >= hard_stop:
            total_pnl += total_size * (avg_entry - price)
            trades.append({"bar": bar_idx, "action": "hard_stop", "price": price})
            return GridSimulationResult(
                total_pnl=total_pnl, max_exposure=max_exposure,
                n_levels_filled=n_filled, n_grids_opened=1,
                n_grids_closed_sl=1, n_grids_closed_tp=0,
                equity_curve=equity_curve, trades=trades,
            )

        # Check take profit
        if is_long and price >= take_profit:
            pnl = total_size * (price - avg_entry) - total_size * price * commission_rate
            total_pnl += pnl
            trades.append({"bar": bar_idx, "action": "take_profit", "price": price, "pnl": pnl})
            return GridSimulationResult(
                total_pnl=total_pnl, max_exposure=max_exposure,
                n_levels_filled=n_filled, n_grids_opened=1,
                n_grids_closed_tp=1, n_grids_closed_sl=0,
                equity_curve=equity_curve, trades=trades,
            )
        elif not is_long and price <= take_profit:
            pnl = total_size * (avg_entry - price) - total_size * price * commission_rate
            total_pnl += pnl
            trades.append({"bar": bar_idx, "action": "take_profit", "price": price, "pnl": pnl})
            return GridSimulationResult(
                total_pnl=total_pnl, max_exposure=max_exposure,
                n_levels_filled=n_filled, n_grids_opened=1,
                n_grids_closed_tp=1, n_grids_closed_sl=0,
                equity_curve=equity_curve, trades=trades,
            )

        equity_curve.append(total_pnl + total_size * (price - avg_entry if is_long else avg_entry - price))

    # End of data — close at last price
    last_price = prices[-1]
    if is_long:
        total_pnl += total_size * (last_price - avg_entry)
    else:
        total_pnl += total_size * (avg_entry - last_price)

    return GridSimulationResult(
        total_pnl=total_pnl, max_exposure=max_exposure,
        n_levels_filled=n_filled, n_grids_opened=1,
        n_grids_closed_tp=0, n_grids_closed_sl=0,
        equity_curve=equity_curve, trades=trades,
    )

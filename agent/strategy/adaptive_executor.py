"""Regime-adaptive execution router.

Concepts: Ch 13 (regime detection), Ch 23 (operational risk).
Routes to trend_executor, smart_grid, or flat based on regime + confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.strategy.gold_strategy import TradeSignal


@dataclass(frozen=True)
class ExecutionPlan:
    mode: str  # "trend", "range_grid", "flat"
    direction: str  # "long", "short", "flat"
    size_multiplier: float  # 0.0 to 1.0+
    stop_loss_price: float | None
    take_profit_price: float | None
    grid_levels: list[dict[str, Any]]  # For range_grid mode: [{price, size, is_limit}]
    notes: list[str]


def route_execution(
    signal: TradeSignal,
    *,
    current_price: float,
    predicted_high: float | None = None,
    predicted_low: float | None = None,
    bounce_probability: float = 0.5,
    vol_trailing_stop_distance: float | None = None,
    grid_enabled: bool = True,
    max_grid_levels: int = 3,
    grid_sizing_decay: float = 0.7,
    base_size: float = 1.0,
    atr: float | None = None,
) -> ExecutionPlan:
    """Route to the appropriate execution mode based on regime and conditions."""
    notes: list[str] = []

    # Flat mode checks
    if signal.direction == "flat":
        return ExecutionPlan(
            mode="flat",
            direction="flat",
            size_multiplier=0.0,
            stop_loss_price=None,
            take_profit_price=None,
            grid_levels=[],
            notes=["signal_direction_is_flat"],
        )

    if signal.event_risk:
        return ExecutionPlan(
            mode="flat",
            direction="flat",
            size_multiplier=0.0,
            stop_loss_price=None,
            take_profit_price=None,
            grid_levels=[],
            notes=["event_risk_detected_no_new_trades"],
        )

    is_ranging = signal.regime == "ranging"
    is_trending = signal.regime in ("trending_up", "trending_down")
    is_volatile = signal.regime == "volatile"

    # Volatile regime → flat or reduced size
    if is_volatile:
        notes.append("volatile_regime_reduced_size")
        return _trend_execution(
            signal,
            current_price=current_price,
            vol_trailing_stop_distance=vol_trailing_stop_distance,
            predicted_high=predicted_high,
            predicted_low=predicted_low,
            base_size=base_size * 0.5,
            atr=atr,
            notes=notes,
        )

    # Ranging regime → smart grid
    if is_ranging and grid_enabled and signal.direction != "flat":
        if bounce_probability > 0.55 and predicted_high and predicted_low:
            return _range_grid_execution(
                signal,
                current_price=current_price,
                predicted_high=predicted_high,
                predicted_low=predicted_low,
                bounce_probability=bounce_probability,
                max_levels=max_grid_levels,
                sizing_decay=grid_sizing_decay,
                base_size=base_size,
                atr=atr,
                notes=notes,
            )
        notes.append("ranging_but_low_bounce_prob_using_trend_mode")

    # Trending regime → trend execution
    return _trend_execution(
        signal,
        current_price=current_price,
        vol_trailing_stop_distance=vol_trailing_stop_distance,
        predicted_high=predicted_high,
        predicted_low=predicted_low,
        base_size=base_size,
        atr=atr,
        notes=notes,
    )


def _trend_execution(
    signal: TradeSignal,
    *,
    current_price: float,
    vol_trailing_stop_distance: float | None,
    predicted_high: float | None = None,
    predicted_low: float | None = None,
    base_size: float,
    atr: float | None,
    notes: list[str],
) -> ExecutionPlan:
    """Trend mode: single entry with trailing stop."""
    size = base_size
    if signal.conviction == "high":
        size *= 1.0
    elif signal.conviction == "medium":
        size *= 0.7
    else:
        size *= 0.5

    sl_distance = vol_trailing_stop_distance or (atr or current_price * 0.01) * 2.0

    if signal.direction == "long":
        stop_loss = current_price - sl_distance
        take_profit = current_price + sl_distance * 1.5
        if predicted_high is not None and take_profit > predicted_high:
            take_profit = predicted_high
            notes.append("tp_capped_by_sr")
    else:
        stop_loss = current_price + sl_distance
        take_profit = current_price - sl_distance * 1.5
        if predicted_low is not None and take_profit < predicted_low:
            take_profit = predicted_low
            notes.append("tp_capped_by_sr")

    return ExecutionPlan(
        mode="trend",
        direction=signal.direction,
        size_multiplier=size,
        stop_loss_price=round(stop_loss, 2),
        take_profit_price=round(take_profit, 2),
        grid_levels=[],
        notes=notes + [f"trend_mode_conviction_{signal.conviction}"],
    )


def _range_grid_execution(
    signal: TradeSignal,
    *,
    current_price: float,
    predicted_high: float,
    predicted_low: float,
    bounce_probability: float,
    max_levels: int,
    sizing_decay: float,
    base_size: float,
    atr: float | None,
    notes: list[str],
) -> ExecutionPlan:
    """Range mode: smart grid with S/R-based recovery levels."""
    grid_levels: list[dict[str, Any]] = []
    is_long = signal.direction == "long"

    # Primary entry
    grid_levels.append(
        {
            "price": round(current_price, 2),
            "size": base_size,
            "is_limit": False,
            "label": "primary",
        }
    )

    # Recovery levels at predicted S/R
    if is_long:
        # For longs, recovery levels are below entry at support levels
        range_step = (current_price - predicted_low) / max(max_levels, 1)
        for i in range(1, max_levels + 1):
            level_price = current_price - range_step * i
            if level_price <= predicted_low:
                break
            grid_levels.append(
                {
                    "price": round(level_price, 2),
                    "size": base_size * (sizing_decay**i),
                    "is_limit": True,
                    "label": f"grid_support_{i}",
                }
            )
        hard_stop = predicted_low - (atr or current_price * 0.005)
        take_profit = predicted_high
    else:
        # For shorts, recovery levels are above entry at resistance levels
        range_step = (predicted_high - current_price) / max(max_levels, 1)
        for i in range(1, max_levels + 1):
            level_price = current_price + range_step * i
            if level_price >= predicted_high:
                break
            grid_levels.append(
                {
                    "price": round(level_price, 2),
                    "size": base_size * (sizing_decay**i),
                    "is_limit": True,
                    "label": f"grid_resistance_{i}",
                }
            )
        hard_stop = predicted_high + (atr or current_price * 0.005)
        take_profit = predicted_low

    total_size = sum(g["size"] for g in grid_levels)

    return ExecutionPlan(
        mode="range_grid",
        direction=signal.direction,
        size_multiplier=base_size,
        stop_loss_price=round(hard_stop, 2),
        take_profit_price=round(take_profit, 2),
        grid_levels=grid_levels,
        notes=notes
        + [
            f"range_grid_bounce_{bounce_probability:.2f}",
            f"grid_levels_{len(grid_levels)}",
            f"total_grid_size_{total_size:.2f}",
        ],
    )

"""Parity checks for live and backtest smart-grid planning."""

from __future__ import annotations

import numpy as np

from agent.autonomous_loop import _grid_plan_to_order_intents
from agent.backtest.grid_simulator import simulate_grid
from agent.strategy.adaptive_executor import route_execution
from agent.strategy.gold_strategy import TradeSignal


def _ranging_signal(direction: str = "long") -> TradeSignal:
    return TradeSignal(
        direction=direction,
        pred_class=3 if direction == "long" else 1,
        score=0.75 if direction == "long" else -0.75,
        p_up=0.8 if direction == "long" else 0.2,
        p_down=0.2 if direction == "long" else 0.8,
        regime="ranging",
        sentiment_score=0.0,
        event_risk=False,
        conviction="high",
    )


def test_live_grid_plan_becomes_primary_and_recovery_orders() -> None:
    plan = route_execution(
        _ranging_signal("long"),
        current_price=2050.0,
        predicted_high=2075.0,
        predicted_low=2020.0,
        bounce_probability=0.8,
        max_grid_levels=3,
        grid_sizing_decay=0.7,
        base_size=0.01,
        atr=8.0,
    )

    intents = _grid_plan_to_order_intents(plan, symbol="XAUUSD", timestamp_ns=1_000)

    assert plan.mode == "range_grid"
    assert len(plan.grid_levels) == 3
    assert len(intents) == len(plan.grid_levels)
    assert intents[0].is_market()
    assert intents[0].volume == 0.01
    assert [intent.price for intent in intents[1:]] == [2040.0, 2030.0]
    assert [round(intent.volume, 5) for intent in intents] == [0.01, 0.007, 0.0049]


def test_backtest_grid_simulator_uses_live_sr_levels() -> None:
    prices = np.array([2050.0, 2040.0, 2030.0, 2075.0])
    sim = simulate_grid(
        prices,
        "long",
        0,
        predicted_high=2075.0,
        predicted_low=2020.0,
        base_size=0.01,
        max_levels=3,
        sizing_decay=0.7,
        commission_rate=0.0,
        atr=8.0,
    )

    assert sim.n_levels_filled == 3
    assert round(sim.max_exposure, 5) == 0.0219
    assert sim.n_grids_closed_tp == 1

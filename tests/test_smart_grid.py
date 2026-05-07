"""Tests for smart grid."""

import pytest
from agent.strategy.smart_grid import SmartGrid, GridState, GridPosition


def test_create_grid():
    grid = SmartGrid(max_levels=3, sizing_decay=0.7)
    levels = [
        {"price": 2050.0, "size": 1.0, "is_limit": False, "label": "primary"},
        {"price": 2045.0, "size": 0.7, "is_limit": True, "label": "grid_1"},
        {"price": 2040.0, "size": 0.49, "is_limit": True, "label": "grid_2"},
    ]
    state = grid.create_grid("g1", "long", levels, hard_stop=2035.0, take_profit=2060.0)
    assert state.direction == "long"
    assert len(state.positions) == 3
    assert state.positions[0].is_filled
    assert not state.positions[1].is_filled


def test_should_fill_next_level():
    grid = SmartGrid(max_levels=3, sizing_decay=0.7)
    levels = [
        {"price": 2050.0, "size": 1.0, "is_limit": False},
        {"price": 2045.0, "size": 0.7, "is_limit": True},
        {"price": 2040.0, "size": 0.49, "is_limit": True},
    ]
    grid.create_grid("g1", "long", levels, hard_stop=2035.0, take_profit=2060.0)

    # Price hasn't dropped to level 1
    assert grid.should_fill_next_level("g1", 2046.0) is None

    # Price drops to level 1
    pos = grid.should_fill_next_level("g1", 2044.0)
    assert pos is not None
    assert pos.price == 2045.0


def test_check_grid_close_take_profit():
    grid = SmartGrid(max_levels=3)
    levels = [{"price": 2050.0, "size": 1.0, "is_limit": False}]
    grid.create_grid("g1", "long", levels, hard_stop=2035.0, take_profit=2060.0)

    result = grid.check_grid_close("g1", 2061.0)
    assert result["action"] == "close_all"
    assert result["reason"] == "take_profit_hit"


def test_check_grid_close_hard_stop():
    grid = SmartGrid(max_levels=3)
    levels = [{"price": 2050.0, "size": 1.0, "is_limit": False}]
    grid.create_grid("g1", "long", levels, hard_stop=2035.0, take_profit=2060.0)

    result = grid.check_grid_close("g1", 2034.0)
    assert result["action"] == "close_all"
    assert result["reason"] == "hard_stop_hit"

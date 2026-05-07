"""Order history and position endpoints."""

from __future__ import annotations

from typing import Any

from agent.live_state import read_live_state
from fastapi import APIRouter

router = APIRouter(tags=["orders"])

_order_history: list[dict[str, Any]] = []


@router.get("/orders")
async def get_orders() -> dict[str, Any]:
    """Order history from recent cycles."""
    state = read_live_state()
    orders = state.get("orders") or _order_history
    return {"orders": orders[-50:]}


@router.get("/grid")
async def get_grid() -> dict[str, Any]:
    """Active grid positions + exposure."""
    state = read_live_state()
    positions = state.get("positions", [])
    total_exposure = sum(float(p.get("volume", 0.0)) for p in positions)
    return {"grids": positions, "total_exposure": total_exposure}

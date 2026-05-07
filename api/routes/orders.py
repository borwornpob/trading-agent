"""Order history and position endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

router = APIRouter(tags=["orders"])

_order_history: list[dict[str, Any]] = []


@router.get("/orders")
async def get_orders() -> dict[str, Any]:
    """Order history from recent cycles."""
    return {"orders": _order_history[-50:]}


@router.get("/grid")
async def get_grid() -> dict[str, Any]:
    """Active grid positions + exposure."""
    return {"grids": [], "total_exposure": 0.0}

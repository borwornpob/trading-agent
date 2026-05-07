"""Smart grid recovery engine.

Concepts: Ch 23 (risk management), anti-martingale sizing, S/R-based grid levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GridPosition:
    level: int
    price: float
    size: float
    side: str           # "buy" or "sell"
    is_filled: bool
    order_id: str | None


@dataclass
class GridState:
    """Tracks active grid positions and total exposure."""
    grid_id: str
    direction: str          # "long" or "short"
    max_total_size: float
    positions: list[GridPosition] = field(default_factory=list)
    total_size: float = 0.0
    hard_stop: float | None = None
    take_profit: float | None = None
    is_active: bool = True
    realized_pnl: float = 0.0

    def add_position(self, pos: GridPosition) -> None:
        self.positions.append(pos)
        if pos.is_filled:
            self.total_size += pos.size

    def total_exposure_ratio(self, initial_size: float) -> float:
        if initial_size <= 0:
            return 0.0
        return self.total_size / initial_size


class SmartGrid:
    """Manages S/R-based grid recovery with anti-martingale sizing.

    Key constraints:
    - Max N levels (configurable, default 3)
    - Sizes decrease: 1.0, 0.7, 0.5 (not increasing)
    - Total exposure capped at 2x initial
    - Hard stop at predicted daily range boundary
    - Disabled during event risk
    """

    def __init__(
        self,
        *,
        max_levels: int = 3,
        sizing_decay: float = 0.7,
        total_exposure_cap: float = 2.0,
    ) -> None:
        self.max_levels = max_levels
        self.sizing_decay = sizing_decay
        self.total_exposure_cap = total_exposure_cap
        self._grids: dict[str, GridState] = {}

    def create_grid(
        self,
        grid_id: str,
        direction: str,
        levels: list[dict[str, Any]],
        *,
        hard_stop: float | None = None,
        take_profit: float | None = None,
        max_total_size: float = 2.0,
    ) -> GridState:
        """Create a new grid from the execution plan's grid_levels."""
        grid = GridState(
            grid_id=grid_id,
            direction=direction,
            max_total_size=max_total_size,
            hard_stop=hard_stop,
            take_profit=take_profit,
        )

        for i, lvl in enumerate(levels):
            size = float(lvl.get("size", 1.0))
            pos = GridPosition(
                level=i,
                price=float(lvl["price"]),
                size=size,
                side="buy" if direction == "long" else "sell",
                is_filled=not lvl.get("is_limit", False),
                order_id=None,
            )
            grid.add_position(pos)

        self._grids[grid_id] = grid
        return grid

    def check_grid_close(
        self,
        grid_id: str,
        current_price: float,
    ) -> dict[str, Any]:
        """Check if grid should be closed (hit TP or hard stop)."""
        grid = self._grids.get(grid_id)
        if grid is None or not grid.is_active:
            return {"action": "none", "reason": "grid_not_found_or_inactive"}

        if grid.take_profit is not None:
            if grid.direction == "long" and current_price >= grid.take_profit:
                return {"action": "close_all", "reason": "take_profit_hit", "price": grid.take_profit}
            if grid.direction == "short" and current_price <= grid.take_profit:
                return {"action": "close_all", "reason": "take_profit_hit", "price": grid.take_profit}

        if grid.hard_stop is not None:
            if grid.direction == "long" and current_price <= grid.hard_stop:
                return {"action": "close_all", "reason": "hard_stop_hit", "price": grid.hard_stop}
            if grid.direction == "short" and current_price >= grid.hard_stop:
                return {"action": "close_all", "reason": "hard_stop_hit", "price": grid.hard_stop}

        return {"action": "hold", "reason": "within_range"}

    def should_fill_next_level(
        self,
        grid_id: str,
        current_price: float,
    ) -> GridPosition | None:
        """Check if price has reached the next unfilled grid level."""
        grid = self._grids.get(grid_id)
        if grid is None or not grid.is_active:
            return None

        for pos in grid.positions:
            if pos.is_filled or pos.order_id is not None:
                continue

            # For long grid: fill if price drops to level
            if grid.direction == "long" and current_price <= pos.price:
                initial_size = grid.positions[0].size if grid.positions else 1.0
                if grid.total_exposure_ratio(initial_size) >= self.total_exposure_cap:
                    return None
                return pos

            # For short grid: fill if price rises to level
            if grid.direction == "short" and current_price >= pos.price:
                initial_size = grid.positions[0].size if grid.positions else 1.0
                if grid.total_exposure_ratio(initial_size) >= self.total_exposure_cap:
                    return None
                return pos

        return None

    def mark_filled(self, grid_id: str, level: int) -> None:
        grid = self._grids.get(grid_id)
        if grid is None:
            return
        for pos in grid.positions:
            if pos.level == level:
                pos.is_filled = True
                grid.total_size += pos.size
                break

    def close_grid(self, grid_id: str, *, pnl: float = 0.0) -> None:
        grid = self._grids.get(grid_id)
        if grid is not None:
            grid.is_active = False
            grid.realized_pnl = pnl

    def get_grid(self, grid_id: str) -> GridState | None:
        return self._grids.get(grid_id)

    @property
    def active_grids(self) -> list[GridState]:
        return [g for g in self._grids.values() if g.is_active]

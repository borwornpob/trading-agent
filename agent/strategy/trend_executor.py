"""Trend mode executor: single entry with trailing stop.

Concepts: Ch 05 (trailing stops, risk management), Ch 09 (GARCH for stop distance).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrendPosition:
    entry_price: float
    direction: str          # "long" or "short"
    size: float
    initial_stop: float
    trailing_stop: float
    highest_profit_price: float  # Track for trailing


class TrendExecutor:
    """Manages trend-mode positions with trailing stops based on volatility."""

    def __init__(self, *, trail_atr_multiplier: float = 2.0) -> None:
        self.trail_atr_multiplier = trail_atr_multiplier
        self._position: TrendPosition | None = None

    def open(
        self,
        direction: str,
        entry_price: float,
        size: float,
        *,
        stop_distance: float,
    ) -> TrendPosition:
        if direction == "long":
            stop = entry_price - stop_distance
        else:
            stop = entry_price + stop_distance

        self._position = TrendPosition(
            entry_price=entry_price,
            direction=direction,
            size=size,
            initial_stop=round(stop, 2),
            trailing_stop=round(stop, 2),
            highest_profit_price=entry_price,
        )
        return self._position

    def update_trail(self, current_price: float, *, atr: float | None = None) -> dict:
        """Update trailing stop. Returns action dict."""
        if self._position is None:
            return {"action": "none", "reason": "no_position"}

        pos = self._position
        trail_dist = atr * self.trail_atr_multiplier if atr else abs(pos.entry_price - pos.trailing_stop)

        if pos.direction == "long":
            new_high = max(pos.highest_profit_price, current_price)
            new_trail = new_high - trail_dist
            new_trail = max(new_trail, pos.trailing_stop)  # Trail only moves up

            if current_price <= pos.trailing_stop:
                return {"action": "close", "reason": "trailing_stop_hit", "price": pos.trailing_stop}

            self._position = TrendPosition(
                entry_price=pos.entry_price,
                direction=pos.direction,
                size=pos.size,
                initial_stop=pos.initial_stop,
                trailing_stop=round(new_trail, 2),
                highest_profit_price=new_high,
            )
        else:
            new_low = min(pos.highest_profit_price, current_price)
            new_trail = new_low + trail_dist
            new_trail = min(new_trail, pos.trailing_stop)  # Trail only moves down

            if current_price >= pos.trailing_stop:
                return {"action": "close", "reason": "trailing_stop_hit", "price": pos.trailing_stop}

            self._position = TrendPosition(
                entry_price=pos.entry_price,
                direction=pos.direction,
                size=pos.size,
                initial_stop=pos.initial_stop,
                trailing_stop=round(new_trail, 2),
                highest_profit_price=new_low,
            )

        return {"action": "hold", "trailing_stop": self._position.trailing_stop}

    def close(self) -> TrendPosition | None:
        pos = self._position
        self._position = None
        return pos

    @property
    def position(self) -> TrendPosition | None:
        return self._position

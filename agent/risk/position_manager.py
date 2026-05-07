"""Position manager: target positions -> order intents, reconciliation.

Concepts: Ch 00 bridge (execution layer), Ch 23 (operational risk).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: str          # "buy" or "sell"
    volume: float
    price: float | None = None  # None = market order
    stop_loss: float | None = None
    take_profit: float | None = None
    label: str = ""
    client_order_id: str = ""

    def is_market(self) -> bool:
        return self.price is None


@dataclass(frozen=True)
class PositionSnapshot:
    symbol: str
    side: str          # "buy" or "sell"
    volume: float
    entry_price: float
    unrealized_pnl: float = 0.0


def make_client_order_id(symbol: str, side: str, price: float | None, timestamp_ns: int) -> str:
    """Deterministic SHA-256 idempotency key."""
    raw = f"{symbol}|{side}|{price}|{timestamp_ns}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def compute_order_intents(
    target_direction: str,
    target_size: float,
    current_position: PositionSnapshot | None,
    *,
    symbol: str = "XAUUSD",
    price: float | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    timestamp_ns: int = 0,
) -> list[OrderIntent]:
    """Compute the diff between target and current position → list of order intents."""
    current_volume = 0.0
    if current_position is not None:
        if current_position.side == "buy":
            current_volume = current_position.volume
        else:
            current_volume = -current_position.volume

    target_volume = 0.0
    if target_direction == "long":
        target_volume = target_size
    elif target_direction == "short":
        target_volume = -target_size

    delta = target_volume - current_volume
    if abs(delta) < 0.01:
        return []

    side = "buy" if delta > 0 else "sell"
    return [
        OrderIntent(
            symbol=symbol,
            side=side,
            volume=abs(delta),
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            label="position_adjust",
            client_order_id=make_client_order_id(symbol, side, price, timestamp_ns),
        )
    ]


def compute_close_intent(
    current_position: PositionSnapshot,
    *,
    symbol: str = "XAUUSD",
    timestamp_ns: int = 0,
) -> OrderIntent | None:
    """Generate a close order for the current position."""
    if current_position is None:
        return None
    close_side = "sell" if current_position.side == "buy" else "buy"
    return OrderIntent(
        symbol=symbol,
        side=close_side,
        volume=current_position.volume,
        price=None,  # Market order for close
        label="close_position",
        client_order_id=make_client_order_id(symbol, close_side, None, timestamp_ns),
    )

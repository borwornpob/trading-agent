"""MetaTrader 5 execution gateway using the mt5_api socket bridge.

The bridge runs as a Python TCP server. The MT5 Expert Advisor connects to it,
then this wrapper exposes synchronous methods for the existing live loop.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import sys
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from agent.config import MT5Settings


@dataclass
class PositionInfo:
    position_id: int
    symbol: str
    volume: float
    side: str
    entry_price: float
    pnl: float = 0.0


class MT5Gateway:
    """Synchronous adapter over the async mt5_api bridge."""

    def __init__(self, settings: MT5Settings) -> None:
        self.settings = settings
        self._bridge: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def is_configured(self) -> bool:
        return True

    def connect(self) -> Any:
        """Start the TCP server and wait for the MT5 EA to connect."""
        if self._bridge is not None:
            return self._bridge

        self._ensure_import_path()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="mt5-gateway-loop", daemon=True)
        self._thread.start()
        return self._call(self._connect_async(), timeout=self.settings.connection_timeout_seconds + 10)

    def disconnect(self) -> None:
        if self._loop is None:
            return
        try:
            if self._bridge is not None:
                try:
                    self._call(self._bridge.stop(), timeout=self.settings.command_timeout_seconds)
                except Exception:
                    pass
        finally:
            self._bridge = None
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=5)
            self._thread = None
            self._loop = None

    def reconcile(self) -> list[PositionInfo]:
        self._require_bridge()
        positions = self._call(self._bridge.get_positions(), timeout=self.settings.command_timeout_seconds)
        return [
            PositionInfo(
                position_id=int(getattr(position, "ticket", 0)),
                symbol=str(getattr(position, "symbol", "")),
                volume=float(getattr(position, "volume", 0.0)),
                side=self._position_side(str(getattr(position, "order_type", ""))),
                entry_price=float(getattr(position, "open_price", 0.0)),
                pnl=float(getattr(position, "profit", 0.0)),
            )
            for position in positions
        ]

    def submit_market_order(
        self,
        *,
        symbol: str,
        side_buy: bool,
        volume: float,
        label: str = "agent",
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Any:
        self._require_bridge()
        if side_buy:
            return self._call(
                self._bridge.buy(
                    symbol,
                    volume,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    magic=self.settings.magic,
                    comment=label,
                ),
                timeout=self.settings.command_timeout_seconds,
            )
        return self._call(
            self._bridge.sell(
                symbol,
                volume,
                stop_loss=stop_loss,
                take_profit=take_profit,
                magic=self.settings.magic,
                comment=label,
            ),
            timeout=self.settings.command_timeout_seconds,
        )

    def submit_limit_order(
        self,
        *,
        symbol: str,
        side_buy: bool,
        volume: float,
        price: float,
        label: str = "agent_limit",
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Any:
        self._require_bridge()
        from mt5_bridge import OrderType

        order_type = OrderType.BUY_LIMIT if side_buy else OrderType.SELL_LIMIT
        return self._call(
            self._bridge.place_pending_order(
                symbol,
                order_type,
                float(price),
                volume,
                stop_loss=stop_loss,
                take_profit=take_profit,
                magic=self.settings.magic,
                comment=label,
            ),
            timeout=self.settings.command_timeout_seconds,
        )

    def close_position(self, position_id: int) -> Any:
        self._require_bridge()
        return self._call(
            self._bridge.close_position(position_id),
            timeout=self.settings.command_timeout_seconds,
        )

    def get_account(self) -> Any:
        self._require_bridge()
        return self._call(
            self._bridge.get_account(),
            timeout=self.settings.command_timeout_seconds,
        )

    def heartbeat(self) -> bool:
        self._require_bridge()
        return bool(
            self._call(
                self._bridge.heartbeat(),
                timeout=self.settings.command_timeout_seconds,
            )
        )

    def fetch_ohlcv(self, *, symbol: str, frequency: str, count: int | None = None) -> pl.DataFrame:
        """Fetch OHLCV bars directly from the connected MT5 EA."""
        self._require_bridge()
        from mt5_bridge import Timeframe

        timeframe = self._timeframe(frequency, Timeframe)
        rates = self._call(
            self._bridge.get_rates(symbol, timeframe, count or self.settings.rate_count),
            timeout=self.settings.command_timeout_seconds,
        )
        if not rates:
            return _empty_ohlcv()

        rows = []
        for rate in rates:
            rows.append({
                "timestamp": _parse_mt5_time(rate.get("time")),
                "open": float(rate.get("open", 0.0)),
                "high": float(rate.get("high", 0.0)),
                "low": float(rate.get("low", 0.0)),
                "close": float(rate.get("close", 0.0)),
                "volume": float(rate.get("volume", 0.0)),
            })
        return (
            pl.DataFrame(rows)
            .with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))
            .sort("timestamp")
        )

    async def _connect_async(self) -> Any:
        from mt5_bridge import MT5Bridge

        self._bridge = MT5Bridge(host=self.settings.host, port=self.settings.port)
        await self._bridge.start()
        await self._bridge.wait_for_connection(timeout=self.settings.connection_timeout_seconds)
        return self._bridge

    def _run_loop(self) -> None:
        if self._loop is None:
            return
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _call(self, coro: Any, *, timeout: float) -> Any:
        if self._loop is None:
            raise RuntimeError("connect() first")
        future: Future[Any] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def _require_bridge(self) -> None:
        if self._bridge is None:
            raise RuntimeError("connect() first")

    def _ensure_import_path(self) -> None:
        if self.settings.api_path is None:
            return
        api_path = Path(self.settings.api_path).expanduser().resolve()
        if not api_path.exists():
            raise RuntimeError(f"MT5_API_PATH does not exist: {api_path}")
        api_path_str = str(api_path)
        if api_path_str not in sys.path:
            sys.path.insert(0, api_path_str)

    @staticmethod
    def _position_side(order_type: str) -> str:
        return "buy" if "BUY" in order_type.upper() else "sell"

    @staticmethod
    def _timeframe(frequency: str, timeframe_enum: Any) -> Any:
        freq = frequency.strip().lower()
        mapping = {
            "1m": "M1",
            "m1": "M1",
            "5m": "M5",
            "m5": "M5",
            "15m": "M15",
            "15min": "M15",
            "15minute": "M15",
            "m15": "M15",
            "30m": "M30",
            "30min": "M30",
            "30minute": "M30",
            "1h": "H1",
            "hourly": "H1",
            "h1": "H1",
            "4h": "H4",
            "4hour": "H4",
            "h4": "H4",
            "daily": "D1",
            "1d": "D1",
            "d1": "D1",
        }
        name = mapping.get(freq)
        if name is None:
            raise ValueError(f"Unsupported MT5 timeframe: {frequency}")
        return getattr(timeframe_enum, name)


def _empty_ohlcv() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "timestamp": pl.Datetime("ms"),
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
    })


def _parse_mt5_time(value: Any) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(float(value), tz=dt.UTC).replace(tzinfo=None)
    text = str(value or "").strip()
    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            pass
    try:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError as exc:
        raise ValueError(f"Unsupported MT5 rate time: {value!r}") from exc

"""MT5 gateway data conversion tests."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from agent.config import MT5Settings
from agent.mt5_gateway import MT5Gateway


def test_fetch_ohlcv_converts_mt5_rates(monkeypatch) -> None:
    fake_module = types.SimpleNamespace(
        Timeframe=types.SimpleNamespace(M15="M15"),
    )
    monkeypatch.setitem(sys.modules, "mt5_bridge", fake_module)

    gateway = MT5Gateway(
        MT5Settings(
            api_path=Path("."),
            host="0.0.0.0",
            port=1111,
            connection_timeout_seconds=1.0,
            command_timeout_seconds=1.0,
            rate_count=2,
            magic=1,
        )
    )
    gateway._bridge = types.SimpleNamespace(get_rates=lambda *args: object())
    monkeypatch.setattr(
        gateway,
        "_call",
        lambda _coro, timeout: [
            {"time": "2026.05.07 15:15:00", "open": 2, "high": 3, "low": 1, "close": 2.5, "volume": 20},
            {"time": "2026.05.07 15:00:00", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10},
        ],
    )

    df = gateway.fetch_ohlcv(symbol="XAUUSD", frequency="15m")

    assert df["close"].to_list() == [1.5, 2.5]
    assert df["volume"].to_list() == [10.0, 20.0]

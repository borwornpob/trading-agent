"""Risk guard: position limits, daily loss tracking, kill switch.

Concepts: Ch 05 (risk metrics), Ch 23 (kill switches, operational risk).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RiskConfig:
    shadow_mode: bool = True
    kill_switch: bool = False
    max_position_volume: float = 500.0
    max_order_volume: float = 200.0
    max_daily_loss_units: float = 10_000.0
    dedup_window_seconds: float = 60.0
    state_path: Path = field(default_factory=lambda: Path("artifacts/risk_state.json"))

    def __post_init__(self) -> None:
        if isinstance(self.state_path, str):
            self.state_path = Path(self.state_path)


@dataclass
class RiskState:
    day_utc: str | None = None
    realized_pnl: float = 0.0
    kill_switch_activated: bool = False
    last_order_ts: float = 0.0
    consecutive_losses: int = 0
    recovery_mode: bool = False

    @classmethod
    def load(cls, path: Path) -> RiskState:
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            day_utc=raw.get("day_utc"),
            realized_pnl=float(raw.get("realized_pnl", 0.0)),
            kill_switch_activated=bool(raw.get("kill_switch_activated", False)),
            last_order_ts=float(raw.get("last_order_ts", 0.0)),
            consecutive_losses=int(raw.get("consecutive_losses", 0)),
            recovery_mode=bool(raw.get("recovery_mode", False)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        payload = {
            "day_utc": self.day_utc,
            "realized_pnl": self.realized_pnl,
            "kill_switch_activated": self.kill_switch_activated,
            "last_order_ts": self.last_order_ts,
            "consecutive_losses": self.consecutive_losses,
            "recovery_mode": self.recovery_mode,
        }
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)


class RiskGuard:
    """Pre-trade risk checks with daily PnL tracking and kill switch."""

    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg
        self._state = RiskState.load(cfg.state_path)

    def refresh_day(self) -> None:
        import datetime as dt
        today = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
        if self._state.day_utc != today:
            self._state.day_utc = today
            self._state.realized_pnl = 0.0
            self._state.save(self.cfg.state_path)

    def allow_order(self, *, volume: float, signed_direction: int) -> tuple[bool, str]:
        if self.cfg.kill_switch or self._state.kill_switch_activated:
            return False, "kill_switch"
        if self.cfg.shadow_mode:
            return True, "shadow_ok"
        now = time.monotonic()
        if now - self._state.last_order_ts < self.cfg.dedup_window_seconds:
            return False, "dedup"
        if abs(volume) > self.cfg.max_order_volume:
            return False, "max_order_volume"
        if abs(volume) > self.cfg.max_position_volume:
            return False, "max_position_volume"
        return True, "ok"

    def record_order_sent(self) -> None:
        self._state.last_order_ts = time.monotonic()
        self._state.save(self.cfg.state_path)

    def record_pnl(self, delta: float) -> None:
        self.refresh_day()
        self._state.realized_pnl += delta
        if delta < 0:
            self._state.consecutive_losses += 1
            if self._state.consecutive_losses >= 3:
                self._state.recovery_mode = True
        else:
            self._state.consecutive_losses = 0
            self._state.recovery_mode = False
        if self._state.realized_pnl <= -abs(self.cfg.max_daily_loss_units):
            self.activate_kill_switch("max_daily_loss")
        self._state.save(self.cfg.state_path)

    def activate_kill_switch(self, reason: str) -> None:
        self._state.kill_switch_activated = True
        self._state.save(self.cfg.state_path)

    def reset_kill_switch(self) -> None:
        self._state.kill_switch_activated = False
        self._state.save(self.cfg.state_path)

    def get_sizing_multiplier(self) -> float:
        """Return position sizing multiplier (reduced in recovery mode)."""
        if self._state.recovery_mode:
            return 0.5
        return 1.0

    def state_dict(self) -> dict[str, Any]:
        return {
            "shadow_mode": self.cfg.shadow_mode,
            "kill_switch": self.cfg.kill_switch,
            "persisted_kill": self._state.kill_switch_activated,
            "day_utc": self._state.day_utc,
            "realized_pnl": self._state.realized_pnl,
            "consecutive_losses": self._state.consecutive_losses,
            "recovery_mode": self._state.recovery_mode,
            "sizing_multiplier": self.get_sizing_multiplier(),
        }

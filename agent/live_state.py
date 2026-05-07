"""Shared live-state persistence for the MT5 loop and dashboard API."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from agent.config import ARTIFACTS, AgentConfig


def live_state_path() -> Path:
    raw = os.environ.get("HARD_LIVE_STATE_PATH", "").strip()
    return Path(raw) if raw else ARTIFACTS / "live_state.json"


def read_live_state(path: Path | None = None) -> dict[str, Any]:
    state_path = path or live_state_path()
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def write_live_state(state: dict[str, Any], path: Path | None = None) -> None:
    state_path = path or live_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    tmp_path.replace(state_path)


def update_live_state(patch: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    state = read_live_state(path)
    state.update(patch)
    write_live_state(state, path)
    return state


def cycle_to_dashboard(result: Any) -> dict[str, Any]:
    return {
        "timestamp_utc": result.timestamp_utc,
        "market_data_source": result.market_data_source,
        "signal": {
            "direction": result.signal.direction if result.signal else "unknown",
            "pred_class": result.signal.pred_class if result.signal else 1,
            "score": result.signal.score if result.signal else 0.0,
            "p_up": result.signal.p_up if result.signal else 0.33,
            "p_down": result.signal.p_down if result.signal else 0.33,
            "regime": result.regime,
            "sentiment": result.genai_sentiment,
            "event_risk": result.event_risk,
            "conviction": result.signal.conviction if result.signal else "low",
        },
        "execution_plan": {
            "mode": result.execution_plan.mode if result.execution_plan else "flat",
            "direction": result.execution_plan.direction if result.execution_plan else "flat",
            "stop_loss": result.execution_plan.stop_loss_price if result.execution_plan else None,
            "take_profit": result.execution_plan.take_profit_price if result.execution_plan else None,
        },
        "sr_prediction": result.sr_prediction,
        "volatility": result.volatility,
        "orders_submitted": result.orders_submitted,
        "notes": result.notes,
    }


def news_from_cycle(result: Any, config: AgentConfig) -> dict[str, Any]:
    return {
        "headlines": getattr(result, "news_headlines", []),
        "sentiment": result.genai_sentiment,
        "event_risk": result.event_risk,
        "status": getattr(result, "news_status", "unknown"),
        "model": getattr(result, "genai_model", config.genai.model),
        "enabled": config.genai.news_enabled,
        "configured": config.genai.can_call(),
    }


def config_snapshot(config: AgentConfig) -> dict[str, Any]:
    return {
        "broker": config.broker,
        "symbol": config.yahoo_symbol,
        "broker_symbol": config.broker_symbol_name,
        "bar_frequency": config.bar_frequency,
        "shadow_mode": config.shadow_mode,
        "kill_switch": config.kill_switch,
        "grid_enabled": config.grid.enabled,
    }


def to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value

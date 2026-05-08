"""Dashboard data endpoints with GenAI sentiment headline storage."""

from __future__ import annotations

from typing import Any

from agent.config import AgentConfig
from agent.live_state import read_live_state
from agent.risk.risk_guard import RiskConfig, RiskGuard
from fastapi import APIRouter

router = APIRouter(tags=["dashboard"])

_last_cycle: dict[str, Any] = {}
_last_news: dict[str, Any] = {
    "headlines": [],
    "sentiment": 0.0,
    "event_risk": False,
    "status": "waiting_for_cycle",
}


def _provider_from_base_url(base_url: str) -> str:
    if "openrouter.ai" in base_url:
        return "OpenRouter"
    if "open.bigmodel.cn" in base_url:
        return "BigModel"
    return "OpenAI-compatible"


def _genai_config(config: AgentConfig) -> dict[str, Any]:
    return {
        "model": config.genai.model,
        "provider": _provider_from_base_url(config.genai.base_url),
        "base_url": config.genai.base_url,
        "news_enabled": config.genai.news_enabled,
        "configured": config.genai.can_call(),
    }


def _news_with_config(news: dict[str, Any], config: AgentConfig) -> dict[str, Any]:
    return {
        **news,
        "model": config.genai.model,
        "provider": _provider_from_base_url(config.genai.base_url),
        "enabled": config.genai.news_enabled,
        "configured": config.genai.can_call(),
    }


@router.get("/dashboard")
async def get_dashboard() -> dict[str, Any]:
    """Current state snapshot: positions, P&L, signals, risk state, news."""
    global _last_cycle, _last_news
    config = AgentConfig.from_env()
    live_state = read_live_state()
    live_config = live_state.get("config") or {}
    live_news = live_state.get("news") or _last_news

    risk_cfg = RiskConfig(
        shadow_mode=config.shadow_mode,
        kill_switch=config.kill_switch,
        state_path=config.risk_state_path,
    )
    guard = RiskGuard(risk_cfg)
    guard.refresh_day()

    return {
        "config": {
            "broker": live_config.get("broker", config.broker),
            "symbol": live_config.get("symbol", config.yahoo_symbol),
            "broker_symbol": live_config.get("broker_symbol", config.broker_symbol_name),
            "bar_frequency": live_config.get("bar_frequency", config.bar_frequency),
            "shadow_mode": live_config.get("shadow_mode", config.shadow_mode),
            "kill_switch": live_config.get("kill_switch", config.kill_switch),
            "grid_enabled": live_config.get("grid_enabled", config.grid.enabled),
            "genai": _genai_config(config),
        },
        "risk": live_state.get("risk") or guard.state_dict(),
        "account": live_state.get("account"),
        "positions": live_state.get("positions", []),
        "orders": live_state.get("orders", []),
        "live_status": {
            "status": live_state.get("status", "waiting_for_live_loop"),
            "updated_at_utc": live_state.get("updated_at_utc"),
            "error": live_state.get("error"),
        },
        "last_cycle": live_state.get("last_cycle") or _last_cycle,
        "cycles": live_state.get("cycles", []),
        "news": _news_with_config(live_news, config),
    }


@router.post("/dashboard/refresh")
async def refresh_dashboard() -> dict[str, Any]:
    """Return the latest live-loop cycle without starting a second trading loop."""
    return (await get_dashboard()).get("last_cycle") or {}


def _collect_news_from_cycle(result: Any) -> dict[str, Any]:
    """Extract news headline data captured during the autonomous cycle."""
    config = AgentConfig.from_env()
    return {
        "headlines": getattr(result, "news_headlines", []),
        "sentiment": result.genai_sentiment,
        "event_risk": result.event_risk,
        "status": getattr(result, "news_status", "unknown"),
        "model": getattr(result, "genai_model", config.genai.model),
        "provider": _provider_from_base_url(config.genai.base_url),
        "enabled": config.genai.news_enabled,
        "configured": config.genai.can_call(),
    }


@router.get("/regime")
async def get_regime() -> dict[str, Any]:
    """Current regime + S/R levels."""
    state = read_live_state()
    last_cycle = state.get("last_cycle") or _last_cycle
    return last_cycle.get("sr_prediction", {})


@router.get("/news")
async def get_news() -> dict[str, Any]:
    """Recent news + sentiment scores with full headline breakdown."""
    state = read_live_state()
    return _news_with_config(state.get("news") or _last_news, AgentConfig.from_env())

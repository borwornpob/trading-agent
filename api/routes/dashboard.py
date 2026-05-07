"""Dashboard data endpoints with GenAI sentiment headline storage."""

from __future__ import annotations

from typing import Any

from agent.autonomous_loop import run_cycle
from agent.config import AgentConfig
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

    risk_cfg = RiskConfig(
        shadow_mode=config.shadow_mode,
        kill_switch=config.kill_switch,
        state_path=config.risk_state_path,
    )
    guard = RiskGuard(risk_cfg)
    guard.refresh_day()

    return {
        "config": {
            "broker": config.broker,
            "symbol": config.yahoo_symbol,
            "broker_symbol": config.broker_symbol_name,
            "bar_frequency": config.bar_frequency,
            "shadow_mode": config.shadow_mode,
            "kill_switch": config.kill_switch,
            "grid_enabled": config.grid.enabled,
            "genai": _genai_config(config),
        },
        "risk": guard.state_dict(),
        "last_cycle": _last_cycle,
        "news": _news_with_config(_last_news, config),
    }


@router.post("/dashboard/refresh")
async def refresh_dashboard() -> dict[str, Any]:
    """Trigger a new autonomous cycle and return results."""
    global _last_cycle, _last_news
    result = run_cycle()

    _last_cycle = {
        "timestamp_utc": result.timestamp_utc,
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
            "direction": result.execution_plan.direction
            if result.execution_plan
            else "flat",
            "stop_loss": result.execution_plan.stop_loss_price
            if result.execution_plan
            else None,
            "take_profit": result.execution_plan.take_profit_price
            if result.execution_plan
            else None,
        },
        "sr_prediction": result.sr_prediction,
        "volatility": result.volatility,
        "orders_submitted": result.orders_submitted,
        "notes": result.notes,
    }

    # Collect full news sentiment data from the cycle
    _last_news = _collect_news_from_cycle(result)

    return _last_cycle


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
    return _last_cycle.get("sr_prediction", {})


@router.get("/news")
async def get_news() -> dict[str, Any]:
    """Recent news + sentiment scores with full headline breakdown."""
    return _news_with_config(_last_news, AgentConfig.from_env())

"""Live configuration tweaking endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["config"])


class ConfigUpdate(BaseModel):
    broker: str | None = None
    shadow_mode: bool | None = None
    kill_switch: bool | None = None
    grid_enabled: bool | None = None
    grid_max_levels: int | None = None
    ensemble_primary_weight: float | None = None
    volume_units: float | None = None


@router.put("/config")
async def update_config(update: ConfigUpdate) -> dict[str, Any]:
    """Tweak live parameters via environment variable overrides.

    Note: these changes are in-process only and reset on restart.
    For persistent changes, update .env file.
    """
    import os

    changes: dict[str, Any] = {}

    if update.broker is not None:
        broker = update.broker.strip().lower()
        if broker not in {"ctrader", "mt5"}:
            raise HTTPException(status_code=400, detail="broker must be 'ctrader' or 'mt5'")
        os.environ["HARD_BROKER"] = broker
        changes["broker"] = broker

    if update.shadow_mode is not None:
        os.environ["HARD_SHADOW_MODE"] = str(update.shadow_mode).lower()
        changes["shadow_mode"] = update.shadow_mode

    if update.kill_switch is not None:
        os.environ["HARD_KILL_SWITCH"] = str(update.kill_switch).lower()
        changes["kill_switch"] = update.kill_switch

    if update.grid_enabled is not None:
        os.environ["HARD_GRID_ENABLED"] = str(update.grid_enabled).lower()
        changes["grid_enabled"] = update.grid_enabled

    if update.grid_max_levels is not None:
        os.environ["HARD_GRID_MAX_LEVELS"] = str(update.grid_max_levels)
        changes["grid_max_levels"] = update.grid_max_levels

    if update.ensemble_primary_weight is not None:
        w = max(0.0, min(1.0, update.ensemble_primary_weight))
        os.environ["HARD_ENSEMBLE_PRIMARY_WEIGHT"] = str(w)
        changes["ensemble_primary_weight"] = w

    if update.volume_units is not None:
        if os.environ.get("HARD_BROKER", "ctrader").strip().lower() == "mt5":
            os.environ["MT5_VOLUME_LOTS"] = str(update.volume_units)
        else:
            os.environ["HARD_VOLUME_UNITS"] = str(update.volume_units)
        changes["volume_units"] = update.volume_units

    return {"updated": changes}

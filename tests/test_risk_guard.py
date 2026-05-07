"""Tests for risk guard."""

import pytest
from pathlib import Path
from agent.risk.risk_guard import RiskConfig, RiskGuard


def test_risk_guard_shadow_mode(artifacts_dir: Path):
    cfg = RiskConfig(shadow_mode=True, kill_switch=False, state_path=artifacts_dir / "risk.json")
    guard = RiskGuard(cfg)
    ok, reason = guard.allow_order(volume=100, signed_direction=1)
    assert ok
    assert reason == "shadow_ok"


def test_risk_guard_kill_switch(artifacts_dir: Path):
    cfg = RiskConfig(shadow_mode=False, kill_switch=True, state_path=artifacts_dir / "risk.json")
    guard = RiskGuard(cfg)
    ok, reason = guard.allow_order(volume=100, signed_direction=1)
    assert not ok
    assert reason == "kill_switch"


def test_risk_guard_max_volume(artifacts_dir: Path):
    cfg = RiskConfig(shadow_mode=False, kill_switch=False, max_order_volume=100, state_path=artifacts_dir / "risk.json")
    guard = RiskGuard(cfg)
    ok, reason = guard.allow_order(volume=200, signed_direction=1)
    assert not ok
    assert reason == "max_order_volume"


def test_risk_guard_daily_loss_activates_kill(artifacts_dir: Path):
    cfg = RiskConfig(shadow_mode=False, kill_switch=False, max_daily_loss_units=100, state_path=artifacts_dir / "risk.json")
    guard = RiskGuard(cfg)
    guard.refresh_day()
    guard.record_pnl(-150)
    assert guard._state.kill_switch_activated


def test_risk_guard_recovery_mode(artifacts_dir: Path):
    cfg = RiskConfig(shadow_mode=False, kill_switch=False, state_path=artifacts_dir / "risk.json")
    guard = RiskGuard(cfg)
    guard.refresh_day()
    guard.record_pnl(-10)
    guard.record_pnl(-10)
    guard.record_pnl(-10)
    assert guard.get_sizing_multiplier() == 0.5
    guard.record_pnl(10)
    assert guard.get_sizing_multiplier() == 1.0

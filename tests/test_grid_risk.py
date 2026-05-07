"""Tests for grid risk."""

import pytest
from agent.risk.grid_risk import GridRiskConfig, check_grid_risk, compute_grid_level_sizes, compute_total_grid_exposure


def test_check_grid_risk_allowed():
    cfg = GridRiskConfig()
    result = check_grid_risk(cfg, bounce_probability=0.6, event_risk=False, regime="ranging")
    assert result.allowed
    assert result.max_allowed_levels == 3


def test_check_grid_risk_disabled():
    cfg = GridRiskConfig(enabled=False)
    result = check_grid_risk(cfg, bounce_probability=0.6, event_risk=False, regime="ranging")
    assert not result.allowed


def test_check_grid_risk_event_risk():
    cfg = GridRiskConfig()
    result = check_grid_risk(cfg, bounce_probability=0.6, event_risk=True, regime="ranging")
    assert not result.allowed


def test_check_grid_risk_wrong_regime():
    cfg = GridRiskConfig()
    result = check_grid_risk(cfg, bounce_probability=0.6, event_risk=False, regime="trending_up")
    assert not result.allowed


def test_check_grid_risk_low_bounce():
    cfg = GridRiskConfig(min_bounce_probability=0.55)
    result = check_grid_risk(cfg, bounce_probability=0.4, event_risk=False, regime="ranging")
    assert not result.allowed


def test_check_grid_risk_recovery_mode():
    cfg = GridRiskConfig(recovery_size_reduction=0.5)
    result = check_grid_risk(cfg, bounce_probability=0.6, event_risk=False, regime="ranging", in_recovery_mode=True)
    assert result.allowed
    assert result.adjusted_base_size == 0.5


def test_compute_grid_level_sizes():
    sizes = compute_grid_level_sizes(1.0, 3, decay=0.7)
    assert len(sizes) == 3
    assert sizes[0] == 1.0
    assert abs(sizes[1] - 0.7) < 1e-10
    assert abs(sizes[2] - 0.49) < 1e-10


def test_compute_total_grid_exposure():
    total = compute_total_grid_exposure(1.0, 3, decay=0.7)
    assert abs(total - 2.19) < 0.01

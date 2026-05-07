"""Timeframe-aware artifact selection tests."""

from __future__ import annotations

from agent.config import AgentConfig


def test_15m_defaults_use_15m_artifacts(monkeypatch) -> None:
    monkeypatch.setenv("HARD_BAR_FREQUENCY", "15m")
    monkeypatch.delenv("HARD_MODEL_PATH", raising=False)
    monkeypatch.delenv("HARD_SR_MODEL_PATH", raising=False)
    monkeypatch.delenv("HARD_REGIME_MODEL_PATH", raising=False)

    cfg = AgentConfig.from_env()

    assert cfg.model_path.name == "gold_lgbm_15m.pkl"
    assert cfg.sr_model_path.name == "sr_predictor_15m.pkl"
    assert cfg.regime_model_path.name == "regime_gmm_15m.pkl"


def test_daily_defaults_use_daily_artifacts(monkeypatch) -> None:
    monkeypatch.setenv("HARD_BAR_FREQUENCY", "daily")
    monkeypatch.delenv("HARD_MODEL_PATH", raising=False)
    monkeypatch.delenv("HARD_SR_MODEL_PATH", raising=False)
    monkeypatch.delenv("HARD_REGIME_MODEL_PATH", raising=False)

    cfg = AgentConfig.from_env()

    assert cfg.model_path.name == "gold_lgbm.pkl"
    assert cfg.sr_model_path.name == "sr_predictor.pkl"
    assert cfg.regime_model_path.name == "regime_gmm.pkl"

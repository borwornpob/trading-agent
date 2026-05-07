"""Environment-backed configuration for the gold trading agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _truthy(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_float(name: str, default: float = 0.0) -> float:
    v = os.environ.get(name)
    return float(v) if v else default


def _env_int(name: str, default: int = 0) -> int:
    v = os.environ.get(name)
    return int(v) if v else default


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


@dataclass(frozen=True)
class CTraderSettings:
    client_id: str | None
    client_secret: str | None
    access_token: str | None
    refresh_token: str | None
    account_id: int | None
    symbol_id: int | None
    demo: bool
    redirect_uri: str | None

    @classmethod
    def from_env(cls) -> CTraderSettings:
        aid = os.environ.get("CTRADER_ACCOUNT_ID")
        sid = os.environ.get("CTRADER_SYMBOL_ID")
        return cls(
            client_id=os.environ.get("CTRADER_CLIENT_ID"),
            client_secret=os.environ.get("CTRADER_CLIENT_SECRET"),
            access_token=os.environ.get("CTRADER_ACCESS_TOKEN"),
            refresh_token=os.environ.get("CTRADER_REFRESH_TOKEN"),
            account_id=int(aid) if aid else None,
            symbol_id=int(sid) if sid else None,
            demo=_truthy("CTRADER_DEMO", default=True),
            redirect_uri=os.environ.get("CTRADER_REDIRECT_URI", "http://localhost:8080/callback"),
        )

    def can_authenticate(self) -> bool:
        return bool(
            self.client_id
            and self.client_secret
            and self.access_token
            and self.account_id
            and self.symbol_id,
        )


@dataclass(frozen=True)
class MT5Settings:
    api_path: Path | None
    host: str
    port: int
    connection_timeout_seconds: float
    command_timeout_seconds: float
    rate_count: int
    magic: int

    @classmethod
    def from_env(cls) -> MT5Settings:
        api_path_raw = _env("MT5_API_PATH")
        return cls(
            api_path=Path(api_path_raw).expanduser() if api_path_raw else None,
            host=_env("MT5_HOST", "0.0.0.0"),
            port=_env_int("MT5_PORT", 1111),
            connection_timeout_seconds=_env_float("MT5_CONNECTION_TIMEOUT_SECONDS", 120.0),
            command_timeout_seconds=_env_float("MT5_COMMAND_TIMEOUT_SECONDS", 45.0),
            rate_count=_env_int("MT5_RATE_COUNT", 1500),
            magic=_env_int("MT5_MAGIC", 240501),
        )


@dataclass(frozen=True)
class GenAISettings:
    api_key: str | None
    base_url: str
    model: str
    news_enabled: bool
    news_cache_hours: int
    news_rss_feeds: list[str]

    @classmethod
    def from_env(cls) -> GenAISettings:
        feeds_raw = _env("GENAI_NEWS_RSS_FEEDS")
        feeds = [f.strip() for f in feeds_raw.split(",") if f.strip()]
        return cls(
            api_key=os.environ.get("GENAI_API_KEY"),
            base_url=_env("GENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
            model=_env("GENAI_MODEL", "glm-4-flash"),
            news_enabled=_truthy("GENAI_NEWS_ENABLED", default=True),
            news_cache_hours=_env_int("GENAI_NEWS_CACHE_HOURS", 4),
            news_rss_feeds=feeds or [
                "https://www.investing.com/rss/news_301.rss",
            ],
        )

    def can_call(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class GridSettings:
    enabled: bool
    max_levels: int
    total_exposure_cap: float
    sizing_decay: float

    @classmethod
    def from_env(cls) -> GridSettings:
        return cls(
            enabled=_truthy("HARD_GRID_ENABLED", default=True),
            max_levels=_env_int("HARD_GRID_MAX_LEVELS", 3),
            total_exposure_cap=_env_float("HARD_GRID_TOTAL_EXPOSURE_CAP", 2.0),
            sizing_decay=_env_float("HARD_GRID_SIZING_DECAY", 0.7),
        )


@dataclass(frozen=True)
class AgentConfig:
    broker: str
    yahoo_symbol: str
    broker_symbol_name: str
    bar_frequency: str
    session_filter: str | None
    shadow_mode: bool
    kill_switch: bool
    ensemble_primary_weight: float
    volume_units: float
    max_position_volume: float
    max_order_volume: float
    max_daily_loss_units: float
    risk_state_path: Path
    model_path: Path
    sr_model_path: Path
    regime_model_path: Path
    dataset_parquet: Path
    grid: GridSettings
    genai: GenAISettings

    @classmethod
    def from_env(cls) -> AgentConfig:
        broker = _env("HARD_BROKER", "ctrader").lower()
        bar_frequency = _env("HARD_BAR_FREQUENCY", "daily")
        default_model_path = _default_model_path(bar_frequency)
        default_sr_model_path = _default_sr_model_path(bar_frequency)
        default_regime_model_path = _default_regime_model_path(bar_frequency)
        default_volume = 0.01 if broker == "mt5" else 100.0
        default_max_position = 0.05 if broker == "mt5" else 500.0
        default_max_order = 0.02 if broker == "mt5" else 200.0
        return cls(
            broker=broker,
            yahoo_symbol=_env("HARD_YAHOO_SYMBOL", "GC=F"),
            broker_symbol_name=_env("HARD_BROKER_SYMBOL", "XAUUSD"),
            bar_frequency=bar_frequency,
            session_filter=_env("HARD_SESSION_FILTER") or None,
            shadow_mode=_truthy("HARD_SHADOW_MODE", default=True),
            kill_switch=_truthy("HARD_KILL_SWITCH", default=False),
            ensemble_primary_weight=max(0.0, min(1.0, _env_float("HARD_ENSEMBLE_PRIMARY_WEIGHT", 0.65))),
            volume_units=_env_float("MT5_VOLUME_LOTS", default_volume)
            if broker == "mt5"
            else _env_float("HARD_VOLUME_UNITS", default_volume),
            max_position_volume=_env_float("HARD_MAX_POSITION_VOLUME", default_max_position),
            max_order_volume=_env_float("HARD_MAX_ORDER_VOLUME", default_max_order),
            max_daily_loss_units=_env_float("HARD_MAX_DAILY_LOSS_UNITS", 10_000.0),
            risk_state_path=Path(_env("HARD_RISK_STATE", str(ARTIFACTS / "risk_state.json"))),
            model_path=Path(_env("HARD_MODEL_PATH", str(default_model_path))),
            sr_model_path=Path(_env("HARD_SR_MODEL_PATH", str(default_sr_model_path))),
            regime_model_path=Path(_env("HARD_REGIME_MODEL_PATH", str(default_regime_model_path))),
            dataset_parquet=Path(_env("HARD_DATASET_PARQUET", str(ARTIFACTS / "gold_training.parquet"))),
            grid=GridSettings.from_env(),
            genai=GenAISettings.from_env(),
        )


def _default_model_path(bar_frequency: str) -> Path:
    freq = bar_frequency.strip().lower()
    if freq in {"15m", "15min", "15minute"}:
        return ARTIFACTS / "gold_lgbm_15m.pkl"
    if freq in {"1h", "hourly"}:
        return ARTIFACTS / "gold_lgbm_1h.pkl"
    if freq in {"4h", "4hour"}:
        return ARTIFACTS / "gold_lgbm_4h.pkl"
    return ARTIFACTS / "gold_lgbm.pkl"


def _default_sr_model_path(bar_frequency: str) -> Path:
    freq = bar_frequency.strip().lower()
    if freq in {"15m", "15min", "15minute"}:
        return ARTIFACTS / "sr_predictor_15m.pkl"
    return ARTIFACTS / "sr_predictor.pkl"


def _default_regime_model_path(bar_frequency: str) -> Path:
    freq = bar_frequency.strip().lower()
    if freq in {"15m", "15min", "15minute"}:
        return ARTIFACTS / "regime_gmm_15m.pkl"
    if freq in {"1h", "hourly"}:
        return ARTIFACTS / "regime_gmm_1h.pkl"
    if freq in {"4h", "4hour"}:
        return ARTIFACTS / "regime_gmm_4h.pkl"
    return ARTIFACTS / "regime_gmm.pkl"


@dataclass(frozen=True)
class DashboardConfig:
    host: str
    port: int
    reload: bool

    @classmethod
    def from_env(cls) -> DashboardConfig:
        return cls(
            host=_env("DASHBOARD_HOST", "0.0.0.0"),
            port=_env_int("DASHBOARD_PORT", 8000),
            reload=_truthy("DASHBOARD_RELOAD", default=False),
        )

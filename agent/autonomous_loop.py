"""Autonomous trading loop: perceive → infer → govern → execute.

Concepts: Ch 00 (agent loop pattern), Ch 23 (operational risk, kill switches).
"""

from __future__ import annotations

import datetime as dt
from email.utils import parsedate_to_datetime
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from agent.config import AgentConfig, CTraderSettings, MT5Settings
from agent.data.genai_sentiment import SentimentCache, score_batch, sentiment_to_features
from agent.data.news_fetcher import NewsCache, fetch_rss_headlines
from agent.data.pipeline import (
    build_feature_table,
    fetch_ohlcv_yahoo,
    feature_columns_present,
    is_intraday,
)
from agent.data.session_analyzer import get_session_name
from agent.data.sr_features import compute_all_sr_features
from agent.models.ensemble import MomentumQuantileClassifier, predict_ensemble
from agent.models.lgbm_model import load_bundle, predict_table
from agent.models.mtf_signal import MTFSignal, aggregate_mtf_signals
from agent.models.regime import GMMRegimeDetector
from agent.models.sr_predictor import SRPredictor
from agent.models.volatility import GARCHVolatility, compute_volatility_forecast
from agent.risk.grid_risk import GridRiskConfig, check_grid_risk
from agent.risk.position_manager import (
    OrderIntent,
    PositionSnapshot,
    compute_order_intents,
    make_client_order_id,
)
from agent.risk.risk_guard import RiskConfig, RiskGuard
from agent.risk.session_risk import get_session_multiplier
from agent.strategy.adaptive_executor import ExecutionPlan, route_execution
from agent.strategy.gold_strategy import TradeSignal, signal_from_prediction
from agent.strategy.signal_gates import run_all_gates


NEWS_SENTIMENT_LIMIT = 5


def _grid_plan_to_order_intents(
    plan: ExecutionPlan,
    *,
    symbol: str,
    timestamp_ns: int,
) -> list[OrderIntent]:
    """Convert a range-grid plan into broker order intents."""
    intents: list[OrderIntent] = []
    side = "buy" if plan.direction == "long" else "sell"
    for idx, level in enumerate(plan.grid_levels):
        price = None if not level.get("is_limit", False) else float(level["price"])
        label = str(level.get("label") or f"grid_{idx}")
        intents.append(
            OrderIntent(
                symbol=symbol,
                side=side,
                volume=float(level["size"]),
                price=price,
                stop_loss=plan.stop_loss_price,
                take_profit=plan.take_profit_price,
                label=label,
                client_order_id=make_client_order_id(
                    symbol,
                    side,
                    price,
                    timestamp_ns + idx,
                ),
            )
        )
    return intents


@dataclass
class CycleResult:
    timestamp_utc: str
    signal: TradeSignal | None = None
    regime: str = "unknown"
    execution_plan: ExecutionPlan | None = None
    sr_prediction: dict[str, Any] = field(default_factory=dict)
    volatility: dict[str, Any] = field(default_factory=dict)
    risk_state: dict[str, Any] = field(default_factory=dict)
    genai_sentiment: float = 0.0
    event_risk: bool = False
    news_headlines: list[dict[str, Any]] = field(default_factory=list)
    news_status: str = "not_requested"
    genai_model: str = ""
    session: str = "unknown"
    orders_submitted: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _latest_news_items(items: list[Any], limit: int = NEWS_SENTIMENT_LIMIT) -> list[Any]:
    indexed = list(enumerate(items))

    def sort_key(entry: tuple[int, Any]) -> tuple[float, int]:
        idx, item = entry
        try:
            published = parsedate_to_datetime(item.published_utc)
            ts = published.timestamp()
        except Exception:
            ts = float("-inf")
        return ts, -idx

    return [item for _, item in sorted(indexed, key=sort_key, reverse=True)[:limit]]


def run_cycle(
    *,
    config: AgentConfig | None = None,
    ctrader_settings: CTraderSettings | None = None,
    mt5_settings: MT5Settings | None = None,
    mt5_gateway: Any | None = None,
) -> CycleResult:
    """Run one autonomous cycle: perceive → infer → govern → execute."""
    config = config or AgentConfig.from_env()
    ctrader_settings = ctrader_settings or CTraderSettings.from_env()
    mt5_settings = mt5_settings or MT5Settings.from_env()

    now_utc = dt.datetime.now(dt.UTC)
    result = CycleResult(
        timestamp_utc=now_utc.isoformat(),
        session=get_session_name(now_utc),
    )
    notes: list[str] = []

    # ── PERCEIVE: fetch data ──────────────────────────────────
    end = now_utc.date().isoformat()
    start_dt = now_utc - dt.timedelta(days=400)
    start = start_dt.date().isoformat()

    try:
        if is_intraday(config.bar_frequency):
            raw = fetch_ohlcv_yahoo(
                config.yahoo_symbol,
                "",
                "",
                frequency=config.bar_frequency,
                period="60d",
            )
        else:
            raw = fetch_ohlcv_yahoo(config.yahoo_symbol, start, end, frequency=config.bar_frequency)
    except Exception as e:
        notes.append(f"data_fetch_error:{e}")
        result.notes = notes
        return result

    if raw.is_empty():
        notes.append("no_data")
        result.notes = notes
        return result

    # Feature engineering
    features = compute_all_sr_features(build_feature_table(raw, bar_frequency=config.bar_frequency))
    feature_names = feature_columns_present(features)

    # ── PERCEIVE: GenAI news sentiment ────────────────────────
    sentiment_score = 0.0
    event_risk = False
    result.genai_model = config.genai.model
    if not config.genai.news_enabled:
        result.news_status = "disabled"
    elif not config.genai.can_call():
        result.news_status = "missing_api_key"
    else:
        try:
            items = fetch_rss_headlines(config.genai.news_rss_feeds)
            items = _latest_news_items(items)
            if not items:
                result.news_status = "no_headlines"
            scored = score_batch(items, config.genai)
            if scored:
                sent_df = sentiment_to_features(scored)
                sentiment_score = float(sent_df["weighted_sentiment"].item())
                event_risk = any(r.event_risk for r in scored)
                source_by_hash = {item.content_hash: item for item in items}
                result.news_headlines = [
                    {
                        "headline": s.headline,
                        "sentiment": s.sentiment,
                        "confidence": s.confidence,
                        "key_drivers": s.key_drivers,
                        "impact_horizon": s.impact_horizon,
                        "event_risk": s.event_risk,
                        "gold_relevant": s.gold_relevant,
                        "source": source_by_hash.get(s.content_hash).source
                        if source_by_hash.get(s.content_hash)
                        else "",
                        "url": source_by_hash.get(s.content_hash).url
                        if source_by_hash.get(s.content_hash)
                        else "",
                    }
                    for s in scored[:20]
                ]
                result.news_status = "scored"
        except Exception as e:
            notes.append(f"genai_error:{e}")
            result.news_status = "error"

    result.genai_sentiment = sentiment_score
    result.event_risk = event_risk

    # ── INFER: model predictions ──────────────────────────────
    # Load primary model
    try:
        bundle = load_bundle(config.model_path)
    except Exception:
        notes.append("no_model_bundle")
        result.notes = notes
        return result

    model = bundle["model"]
    feats = tuple(bundle.get("feature_names", list(feature_names)))
    secondary = bundle.get("ensemble_secondary")

    # Predict
    last = features.tail(1)
    if secondary is not None:
        pred_df = predict_ensemble(model, last, feats, secondary, primary_weight=config.ensemble_primary_weight)
    else:
        pred_df = predict_table(model, last, feats)

    row = pred_df.row(0, named=True)

    # ── INFER: regime detection ───────────────────────────────
    regime_name = "unknown"
    try:
        regime_detector = GMMRegimeDetector.load(config.regime_model_path)
        regime_result = regime_detector.predict_latest(features)
        regime_name = regime_result.regime_name
    except Exception:
        notes.append("regime_model_not_available")
    result.regime = regime_name

    # ── INFER: S/R prediction ─────────────────────────────────
    sr_pred = None
    try:
        sr_model = SRPredictor.load(config.sr_model_path)
        sr_pred = sr_model.predict(features)
        result.sr_prediction = {
            "predicted_high": sr_pred.predicted_high,
            "predicted_low": sr_pred.predicted_low,
            "bounce_probability": sr_pred.bounce_probability,
        }
    except Exception:
        notes.append("sr_model_not_available")

    # ── INFER: volatility forecast ────────────────────────────
    current_price = float(last["close"].item())
    vol_forecast = compute_volatility_forecast(features, current_price=current_price)
    result.volatility = {
        "conditional_vol": vol_forecast.conditional_vol,
        "vol_regime": vol_forecast.vol_regime,
        "trailing_stop_distance": vol_forecast.trailing_stop_distance,
    }

    # ── GOVERN: signal + gates ─────────────────────────────────
    signal = signal_from_prediction(
        row, regime=regime_name,
        sentiment_score=sentiment_score,
        event_risk=event_risk,
    )
    result.signal = signal

    gates_passed, gate_results = run_all_gates(
        signal.direction, signal.p_up, signal.p_down, signal.score,
        sentiment_score, regime_name,
    )

    if not gates_passed:
        notes.append(f"gates_failed:{[g.reason for g in gate_results if not g.passed]}")
        result.notes = notes
        return result

    # ── GOVERN: risk checks ────────────────────────────────────
    risk_cfg = RiskConfig(
        shadow_mode=config.shadow_mode,
        kill_switch=config.kill_switch,
        max_position_volume=config.max_position_volume,
        max_order_volume=config.max_order_volume,
        max_daily_loss_units=config.max_daily_loss_units,
        state_path=config.risk_state_path,
    )
    guard = RiskGuard(risk_cfg)
    guard.refresh_day()
    result.risk_state = guard.state_dict()

    session_mult = get_session_multiplier(now_utc)
    sizing_mult = guard.get_sizing_multiplier()
    base_size = config.volume_units * session_mult * sizing_mult

    vol = features["atr"].tail(1).item() if "atr" in features.columns else None

    # ── EXECUTE: route to strategy mode ────────────────────────
    plan = route_execution(
        signal,
        current_price=current_price,
        predicted_high=sr_pred.predicted_high if sr_pred else None,
        predicted_low=sr_pred.predicted_low if sr_pred else None,
        bounce_probability=sr_pred.bounce_probability if sr_pred else 0.5,
        vol_trailing_stop_distance=vol_forecast.trailing_stop_distance,
        grid_enabled=config.grid.enabled,
        max_grid_levels=config.grid.max_levels,
        grid_sizing_decay=config.grid.sizing_decay,
        base_size=base_size,
        atr=float(vol) if vol else None,
    )
    result.execution_plan = plan

    # ── EXECUTE: submit orders (if not shadow) ─────────────────
    if plan.direction != "flat" and not config.shadow_mode and not config.kill_switch:
        if config.broker == "mt5":
            from agent.mt5_gateway import MT5Gateway
            gw = mt5_gateway or MT5Gateway(mt5_settings)
            try:
                if mt5_gateway is None:
                    gw.connect()
                positions = gw.reconcile()
                current_pos = None
                for p in positions:
                    if p.symbol == config.broker_symbol_name:
                        current_pos = PositionSnapshot(
                            symbol=config.broker_symbol_name,
                            side=p.side,
                            volume=p.volume,
                            entry_price=p.entry_price,
                            unrealized_pnl=p.pnl,
                        )
                        break

                timestamp_ns = int(now_utc.timestamp() * 1e9)
                if plan.mode == "range_grid" and current_pos is None:
                    intents = _grid_plan_to_order_intents(
                        plan,
                        symbol=config.broker_symbol_name,
                        timestamp_ns=timestamp_ns,
                    )
                else:
                    intents = compute_order_intents(
                        target_direction=plan.direction,
                        target_size=base_size,
                        current_position=current_pos,
                        symbol=config.broker_symbol_name,
                        stop_loss=plan.stop_loss_price,
                        take_profit=plan.take_profit_price,
                        timestamp_ns=timestamp_ns,
                    )

                for intent in intents:
                    ok, reason = guard.allow_order(volume=float(intent.volume), signed_direction=1 if intent.side == "buy" else -1)
                    if ok and reason != "shadow_ok":
                        if intent.is_market():
                            gw.submit_market_order(
                                symbol=intent.symbol,
                                side_buy=intent.side == "buy",
                                volume=float(intent.volume),
                                label=intent.label,
                                stop_loss=intent.stop_loss,
                                take_profit=intent.take_profit,
                            )
                            result.orders_submitted.append({
                                "side": intent.side, "volume": intent.volume,
                                "type": "market", "label": intent.label,
                                "broker": "mt5",
                            })
                        else:
                            gw.submit_limit_order(
                                symbol=intent.symbol,
                                side_buy=intent.side == "buy",
                                volume=float(intent.volume),
                                price=float(intent.price),
                                label=intent.label,
                                stop_loss=intent.stop_loss,
                                take_profit=intent.take_profit,
                            )
                            result.orders_submitted.append({
                                "side": intent.side, "volume": intent.volume,
                                "price": intent.price, "type": "limit", "label": intent.label,
                                "broker": "mt5",
                            })
                        guard.record_order_sent()
                    else:
                        notes.append(f"order_blocked:{reason}")
            except Exception as e:
                notes.append(f"execution_error:{e}")
            finally:
                if mt5_gateway is None:
                    gw.disconnect()
        elif config.broker == "ctrader" and ctrader_settings.can_authenticate():
            from agent.ctrader_gateway import CTraderGateway
            gw = CTraderGateway(ctrader_settings)
            try:
                gw.connect()
                # Reconcile current positions
                positions = gw.reconcile()
                current_pos = None
                for p in positions:
                    if p.symbol_id == ctrader_settings.symbol_id:
                        current_pos = PositionSnapshot(
                            symbol=config.broker_symbol_name,
                            side=p.side,
                            volume=p.volume,
                            entry_price=p.entry_price,
                        )
                        break

                # Compute order intents
                timestamp_ns = int(now_utc.timestamp() * 1e9)
                if plan.mode == "range_grid" and current_pos is None:
                    intents = _grid_plan_to_order_intents(
                        plan,
                        symbol=config.broker_symbol_name,
                        timestamp_ns=timestamp_ns,
                    )
                else:
                    intents = compute_order_intents(
                        target_direction=plan.direction,
                        target_size=base_size,
                        current_position=current_pos,
                        symbol=config.broker_symbol_name,
                        stop_loss=plan.stop_loss_price,
                        take_profit=plan.take_profit_price,
                        timestamp_ns=timestamp_ns,
                    )

                for intent in intents:
                    ok, reason = guard.allow_order(volume=float(intent.volume), signed_direction=1 if intent.side == "buy" else -1)
                    if ok and reason != "shadow_ok":
                        if intent.is_market():
                            gw.submit_market_order(
                                side_buy=intent.side == "buy",
                                volume=int(intent.volume),
                                label=intent.label,
                                stop_loss=intent.stop_loss,
                                take_profit=intent.take_profit,
                            )
                            result.orders_submitted.append({
                                "side": intent.side, "volume": intent.volume,
                                "type": "market", "label": intent.label,
                                "broker": "ctrader",
                            })
                        else:
                            gw.submit_limit_order(
                                side_buy=intent.side == "buy",
                                volume=int(intent.volume),
                                price=intent.price,
                                label=intent.label,
                            )
                            result.orders_submitted.append({
                                "side": intent.side, "volume": intent.volume,
                                "price": intent.price, "type": "limit", "label": intent.label,
                                "broker": "ctrader",
                            })
                        guard.record_order_sent()
                    else:
                        notes.append(f"order_blocked:{reason}")
            except Exception as e:
                notes.append(f"execution_error:{e}")
            finally:
                gw.disconnect()
        else:
            notes.append(f"{config.broker}:not_configured")
    elif plan.direction != "flat" and config.shadow_mode:
        notes.append("shadow_mode:no_orders_submitted")

    result.notes = notes
    return result

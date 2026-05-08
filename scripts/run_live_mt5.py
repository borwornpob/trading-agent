"""Run the live agent with a persistent MT5 bridge connection.

Usage:
    python -m scripts.run_live_mt5 --continuous --interval 300 --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import datetime as dt
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _cycle_summary(result: Any, account: Any | None = None) -> dict[str, Any]:
    from agent.live_state import to_jsonable

    return {
        "timestamp": result.timestamp_utc,
        "market_data_source": result.market_data_source,
        "account": to_jsonable(account) if account is not None else None,
        "direction": result.signal.direction if result.signal else "flat",
        "pred_class": result.signal.pred_class if result.signal else None,
        "p_up": result.signal.p_up if result.signal else None,
        "p_down": result.signal.p_down if result.signal else None,
        "score": result.signal.score if result.signal else None,
        "conviction": result.signal.conviction if result.signal else None,
        "regime": result.regime,
        "execution_mode": result.execution_plan.mode if result.execution_plan else "flat",
        "orders": result.orders_submitted,
        "notes": result.notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live MT5 trading agent")
    parser.add_argument("--continuous", action="store_true", help="Run in a continuous loop")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between cycles")
    parser.add_argument("--json", action="store_true", help="Write JSON summaries")
    args = parser.parse_args()

    from agent.autonomous_loop import run_cycle
    from agent.config import AgentConfig, MT5Settings
    from agent.live_state import (
        config_snapshot,
        cycle_to_dashboard,
        news_from_cycle,
        to_jsonable,
        update_live_state,
    )
    from agent.mt5_gateway import MT5Gateway

    config = AgentConfig.from_env()
    mt5_settings = MT5Settings.from_env()
    if config.broker != "mt5":
        raise SystemExit(f"HARD_BROKER must be mt5 for this runner, got {config.broker!r}")

    gateway = MT5Gateway(mt5_settings)
    print(
        json.dumps(
            {
                "event": "starting_mt5_live_runner",
                "host": mt5_settings.host,
                "port": mt5_settings.port,
                "broker_symbol": config.broker_symbol_name,
                "bar_frequency": config.bar_frequency,
                "shadow_mode": config.shadow_mode,
                "kill_switch": config.kill_switch,
                "interval": args.interval,
            },
            default=str,
        ),
        flush=True,
    )

    try:
        gateway.connect()
        account = gateway.get_account()
        update_live_state({
            "status": "connected",
            "updated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "config": config_snapshot(config),
            "account": to_jsonable(account),
            "positions": [],
            "orders": [],
            "last_cycle": {},
        })
        print(json.dumps({"event": "mt5_connected", "account": to_jsonable(account)}, default=str), flush=True)

        def run_once() -> None:
            result = run_cycle(config=config, mt5_settings=mt5_settings, mt5_gateway=gateway)
            account_latest = gateway.get_account()
            positions = gateway.reconcile()
            state = update_live_state({
                "status": "running",
                "updated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                "config": config_snapshot(config),
                "risk": result.risk_state,
                "account": to_jsonable(account_latest),
                "positions": to_jsonable(positions),
                "last_cycle": cycle_to_dashboard(result),
                "news": news_from_cycle(result, config),
            })
            if result.orders_submitted:
                orders = list(state.get("orders", []))
                orders.extend({
                    "timestamp_utc": result.timestamp_utc,
                    **to_jsonable(order),
                } for order in result.orders_submitted)
                update_live_state({"orders": orders[-200:]})
            cycles = list(state.get("cycles", []))
            cycles.append({
                "timestamp_utc": result.timestamp_utc,
                "market_data_source": result.market_data_source,
                "direction": result.signal.direction if result.signal else "flat",
                "pred_class": result.signal.pred_class if result.signal else None,
                "p_up": result.signal.p_up if result.signal else None,
                "p_down": result.signal.p_down if result.signal else None,
                "score": result.signal.score if result.signal else None,
                "conviction": result.signal.conviction if result.signal else None,
                "regime": result.regime,
                "execution_mode": result.execution_plan.mode if result.execution_plan else "flat",
                "orders_count": len(result.orders_submitted),
                "notes": result.notes,
            })
            update_live_state({"cycles": cycles[-500:]})
            summary = _cycle_summary(result, account=account_latest)
            if args.json:
                print(json.dumps(summary, default=str), flush=True)
            else:
                print(
                    (
                        f"{summary['timestamp']} "
                        f"data={summary['market_data_source']} "
                        f"direction={summary['direction']} "
                        f"regime={summary['regime']} "
                        f"mode={summary['execution_mode']} "
                        f"orders={len(summary['orders'])} "
                        f"notes={summary['notes']}"
                    ),
                    flush=True,
                )

        if args.continuous:
            while True:
                try:
                    run_once()
                except Exception as exc:
                    update_live_state({
                        "status": "cycle_error",
                        "updated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                        "error": str(exc),
                    })
                    print(json.dumps({"event": "cycle_error", "error": str(exc)}, default=str), flush=True)
                time.sleep(max(1, args.interval))
        else:
            run_once()
    except KeyboardInterrupt:
        print(json.dumps({"event": "stopped"}), flush=True)
    finally:
        gateway.disconnect()


if __name__ == "__main__":
    main()

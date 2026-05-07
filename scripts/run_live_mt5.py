"""Run the live agent with a persistent MT5 bridge connection.

Usage:
    python -m scripts.run_live_mt5 --continuous --interval 300 --json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _cycle_summary(result: Any, account: Any | None = None) -> dict[str, Any]:
    return {
        "timestamp": result.timestamp_utc,
        "account": _jsonable(account) if account is not None else None,
        "direction": result.signal.direction if result.signal else "flat",
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
        print(json.dumps({"event": "mt5_connected", "account": _jsonable(account)}, default=str), flush=True)

        def run_once() -> None:
            result = run_cycle(config=config, mt5_settings=mt5_settings, mt5_gateway=gateway)
            summary = _cycle_summary(result, account=account)
            if args.json:
                print(json.dumps(summary, default=str), flush=True)
            else:
                print(
                    (
                        f"{summary['timestamp']} "
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

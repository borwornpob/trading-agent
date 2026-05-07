"""Run one autonomous trading cycle (or continuous loop).

Usage: python -m scripts.run_live [--continuous] [--interval 300]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live trading agent")
    parser.add_argument("--continuous", action="store_true", help="Run in continuous loop")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between cycles")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    from agent.autonomous_loop import run_cycle

    def run_once() -> None:
        result = run_cycle()
        if args.json:
            print(json.dumps({
                "timestamp": result.timestamp_utc,
                "direction": result.signal.direction if result.signal else "flat",
                "regime": result.regime,
                "execution_mode": result.execution_plan.mode if result.execution_plan else "flat",
                "orders": result.orders_submitted,
                "notes": result.notes,
            }, indent=2, default=str))
        else:
            print(f"\n{'='*60}")
            print(f"  Cycle: {result.timestamp_utc}")
            print(f"  Regime: {result.regime}")
            if result.signal:
                print(f"  Signal: {result.signal.direction} (score={result.signal.score:.3f}, conviction={result.signal.conviction})")
                print(f"  P(UP)={result.signal.p_up:.3f}  P(DOWN)={result.signal.p_down:.3f}")
            if result.execution_plan:
                print(f"  Execution: {result.execution_plan.mode} mode")
                if result.execution_plan.stop_loss_price:
                    print(f"  SL={result.execution_plan.stop_loss_price:.2f}  TP={result.execution_plan.take_profit_price:.2f}")
            print(f"  Sentiment: {result.genai_sentiment:.2f}  Event risk: {result.event_risk}")
            if result.orders_submitted:
                print(f"  Orders submitted: {len(result.orders_submitted)}")
            if result.notes:
                print(f"  Notes: {', '.join(result.notes)}")
            print(f"{'='*60}")

    if args.continuous:
        print(f"Starting continuous loop (interval={args.interval}s). Ctrl+C to stop.")
        try:
            while True:
                try:
                    run_once()
                except Exception as e:
                    print(f"Cycle error: {e}")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        run_once()


if __name__ == "__main__":
    main()

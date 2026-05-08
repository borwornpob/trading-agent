"""Direction signal to position mapping.

Concepts: Ch 05 (strategy evaluation), Ch 12 (trading signals).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass(frozen=True)
class TradeSignal:
    direction: str          # "long", "short", "flat"
    pred_class: int         # 0=down, 1=neutral, 2=up
    score: float            # p_up - p_down
    p_up: float
    p_down: float
    regime: str             # trending_up, trending_down, ranging, volatile
    sentiment_score: float  # GenAI sentiment -1 to +1
    event_risk: bool        # GenAI flagged event risk
    conviction: str         # "high", "medium", "low"


def signal_from_prediction(
    row: dict[str, Any],
    *,
    regime: str = "unknown",
    sentiment_score: float = 0.0,
    event_risk: bool = False,
    min_p_up: float = 0.40,
    min_p_down: float = 0.40,
) -> TradeSignal:
    """Convert a prediction row into a structured trade signal."""
    pc = int(row.get("pred_class", 1))
    p_up = float(row.get("p_up", 0.33))
    p_down = float(row.get("p_down", 0.33))
    score = float(row.get("score", 0.0))

    direction = "flat"
    # Classes: 0=down, 1=neutral, 2=up.
    if pc == 2 and p_up >= min_p_up:
        direction = "long"
    elif pc == 0 and p_down >= min_p_down:
        direction = "short"

    # GenAI veto: if strong sentiment against direction, flatten
    if direction == "long" and sentiment_score < -0.6:
        direction = "flat"
    elif direction == "short" and sentiment_score > 0.6:
        direction = "flat"

    conviction = "low"
    max_p = max(p_up, p_down)
    if max_p > 0.55:
        conviction = "high"
    elif max_p > 0.45:
        conviction = "medium"

    return TradeSignal(
        direction=direction,
        pred_class=pc,
        score=score,
        p_up=p_up,
        p_down=p_down,
        regime=regime,
        sentiment_score=sentiment_score,
        event_risk=event_risk,
        conviction=conviction,
    )


def pred_to_side(row: dict[str, Any], *, allow_short: bool = True) -> str:
    """Simple direction mapping (backward compat with hard_demo pattern)."""
    pc = int(row.get("pred_class", 1))
    if pc == 2:
        return "buy"
    if pc == 0 and allow_short:
        return "sell"
    return "flat"

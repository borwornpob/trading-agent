"""Multi-timeframe signal aggregation (15m + 4h + daily).

Concepts: Ch 11 (ensembles), Ch 12 (trading signals), Ch 06 (walk-forward validation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from agent.models.lgbm_model import load_bundle, predict_table
from agent.models.ensemble import predict_ensemble


@dataclass(frozen=True)
class MTFSignal:
    """Consensus signal across multiple timeframes."""
    direction: str              # "long", "short", "flat"
    pred_class: int             # 0=down, 1=neutral, 2=up
    p_up: float
    p_down: float
    score: float
    agreement: int              # How many timeframes agree (0-3)
    daily_pred: int
    h4_pred: int
    m15_pred: int
    regime_name: str = "unknown"


def aggregate_mtf_signals(
    daily_df: pl.DataFrame | None = None,
    h4_df: pl.DataFrame | None = None,
    m15_df: pl.DataFrame | None = None,
    *,
    daily_model: Any = None,
    h4_model: Any = None,
    m15_model: Any = None,
    daily_features: tuple[str, ...] = (),
    h4_features: tuple[str, ...] = (),
    m15_features: tuple[str, ...] = (),
    regime_name: str = "unknown",
) -> MTFSignal:
    """Aggregate predictions from multiple timeframes into a consensus signal.

    Rules:
    - All three agree → take the signal (high conviction)
    - Two of three agree → take the signal (medium conviction)
    - All disagree → flat (no trade)
    """
    predictions = {}

    if daily_df is not None and daily_model is not None and daily_features:
        pred = predict_table(daily_model, daily_df.tail(1), daily_features)
        predictions["daily"] = pred

    if h4_df is not None and h4_model is not None and h4_features:
        pred = predict_table(h4_model, h4_df.tail(1), h4_features)
        predictions["h4"] = pred

    if m15_df is not None and m15_model is not None and m15_features:
        pred = predict_table(m15_model, m15_df.tail(1), m15_features)
        predictions["m15"] = pred

    if not predictions:
        return MTFSignal(
            direction="flat", pred_class=1, p_up=0.33, p_down=0.33,
            score=0.0, agreement=0, daily_pred=1, h4_pred=1, m15_pred=1,
            regime_name=regime_name,
        )

    # Collect votes
    votes = []
    avg_p_up = 0.0
    avg_p_down = 0.0
    avg_score = 0.0
    daily_pred = 1
    h4_pred = 1
    m15_pred = 1

    for name, pred_df in predictions.items():
        row = pred_df.row(0, named=True)
        pc = int(row["pred_class"])
        votes.append(pc)
        avg_p_up += float(row["p_up"])
        avg_p_down += float(row["p_down"])
        avg_score += float(row["score"])

        if name == "daily":
            daily_pred = pc
        elif name == "h4":
            h4_pred = pc
        elif name == "m15":
            m15_pred = pc

    n = len(predictions)
    avg_p_up /= n
    avg_p_down /= n
    avg_score /= n

    # Count agreement
    from collections import Counter
    vote_counts = Counter(votes)
    most_common_class, most_common_count = vote_counts.most_common(1)[0]

    if most_common_count == n:
        direction = _class_to_direction(most_common_class)
    elif most_common_count >= 2:
        direction = _class_to_direction(most_common_class)
    else:
        direction = "flat"
        most_common_class = 1

    return MTFSignal(
        direction=direction,
        pred_class=most_common_class,
        p_up=avg_p_up,
        p_down=avg_p_down,
        score=avg_score,
        agreement=most_common_count,
        daily_pred=daily_pred,
        h4_pred=h4_pred,
        m15_pred=m15_pred,
        regime_name=regime_name,
    )


def _class_to_direction(pred_class: int) -> str:
    if pred_class == 2:
        return "long"
    if pred_class == 0:
        return "short"
    return "flat"

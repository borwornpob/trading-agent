"""Session-aware risk: reduce position size outside London/NY overlap.

Concepts: Ch 02 (market microstructure, session calendars), Ch 23 (risk management).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class SessionRiskConfig:
    reduce_outside_overlap: bool = True
    outside_overlap_multiplier: float = 0.5
    no_trade_off_hours: bool = False
    off_hours_start_utc: int = 22  # 22:00 UTC
    off_hours_end_utc: int = 0     # 00:00 UTC


def get_session_multiplier(
    timestamp_utc: dt.datetime,
    config: SessionRiskConfig | None = None,
) -> float:
    """Return position sizing multiplier based on current session.

    London/NY overlap (13:00-17:00 UTC): 1.0 (full size)
    London only (07:00-13:00 UTC): 0.8
    NY only (17:00-22:00 UTC): 0.8
    Asian (00:00-07:00 UTC): 0.5
    Off hours (22:00-00:00 UTC): 0.0 if no_trade_off_hours else 0.3
    """
    config = config or SessionRiskConfig()
    hour = timestamp_utc.hour

    if not config.reduce_outside_overlap:
        return 1.0

    # London/NY overlap — peak liquidity
    if 13 <= hour < 17:
        return 1.0
    # London session
    if 7 <= hour < 13:
        return 0.8
    # NY session (after overlap)
    if 17 <= hour < 22:
        return 0.8
    # Asian session
    if 0 <= hour < 7:
        return config.outside_overlap_multiplier

    # Off hours (22-24)
    if config.no_trade_off_hours:
        return 0.0
    return 0.3


def is_tradeable_session(
    timestamp_utc: dt.datetime,
    config: SessionRiskConfig | None = None,
) -> bool:
    """Check if current time allows new trade entries."""
    config = config or SessionRiskConfig()
    multiplier = get_session_multiplier(timestamp_utc, config)
    return multiplier > 0.0

"""Grid-specific risk controls: exposure cap, anti-martingale recovery sizing.

Concepts: Ch 23 (risk management), Ch 05 (position sizing).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridRiskConfig:
    enabled: bool = True
    max_levels: int = 3
    total_exposure_cap: float = 2.0
    sizing_decay: float = 0.7
    min_bounce_probability: float = 0.55
    recovery_size_reduction: float = 0.5  # After losing grid, reduce next size


@dataclass(frozen=True)
class GridRiskCheck:
    allowed: bool
    max_allowed_levels: int
    adjusted_base_size: float
    reasons: list[str]


def check_grid_risk(
    config: GridRiskConfig,
    *,
    bounce_probability: float,
    event_risk: bool,
    current_exposure_ratio: float = 0.0,
    base_size: float = 1.0,
    in_recovery_mode: bool = False,
    regime: str = "ranging",
) -> GridRiskCheck:
    """Pre-grid risk checks."""
    reasons: list[str] = []

    if not config.enabled:
        return GridRiskCheck(allowed=False, max_allowed_levels=0, adjusted_base_size=0.0, reasons=["grid_disabled"])

    if regime != "ranging":
        reasons.append(f"regime_{regime}_not_ranging")
        return GridRiskCheck(allowed=False, max_allowed_levels=0, adjusted_base_size=0.0, reasons=reasons)

    if event_risk:
        reasons.append("event_risk_active")
        return GridRiskCheck(allowed=False, max_allowed_levels=0, adjusted_base_size=0.0, reasons=reasons)

    if bounce_probability < config.min_bounce_probability:
        reasons.append(f"bounce_prob_{bounce_probability:.2f}_below_{config.min_bounce_probability}")
        return GridRiskCheck(allowed=False, max_allowed_levels=0, adjusted_base_size=0.0, reasons=reasons)

    if current_exposure_ratio >= config.total_exposure_cap:
        reasons.append(f"exposure_{current_exposure_ratio:.2f}_at_cap_{config.total_exposure_cap}")
        return GridRiskCheck(allowed=False, max_allowed_levels=0, adjusted_base_size=0.0, reasons=reasons)

    adjusted_size = base_size
    if in_recovery_mode:
        adjusted_size *= config.recovery_size_reduction
        reasons.append(f"recovery_mode_size_{config.recovery_size_reduction}")

    return GridRiskCheck(
        allowed=True,
        max_allowed_levels=config.max_levels,
        adjusted_base_size=adjusted_size,
        reasons=reasons,
    )


def compute_grid_level_sizes(
    base_size: float,
    n_levels: int,
    *,
    decay: float = 0.7,
) -> list[float]:
    """Anti-martingale sizing: 1.0x, decay*x, decay^2*x, ..."""
    return [base_size * (decay ** i) for i in range(n_levels)]


def compute_total_grid_exposure(
    base_size: float,
    n_levels: int,
    *,
    decay: float = 0.7,
) -> float:
    """Total exposure if all grid levels fill."""
    sizes = compute_grid_level_sizes(base_size, n_levels, decay=decay)
    return sum(sizes)

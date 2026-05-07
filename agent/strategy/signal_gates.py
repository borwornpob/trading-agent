"""Statistical acceptance gates for signal quality.

Concepts: Ch 05 (strategy evaluation), Ch 23 (kill switches, monitoring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GateResult:
    passed: bool
    gate_name: str
    value: float
    threshold: float
    reason: str


def min_confidence_gate(
    p_up: float,
    p_down: float,
    *,
    min_probability: float = 0.42,
) -> GateResult:
    """Gate: minimum class probability for directional trades."""
    max_p = max(p_up, p_down)
    passed = max_p >= min_probability
    return GateResult(
        passed=passed,
        gate_name="min_confidence",
        value=max_p,
        threshold=min_probability,
        reason="passed" if passed else f"max_prob_{max_p:.3f}_below_{min_probability}",
    )


def min_score_gate(
    score: float,
    *,
    min_score: float = 0.05,
) -> GateResult:
    """Gate: minimum score (p_up - p_down) for directional conviction."""
    passed = abs(score) >= min_score
    return GateResult(
        passed=passed,
        gate_name="min_score",
        value=score,
        threshold=min_score,
        reason="passed" if passed else f"score_{score:.3f}_below_{min_score}",
    )


def sentiment_alignment_gate(
    direction: str,
    sentiment_score: float,
    *,
    max_divergence: float = 0.7,
) -> GateResult:
    """Gate: signal should not strongly diverge from GenAI sentiment."""
    if direction == "flat":
        return GateResult(passed=True, gate_name="sentiment_alignment", value=0.0, threshold=max_divergence, reason="flat_no_check")

    is_aligned = (
        (direction == "long" and sentiment_score >= -max_divergence)
        or (direction == "short" and sentiment_score <= max_divergence)
    )
    return GateResult(
        passed=is_aligned,
        gate_name="sentiment_alignment",
        value=sentiment_score,
        threshold=max_divergence,
        reason="aligned" if is_aligned else f"direction_{direction}_vs_sentiment_{sentiment_score:.2f}",
    )


def regime_compatibility_gate(
    regime: str,
    direction: str,
    *,
    volatile_allowed: bool = False,
) -> GateResult:
    """Gate: no new trades in volatile regime (unless explicitly allowed)."""
    if regime == "volatile" and not volatile_allowed:
        return GateResult(
            passed=False, gate_name="regime_compatibility",
            value=0.0, threshold=1.0, reason="volatile_regime_blocked",
        )

    if regime == "ranging" and direction != "flat":
        return GateResult(
            passed=True, gate_name="regime_compatibility",
            value=0.0, threshold=1.0, reason="ranging_grid_eligible",
        )

    return GateResult(passed=True, gate_name="regime_compatibility", value=0.0, threshold=1.0, reason="ok")


def run_all_gates(
    direction: str,
    p_up: float,
    p_down: float,
    score: float,
    sentiment_score: float,
    regime: str,
    *,
    min_probability: float = 0.42,
    min_score_threshold: float = 0.05,
) -> tuple[bool, list[GateResult]]:
    """Run all statistical gates. Returns (all_passed, individual_results)."""
    if direction == "flat":
        return True, []

    results = [
        min_confidence_gate(p_up, p_down, min_probability=min_probability),
        min_score_gate(score, min_score=min_score_threshold),
        sentiment_alignment_gate(direction, sentiment_score),
        regime_compatibility_gate(regime, direction),
    ]

    all_passed = all(r.passed for r in results)
    return all_passed, results

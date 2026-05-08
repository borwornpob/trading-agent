"""Gold signal mapping tests."""

from __future__ import annotations

from agent.strategy.gold_strategy import pred_to_side, signal_from_prediction


def test_signal_mapping_uses_zero_based_model_classes() -> None:
    short = signal_from_prediction({"pred_class": 0, "p_down": 0.5, "p_up": 0.1, "score": -0.4})
    neutral = signal_from_prediction({"pred_class": 1, "p_down": 0.1, "p_up": 0.1, "score": 0.0})
    long = signal_from_prediction({"pred_class": 2, "p_down": 0.1, "p_up": 0.5, "score": 0.4})

    assert short.direction == "short"
    assert neutral.direction == "flat"
    assert long.direction == "long"


def test_pred_to_side_uses_zero_based_model_classes() -> None:
    assert pred_to_side({"pred_class": 0}) == "sell"
    assert pred_to_side({"pred_class": 1}) == "flat"
    assert pred_to_side({"pred_class": 2}) == "buy"

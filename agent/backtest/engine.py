"""Walk-forward backtest engine.

Concepts: Ch 06 (walk-forward CV, embargo), Ch 08 (backtesting), Ch 05 (strategy evaluation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from agent.strategy.adaptive_executor import route_execution
from agent.strategy.gold_strategy import TradeSignal


@dataclass(frozen=True)
class TrainTestRanges:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def walk_forward_ranges(
    n_rows: int,
    *,
    min_train_size: int = 400,
    test_size: int = 100,
    embargo: int = 15,
    max_folds: int | None = None,
) -> list[TrainTestRanges]:
    """Expanding-window walk-forward splits with embargo gap."""
    if n_rows < min_train_size + embargo + test_size + 1:
        return []

    ranges: list[TrainTestRanges] = []
    test_start = min_train_size + embargo

    while test_start + test_size <= n_rows:
        train_end = test_start - embargo
        if train_end < min_train_size:
            break
        ranges.append(
            TrainTestRanges(
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_start + test_size,
            )
        )
        test_start += test_size

    if max_folds is not None and len(ranges) > max_folds:
        ranges = ranges[-max_folds:]
    return ranges


def final_holdout_ranges(
    n_rows: int,
    *,
    holdout_size: int = 126,
    embargo: int = 15,
) -> TrainTestRanges | None:
    """Single final holdout split."""
    if n_rows < holdout_size + embargo + 50:
        return None
    test_start = n_rows - holdout_size
    train_end = test_start - embargo
    if train_end < 50:
        return None
    return TrainTestRanges(
        train_start=0,
        train_end=train_end,
        test_start=test_start,
        test_end=n_rows,
    )


@dataclass
class BacktestResult:
    equity_curve: list[float]
    returns: list[float]
    trades: list[dict[str, Any]]
    metrics: dict[str, float] = field(default_factory=dict)


def run_vectorized_backtest(
    signals: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    initial_cash: float = 100_000.0,
    commission_rate: float = 0.0002,
    slippage_rate: float = 0.0001,
    units: float = 1.0,
) -> BacktestResult:
    """Simple vectorized backtest: long/short/flat based on pred_class.

    signals must have: timestamp, pred_class
    prices must have: timestamp, close
    """
    if "pred_class" not in signals.columns or "close" not in prices.columns:
        raise ValueError("signals needs pred_class; prices needs close")

    merged = signals.join(
        prices.select("timestamp", "close"), on="timestamp", how="inner"
    ).sort("timestamp")
    if merged.is_empty():
        return BacktestResult(equity_curve=[], returns=[], trades=[], metrics={})

    n = len(merged)
    closes = merged["close"].to_numpy()
    preds = merged["pred_class"].to_numpy()
    timestamps = merged["timestamp"].to_list()

    cash = initial_cash
    position = 0.0
    entry_price = 0.0
    equity = initial_cash
    equity_curve = [equity]
    returns_list = [0.0]
    trades = []

    for i in range(1, n):
        prev_pred = preds[i - 1]
        price = closes[i]
        prev_price = closes[i - 1]

        # PnL from existing position
        if position != 0:
            pnl = position * (price - prev_price) / prev_price * abs(position) * equity
            equity += pnl

        # Signal action (classes: 0=down, 1=neutral, 2=up)
        target = 0.0
        if prev_pred == 2:
            target = units
        elif prev_pred == 0:
            target = -units

        if target != position:
            # Close existing
            if position != 0:
                close_pnl = (
                    position
                    * (price - entry_price)
                    / entry_price
                    * abs(position)
                    * equity
                )
                cost = abs(position) * equity * (commission_rate + slippage_rate)
                equity += close_pnl - cost
                trades.append(
                    {
                        "timestamp": timestamps[i],
                        "side": "sell" if position > 0 else "buy",
                        "price": price,
                        "pnl": close_pnl - cost,
                    }
                )

            # Open new
            if target != 0:
                entry_price = price
                cost = abs(target) * equity * (commission_rate + slippage_rate)
                equity -= cost
            position = target

        ret = (
            (equity - equity_curve[-1]) / equity_curve[-1]
            if equity_curve[-1] > 0
            else 0.0
        )
        equity_curve.append(equity)
        returns_list.append(ret)

    from agent.backtest.metrics import compute_metrics

    metrics = compute_metrics(np.array(returns_list[1:]))

    return BacktestResult(
        equity_curve=equity_curve,
        returns=returns_list,
        trades=trades,
        metrics=metrics,
    )


def run_sl_tp_backtest(
    signals: pl.DataFrame,
    ohlcv: pl.DataFrame,
    *,
    initial_cash: float = 100_000.0,
    commission_rate: float = 0.0002,
    slippage_rate: float = 0.0001,
    units: float = 1.0,
    sl_atr_mult: float = 2.0,
    tp_atr_mult: float = 1.5,
    sr_cap_tp: bool = True,
) -> BacktestResult:
    """Backtest with ATR-based stop-loss / take-profit and optional S/R cap on TP.

    signals must have: timestamp, pred_class (optionally: score, p_up, p_down,
    predicted_high, predicted_low).
    ohlcv must have: timestamp, open, high, low, close.
    """
    # ------------------------------------------------------------------
    # 1. Merge signals + OHLCV
    # ------------------------------------------------------------------
    required_signal_cols = {"timestamp", "pred_class"}
    required_ohlcv_cols = {"timestamp", "open", "high", "low", "close"}
    if not required_signal_cols.issubset(set(signals.columns)):
        raise ValueError("signals needs at least: timestamp, pred_class")
    if not required_ohlcv_cols.issubset(set(ohlcv.columns)):
        raise ValueError("ohlcv needs: timestamp, open, high, low, close")

    signal_cols_to_select = [c for c in signals.columns if c != "index"]
    merged = (
        signals.select(signal_cols_to_select)
        .join(
            ohlcv.select("timestamp", "open", "high", "low", "close"),
            on="timestamp",
            how="inner",
        )
        .sort("timestamp")
    )
    if merged.is_empty():
        return BacktestResult(equity_curve=[], returns=[], trades=[], metrics={})

    n = len(merged)
    opens = merged["open"].to_numpy()
    highs = merged["high"].to_numpy()
    lows = merged["low"].to_numpy()
    closes = merged["close"].to_numpy()
    timestamps = merged["timestamp"].to_list()
    preds = merged["pred_class"].to_numpy()

    # ------------------------------------------------------------------
    # 2. Compute ATR (14-period) if not present
    # ------------------------------------------------------------------
    if "atr" in merged.columns:
        atr = merged["atr"].to_numpy()
    else:
        prev_close = np.empty(n, dtype=np.float64)
        prev_close[0] = closes[0]
        prev_close[1:] = closes[:-1]
        tr = np.maximum(
            highs - lows,
            np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)),
        )
        import pandas as pd

        atr = np.asarray(pd.Series(tr).rolling(14).mean())

    # ------------------------------------------------------------------
    # 3. S/R predictions
    # ------------------------------------------------------------------
    has_sr = "predicted_high" in merged.columns and "predicted_low" in merged.columns
    if has_sr:
        pred_highs = merged["predicted_high"].to_numpy()
        pred_lows = merged["predicted_low"].to_numpy()
    else:
        # Simple rolling 20-period high / low
        import pandas as pd

        pred_highs = np.asarray(pd.Series(highs).rolling(20).max())
        pred_lows = np.asarray(pd.Series(lows).rolling(20).min())

    # ------------------------------------------------------------------
    # 4-6. Walk-through trade logic
    # ------------------------------------------------------------------
    cash = initial_cash
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
    sl_price = 0.0
    tp_price = 0.0
    equity = initial_cash
    equity_curve: list[float] = [equity]
    returns_list: list[float] = [0.0]
    trades: list[dict[str, Any]] = []

    for i in range(n):
        # Mark-to-market equity update based on close
        if position != 0:
            mtm_equity = equity  # equity already tracks M2M loosely; we update below

        # --- Check SL / TP for existing position (intra-bar) ---
        if position != 0 and i > entry_idx:
            bar_high = highs[i]
            bar_low = lows[i]
            bar_close = closes[i]
            hit_sl = False
            hit_tp = False

            if position == 1:  # long
                if bar_low <= sl_price:
                    hit_sl = True
                elif bar_high >= tp_price:
                    hit_tp = True
            else:  # short
                if bar_high >= sl_price:
                    hit_sl = True
                elif bar_low <= tp_price:
                    hit_tp = True

            if hit_sl or hit_tp:
                exit_price: float
                exit_reason: str
                if hit_sl:
                    exit_price = sl_price
                    exit_reason = "stop_loss"
                else:
                    exit_price = tp_price
                    exit_reason = "take_profit"

                raw_pnl = (
                    (exit_price - entry_price) / entry_price * abs(position) * units
                )
                cost = abs(position) * units * (commission_rate + slippage_rate)
                trade_pnl = raw_pnl - cost
                pnl_cash = trade_pnl * equity
                equity += pnl_cash
                direction = "long" if position == 1 else "short"
                trades.append(
                    {
                        "timestamp": timestamps[i],
                        "entry_timestamp": timestamps[entry_idx],
                        "side": "sell" if position == 1 else "buy",
                        "direction": direction,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": trade_pnl,
                        "pnl_pct": raw_pnl,
                        "exit_reason": exit_reason,
                        "bars_held": i - entry_idx,
                        "sl_price": sl_price,
                        "tp_price": tp_price,
                    }
                )
                position = 0
                entry_price = 0.0
                sl_price = 0.0
                tp_price = 0.0
                entry_idx = 0

        # --- Mark-to-market equity for bars still holding position ---
        if position != 0 and i > 0:
            price_change_pct = (closes[i] - closes[i - 1]) / closes[i - 1]
            equity += position * units * price_change_pct * equity

        # --- Check for new signal / entry (next-bar logic: we enter at this bar's open
        #     if signal from previous bar is directional and we are flat or flipping) ---
        if i == 0:
            ret = (
                (equity - equity_curve[-1]) / equity_curve[-1]
                if equity_curve[-1] > 0
                else 0.0
            )
            equity_curve.append(equity)
            returns_list.append(ret)
            continue

        prev_pred = preds[i - 1]
        target = 0
        if prev_pred == 2:
            target = 1  # long
        elif prev_pred == 0:
            target = -1  # short
        # pred_class == 1 => neutral => target = 0

        # If target differs from current position, close then potentially open new
        if target != position:
            # Close existing position at this bar's open
            if position != 0:
                exit_price_val = opens[i]
                raw_pnl = (
                    (exit_price_val - entry_price) / entry_price * abs(position) * units
                )
                cost = abs(position) * units * (commission_rate + slippage_rate)
                trade_pnl = raw_pnl - cost
                pnl_cash = trade_pnl * equity
                equity += pnl_cash
                direction = "long" if position == 1 else "short"
                trades.append(
                    {
                        "timestamp": timestamps[i],
                        "entry_timestamp": timestamps[entry_idx],
                        "side": "sell" if position == 1 else "buy",
                        "direction": direction,
                        "entry_price": entry_price,
                        "exit_price": exit_price_val,
                        "pnl": trade_pnl,
                        "pnl_pct": raw_pnl,
                        "exit_reason": "signal_flip",
                        "bars_held": i - entry_idx,
                        "sl_price": sl_price,
                        "tp_price": tp_price,
                    }
                )
                position = 0

            # Open new position at this bar's open
            if target != 0:
                entry_price = opens[i]
                entry_cost = (
                    abs(target) * units * (commission_rate + slippage_rate) * equity
                )
                equity -= entry_cost
                position = target
                entry_idx = i
                atr_val = atr[i] if not np.isnan(atr[i]) else (highs[i] - lows[i])

                if position == 1:  # long
                    sl_price = entry_price - atr_val * sl_atr_mult
                    tp_price = entry_price + atr_val * tp_atr_mult
                    if sr_cap_tp and has_sr:
                        sr_cap = pred_highs[i]
                        if not np.isnan(sr_cap) and sr_cap < tp_price:
                            tp_price = sr_cap
                else:  # short
                    sl_price = entry_price + atr_val * sl_atr_mult
                    tp_price = entry_price - atr_val * tp_atr_mult
                    if sr_cap_tp and has_sr:
                        sr_cap = pred_lows[i]
                        if not np.isnan(sr_cap) and sr_cap > tp_price:
                            tp_price = sr_cap

        ret = (
            (equity - equity_curve[-1]) / equity_curve[-1]
            if equity_curve[-1] > 0
            else 0.0
        )
        equity_curve.append(equity)
        returns_list.append(ret)

    # --- Close any open position at end of data ---
    if position != 0:
        exit_price_val = closes[-1]
        raw_pnl = (exit_price_val - entry_price) / entry_price * abs(position) * units
        cost = abs(position) * units * (commission_rate + slippage_rate)
        trade_pnl = raw_pnl - cost
        pnl_cash = trade_pnl * equity
        equity += pnl_cash
        direction = "long" if position == 1 else "short"
        trades.append(
            {
                "timestamp": timestamps[-1],
                "entry_timestamp": timestamps[entry_idx],
                "side": "sell" if position == 1 else "buy",
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price_val,
                "pnl": trade_pnl,
                "pnl_pct": raw_pnl,
                "exit_reason": "end_of_data",
                "bars_held": n - 1 - entry_idx,
                "sl_price": sl_price,
                "tp_price": tp_price,
            }
        )
        position = 0

    # ------------------------------------------------------------------
    # 7. Return BacktestResult
    # ------------------------------------------------------------------
    from agent.backtest.metrics import compute_metrics

    metrics = compute_metrics(np.array(returns_list[1:]))

    return BacktestResult(
        equity_curve=equity_curve,
        returns=returns_list,
        trades=trades,
        metrics=metrics,
    )


def run_grid_backtest(
    signals: pl.DataFrame,
    ohlcv: pl.DataFrame,
    *,
    initial_cash: float = 100_000.0,
    commission_rate: float = 0.0002,
    slippage_rate: float = 0.0001,
    units: float = 1.0,
    sl_atr_mult: float = 2.0,
    tp_atr_mult: float = 1.5,
    sr_cap_tp: bool = True,
    grid_max_levels: int = 3,
    grid_sizing_decay: float = 0.7,
    grid_atr_stop_mult: float = 1.0,
) -> BacktestResult:
    """Backtest combining SL/TP logic with smart grid recovery.

    When price moves against the position, the system adds scaled recovery
    positions at pre-computed grid levels between entry and SL, lowering the
    average entry price.  A wider hard stop gives the grid room to work.

    Parameters
    ----------
    signals : pl.DataFrame
        Must have ``timestamp`` and ``pred_class`` columns.
    ohlcv : pl.DataFrame
        Must have ``timestamp``, ``open``, ``high``, ``low``, ``close``.
    initial_cash : float
        Starting equity.
    commission_rate : float
        Per-side commission as a fraction of notional.
    slippage_rate : float
        Per-side slippage as a fraction of notional.
    units : float
        Base position size (fraction of equity).
    sl_atr_mult : float
        ATR multiplier for the initial stop-loss level.
    tp_atr_mult : float
        ATR multiplier for the take-profit level.
    sr_cap_tp : bool
        Whether to cap TP at predicted S/R levels.
    grid_max_levels : int
        Number of total grid levels including the primary entry. This matches
        the live range-grid execution plan.
    grid_sizing_decay : float
        Per-level size decay factor (``size_i = units * decay^i``).
    grid_atr_stop_mult : float
        ATR multiplier used to extend the S/R hard stop distance.

    Returns
    -------
    BacktestResult
    """
    # ------------------------------------------------------------------
    # 1. Merge signals + OHLCV
    # ------------------------------------------------------------------
    required_signal_cols = {"timestamp", "pred_class"}
    required_ohlcv_cols = {"timestamp", "open", "high", "low", "close"}
    if not required_signal_cols.issubset(set(signals.columns)):
        raise ValueError("signals needs at least: timestamp, pred_class")
    if not required_ohlcv_cols.issubset(set(ohlcv.columns)):
        raise ValueError("ohlcv needs: timestamp, open, high, low, close")

    signal_cols_to_select = [c for c in signals.columns if c != "index"]
    merged = (
        signals.select(signal_cols_to_select)
        .join(
            ohlcv.select("timestamp", "open", "high", "low", "close"),
            on="timestamp",
            how="inner",
        )
        .sort("timestamp")
    )
    if merged.is_empty():
        return BacktestResult(equity_curve=[], returns=[], trades=[], metrics={})

    n = len(merged)
    opens = merged["open"].to_numpy()
    highs = merged["high"].to_numpy()
    lows = merged["low"].to_numpy()
    closes = merged["close"].to_numpy()
    timestamps = merged["timestamp"].to_list()
    preds = merged["pred_class"].to_numpy()
    scores = merged["score"].to_numpy() if "score" in merged.columns else np.zeros(n)
    p_ups = merged["p_up"].to_numpy() if "p_up" in merged.columns else np.full(n, 0.5)
    p_downs = merged["p_down"].to_numpy() if "p_down" in merged.columns else np.full(n, 0.5)

    # ------------------------------------------------------------------
    # 2. Compute ATR (14-period)
    # ------------------------------------------------------------------
    if "atr" in merged.columns:
        atr = merged["atr"].to_numpy()
    else:
        prev_close = np.empty(n, dtype=np.float64)
        prev_close[0] = closes[0]
        prev_close[1:] = closes[:-1]
        tr = np.maximum(
            highs - lows,
            np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)),
        )
        import pandas as pd

        atr = np.asarray(pd.Series(tr).rolling(14).mean())

    # ------------------------------------------------------------------
    # 3. S/R predictions
    # ------------------------------------------------------------------
    has_sr = "predicted_high" in merged.columns and "predicted_low" in merged.columns
    if has_sr:
        pred_highs = merged["predicted_high"].to_numpy()
        pred_lows = merged["predicted_low"].to_numpy()
    else:
        import pandas as pd

        pred_highs = np.asarray(pd.Series(highs).rolling(20).max())
        pred_lows = np.asarray(pd.Series(lows).rolling(20).min())

    # ------------------------------------------------------------------
    # 4-6. Walk-through grid trade logic
    # ------------------------------------------------------------------
    equity = initial_cash
    equity_curve: list[float] = [equity]
    returns_list: list[float] = [0.0]
    trades: list[dict[str, Any]] = []

    # Position state
    position = 0  # 0=flat, 1=long, -1=short
    entry_idx = 0
    primary_entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    hard_stop_price = 0.0

    # Grid state
    grid_levels: list[dict[str, Any]] = []
    avg_entry_price = 0.0
    total_size = 0.0
    grid_levels_filled = 0

    def _reset_grid() -> None:
        nonlocal position, entry_idx, primary_entry_price, sl_price, tp_price
        nonlocal hard_stop_price, grid_levels, avg_entry_price, total_size
        nonlocal grid_levels_filled
        position = 0
        entry_idx = 0
        primary_entry_price = 0.0
        sl_price = 0.0
        tp_price = 0.0
        hard_stop_price = 0.0
        grid_levels = []
        avg_entry_price = 0.0
        total_size = 0.0
        grid_levels_filled = 0

    def _close_trade(exit_price: float, exit_reason: str, bar_idx: int) -> None:
        nonlocal equity
        if position == 1:
            raw_pnl = (exit_price - avg_entry_price) / avg_entry_price * total_size
        else:
            raw_pnl = (avg_entry_price - exit_price) / avg_entry_price * total_size
        exit_cost = total_size * (commission_rate + slippage_rate)
        trade_pnl = raw_pnl - exit_cost
        pnl_cash = trade_pnl * equity
        equity += pnl_cash
        direction = "long" if position == 1 else "short"
        trades.append(
            {
                "timestamp": timestamps[bar_idx],
                "entry_timestamp": timestamps[entry_idx],
                "side": "sell" if position == 1 else "buy",
                "direction": direction,
                "entry_price": primary_entry_price,
                "exit_price": exit_price,
                "pnl": trade_pnl,
                "pnl_pct": raw_pnl,
                "exit_reason": exit_reason,
                "bars_held": bar_idx - entry_idx,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "grid_levels_filled": grid_levels_filled,
                "avg_entry_price": avg_entry_price,
                "total_size": total_size,
            }
        )
        _reset_grid()

    for i in range(n):
        # --- Intra-bar: grid fill, hard stop, TP ---
        if position != 0 and i > entry_idx:
            # 1. Fill unfilled grid levels
            for level in grid_levels:
                if not level["filled"]:
                    filled_this_bar = False
                    if position == 1 and lows[i] <= level["price"]:
                        filled_this_bar = True
                    elif position == -1 and highs[i] >= level["price"]:
                        filled_this_bar = True
                    if filled_this_bar:
                        level["filled"] = True
                        grid_levels_filled += 1
                        new_total = total_size + level["size"]
                        avg_entry_price = (
                            avg_entry_price * total_size
                            + level["price"] * level["size"]
                        ) / new_total
                        total_size = new_total
                        # Deduct entry commission for this grid fill
                        entry_cost = (
                            level["size"] * (commission_rate + slippage_rate) * equity
                        )
                        equity -= entry_cost

            # 2. Hard stop (only if ALL grid levels are filled)
            all_grid_filled = all(lvl["filled"] for lvl in grid_levels)
            hit_hard_stop = False
            if all_grid_filled:
                if position == 1 and lows[i] <= hard_stop_price:
                    hit_hard_stop = True
                elif position == -1 and highs[i] >= hard_stop_price:
                    hit_hard_stop = True
            if hit_hard_stop:
                _close_trade(hard_stop_price, "hard_stop", i)

            # 3. Take-profit (only if still in position)
            if position != 0:
                hit_tp = False
                if position == 1 and highs[i] >= tp_price:
                    hit_tp = True
                elif position == -1 and lows[i] <= tp_price:
                    hit_tp = True
                if hit_tp:
                    _close_trade(tp_price, "take_profit", i)

        # --- Mark-to-market equity for bars still holding position ---
        if position != 0 and i > 0:
            price_change_pct = (closes[i] - closes[i - 1]) / closes[i - 1]
            equity += position * total_size * price_change_pct * equity

        # --- Signal check / entry ---
        if i == 0:
            ret = (
                (equity - equity_curve[-1]) / equity_curve[-1]
                if equity_curve[-1] > 0
                else 0.0
            )
            equity_curve.append(equity)
            returns_list.append(ret)
            continue

        prev_pred = preds[i - 1]
        target = 0
        if prev_pred == 2:
            target = 1  # long
        elif prev_pred == 0:
            target = -1  # short

        if target != position:
            # Close existing position at this bar's open
            if position != 0:
                _close_trade(opens[i], "signal_flip", i)

            # Open new position at this bar's open
            if target != 0:
                primary_entry_price = opens[i]
                atr_val = atr[i] if not np.isnan(atr[i]) else (highs[i] - lows[i])

                predicted_high = float(pred_highs[i])
                predicted_low = float(pred_lows[i])
                if np.isnan(predicted_high):
                    predicted_high = primary_entry_price + atr_val * tp_atr_mult
                if np.isnan(predicted_low):
                    predicted_low = primary_entry_price - atr_val * sl_atr_mult

                direction = "long" if target == 1 else "short"
                signal = TradeSignal(
                    direction=direction,
                    pred_class=int(prev_pred),
                    score=float(scores[i - 1]),
                    p_up=float(p_ups[i - 1]),
                    p_down=float(p_downs[i - 1]),
                    regime="ranging",
                    sentiment_score=0.0,
                    event_risk=False,
                    conviction="high",
                )
                plan = route_execution(
                    signal,
                    current_price=primary_entry_price,
                    predicted_high=predicted_high,
                    predicted_low=predicted_low,
                    bounce_probability=1.0,
                    grid_enabled=True,
                    max_grid_levels=grid_max_levels,
                    grid_sizing_decay=grid_sizing_decay,
                    base_size=units,
                    atr=atr_val * grid_atr_stop_mult,
                )

                sl_price = plan.stop_loss_price or primary_entry_price
                tp_price = plan.take_profit_price or primary_entry_price
                hard_stop_price = sl_price

                # Primary entry fills immediately
                position = target
                entry_idx = i
                total_size = units
                avg_entry_price = primary_entry_price
                grid_levels_filled = 1  # primary counts as 1

                # Deduct primary entry commission
                entry_cost = units * (commission_rate + slippage_rate) * equity
                equity -= entry_cost

                # Use the same S/R grid levels as the live range-grid plan.
                grid_levels = [
                    {
                        "price": float(level["price"]),
                        "size": float(level["size"]),
                        "filled": False,
                    }
                    for level in plan.grid_levels
                    if level.get("is_limit", False)
                ]

        # --- Equity curve return ---
        ret = (
            (equity - equity_curve[-1]) / equity_curve[-1]
            if equity_curve[-1] > 0
            else 0.0
        )
        equity_curve.append(equity)
        returns_list.append(ret)

    # --- Close any remaining open position at end of data ---
    if position != 0:
        _close_trade(closes[-1], "end_of_data", n - 1)

    # ------------------------------------------------------------------
    # 7. Return BacktestResult
    # ------------------------------------------------------------------
    from agent.backtest.metrics import compute_metrics

    metrics = compute_metrics(np.array(returns_list[1:]))

    return BacktestResult(
        equity_curve=equity_curve,
        returns=returns_list,
        trades=trades,
        metrics=metrics,
    )

"""GARCH(1,1) volatility forecasting for position sizing and trailing stops.

Concepts: Ch 09 (GARCH, conditional volatility).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


@dataclass(frozen=True)
class VolatilityForecast:
    conditional_vol: float          # Current annualized vol estimate
    forecast_1d: float              # 1-day ahead forecast (annualized)
    forecast_5d: float              # 5-day ahead forecast (annualized)
    vol_regime: str                 # low / normal / high
    trailing_stop_distance: float   # ATR-equivalent in price units


class GARCHVolatility:
    """GARCH(1,1) volatility model for gold returns."""

    def __init__(self) -> None:
        self._omega: float = 0.0
        self._alpha: float = 0.1
        self._beta: float = 0.85
        self._last_cond_var: float = 0.0
        self._long_run_var: float = 0.0
        self._mean_return: float = 0.0
        self._fitted: bool = False

    def fit(self, returns: np.ndarray, *, price_mean: float = 2000.0) -> None:
        """Fit GARCH(1,1) via simple MLE approximation."""
        from arch import arch_model

        r = np.asarray(returns, dtype=np.float64)
        r = r[np.isfinite(r)] * 100  # Scale to percentage for numerical stability

        if len(r) < 50:
            self._fallback_fit(r, price_mean)
            return

        try:
            model = arch_model(r, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
            res = model.fit(disp="off")
            params = res.params
            self._omega = float(params.get("omega", 0.1))
            self._alpha = float(params.get("alpha[1]", 0.1))
            self._beta = float(params.get("beta[1]", 0.85))
            self._last_cond_var = float(res.conditional_volatility.values[-1]) ** 2
        except Exception:
            self._fallback_fit(r, price_mean)

        self._long_run_var = self._omega / (1 - self._alpha - self._beta + 1e-10)
        self._mean_return = float(np.mean(returns))
        self._fitted = True

    def _fallback_fit(self, r: np.ndarray, price_mean: float) -> None:
        """Simple EWMA fallback if arch package fails."""
        self._long_run_var = float(np.var(r))
        self._last_cond_var = self._long_run_var
        self._omega = self._long_run_var * 0.05
        self._alpha = 0.1
        self._beta = 0.85
        self._fitted = True

    def forecast(self, horizon: int = 1, current_price: float = 2000.0) -> VolatilityForecast:
        """Produce volatility forecast."""
        if not self._fitted:
            return VolatilityForecast(
                conditional_vol=0.15, forecast_1d=0.15, forecast_5d=0.15,
                vol_regime="normal", trailing_stop_distance=current_price * 0.02,
            )

        alpha, beta, omega = self._alpha, self._beta, self._omega
        h = max(1, horizon)

        # Multi-step ahead variance forecast
        var_t = self._last_cond_var
        for _ in range(h):
            var_t = omega + alpha * var_t + beta * var_t

        vol_1d = np.sqrt(self._last_cond_var)
        var_5d = self._last_cond_var
        for _ in range(5):
            var_5d = omega + alpha * var_5d + beta * var_5d
        vol_5d = np.sqrt(var_5d)

        annualized = vol_1d * np.sqrt(252) / 100

        regime = "normal"
        if annualized < 0.10:
            regime = "low"
        elif annualized > 0.25:
            regime = "high"

        stop_distance = current_price * vol_1d * 2.0 / 100

        return VolatilityForecast(
            conditional_vol=float(annualized),
            forecast_1d=float(vol_1d / 100 * np.sqrt(252)),
            forecast_5d=float(vol_5d / 100 * np.sqrt(252)),
            vol_regime=regime,
            trailing_stop_distance=float(stop_distance),
        )

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "omega": self._omega,
            "alpha": self._alpha,
            "beta": self._beta,
            "last_cond_var": self._last_cond_var,
            "long_run_var": self._long_run_var,
            "mean_return": self._mean_return,
            "fitted": self._fitted,
        }, path)

    @classmethod
    def load(cls, path: Path) -> GARCHVolatility:
        import joblib

        data = joblib.load(path)
        vol = cls()
        vol._omega = data["omega"]
        vol._alpha = data["alpha"]
        vol._beta = data["beta"]
        vol._last_cond_var = data["last_cond_var"]
        vol._long_run_var = data["long_run_var"]
        vol._mean_return = data["mean_return"]
        vol._fitted = data.get("fitted", True)
        return vol


def compute_volatility_forecast(
    df: pl.DataFrame,
    *,
    current_price: float | None = None,
) -> VolatilityForecast:
    """Convenience: fit GARCH on close returns and produce a 1-day forecast."""
    returns = df["close"].pct_change().drop_nulls().to_numpy()
    price = current_price or float(df["close"].tail(1).item())

    model = GARCHVolatility()
    model.fit(returns, price_mean=price)
    return model.forecast(horizon=1, current_price=price)

"""Statistical models: ARIMA (per series, ADF-driven differencing, small AIC
grid) and Holt-Winters exponential smoothing."""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller

from src.constants import DATE_COL, SERIES_COL, TARGET_COL
from src.logger import get_logger
from src.models.base_model import BaseModel

logger = get_logger(__name__)
warnings.filterwarnings("ignore")  # statsmodels convergence chatter


def _multi_step_predict(forecast_fn, future: pd.DataFrame) -> np.ndarray:
    """Map per-series h-step-ahead forecasts onto future rows (rank of each
    date within its series = forecast step)."""
    preds = np.zeros(len(future))
    pos = future.reset_index(drop=True)
    for sid, g in pos.groupby(SERIES_COL):
        h = len(g)
        f = forecast_fn(sid, h)
        preds[g.index] = f[:h]
    return preds


class ARIMAModel(BaseModel):
    """One ARIMA per series. d chosen by ADF test (p < 0.05 -> stationary,
    d=0). Order picked from a small candidate grid by AIC.

    ponytail: 4-candidate AIC grid instead of pmdarima auto_arima —
    upgrade to pmdarima if order search quality matters.
    """

    name = "ARIMA"

    def __init__(self, auto_order: bool = True, seasonal: bool = False):
        self.auto_order = auto_order
        self.fitted: dict[str, object] = {}
        self.orders: dict[str, tuple] = {}

    def fit(self, train: pd.DataFrame, feature_cols: list[str]) -> None:
        for sid, g in train.groupby(SERIES_COL):
            y = g.sort_values(DATE_COL)[TARGET_COL].astype(float).to_numpy()
            d = 0 if adfuller(y)[1] < 0.05 else 1
            candidates = [(1, d, 1), (2, d, 2), (0, d, 1), (1, d, 0)] if self.auto_order else [(1, 1, 1)]
            best, best_aic = None, np.inf
            for order in candidates:
                try:
                    res = ARIMA(y, order=order).fit()
                    if res.aic < best_aic:
                        best, best_aic, self.orders[sid] = res, res.aic, order
                except Exception:
                    continue
            if best is None:
                raise RuntimeError(f"ARIMA failed to fit any order for series {sid}")
            self.fitted[sid] = best
        logger.info("ARIMA fitted for %d series, orders=%s", len(self.fitted), self.orders)

    def predict(self, future: pd.DataFrame, feature_cols: list[str],
                history: pd.DataFrame | None = None) -> np.ndarray:
        return _multi_step_predict(
            lambda sid, h: np.asarray(self.fitted[sid].forecast(h)), future)


class ExpSmoothingModel(BaseModel):
    name = "ExpSmoothing"

    def __init__(self, seasonal_periods: int = 7):
        self.seasonal_periods = seasonal_periods
        self.fitted: dict[str, object] = {}

    def fit(self, train: pd.DataFrame, feature_cols: list[str]) -> None:
        for sid, g in train.groupby(SERIES_COL):
            y = g.sort_values(DATE_COL)[TARGET_COL].astype(float).to_numpy() + 1e-6
            self.fitted[sid] = ExponentialSmoothing(
                y, trend="add", seasonal="add",
                seasonal_periods=self.seasonal_periods).fit()

    def predict(self, future: pd.DataFrame, feature_cols: list[str],
                history: pd.DataFrame | None = None) -> np.ndarray:
        return _multi_step_predict(
            lambda sid, h: np.asarray(self.fitted[sid].forecast(h)), future)

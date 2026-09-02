"""Forecast interface: produce the next-horizon forecast per series from the
saved best model.

Feature-based models forecast recursively (predict day t+1, append it,
regenerate features, repeat). Sequence/statistical models forecast the whole
horizon directly.
"""
import numpy as np
import pandas as pd

from src.constants import (BEST_MODEL_PATH, DATE_COL, FORECAST_PATH, SERIES_COL,
                           TARGET_COL)
from src.features.feature_engineering import build_features
from src.logger import get_logger
from src.models.lstm_model import LSTMModel
from src.models.statistical_models import ARIMAModel, ExpSmoothingModel
from src.utils.serialization import load_pickle

logger = get_logger(__name__)

_DIRECT_MODELS = (ARIMAModel, ExpSmoothingModel, LSTMModel)


def _future_rows(clean: pd.DataFrame, horizon: int, freq: str) -> pd.DataFrame:
    """Blank future rows per series, carrying static columns forward."""
    rows = []
    for sid, g in clean.groupby(SERIES_COL):
        last = g.sort_values(DATE_COL).iloc[-1]
        dates = pd.date_range(last[DATE_COL], periods=horizon + 1, freq=freq)[1:]
        f = pd.DataFrame({DATE_COL: dates, SERIES_COL: sid})
        for c in ("item_id", "store_id", "sell_price"):
            if c in clean.columns:
                f[c] = last[c]
        if "promotion_flag" in clean.columns:
            f["promotion_flag"] = 0
        f[TARGET_COL] = np.nan
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def generate_forecast(clean: pd.DataFrame, config: dict,
                      model_bundle: dict | None = None) -> pd.DataFrame:
    """Return [date, series_id, forecast] for the configured horizon and
    write reports/forecast.csv."""
    bundle = model_bundle or load_pickle(BEST_MODEL_PATH)
    model, fcols = bundle["model"], bundle["feature_cols"]
    horizon = config["forecasting"]["forecast_horizon"]
    freq = clean.attrs.get("frequency", "D")
    logger.info("Forecasting %d periods with %s", horizon, model.name)

    if isinstance(model, _DIRECT_MODELS):
        future = _future_rows(clean, horizon, freq)
        future["forecast"] = model.predict(future, fcols, history=clean)
    else:
        work = clean.copy()
        for _ in range(horizon):
            step = _future_rows(work, 1, freq)
            feats = build_features(pd.concat([work, step], ignore_index=True),
                                   config, save=False)
            last = feats.groupby(SERIES_COL).tail(1)
            step = step.merge(
                last.assign(forecast=np.clip(model.predict(last, fcols), 0, None))
                    [[SERIES_COL, DATE_COL, "forecast"]],
                on=[SERIES_COL, DATE_COL], how="left")
            step[TARGET_COL] = step["forecast"].fillna(0)
            work = pd.concat([work, step.drop(columns=["forecast"])],
                             ignore_index=True)
        n_hist = len(clean)
        future = work.iloc[n_hist:].rename(columns={TARGET_COL: "forecast"})

    out = future[[DATE_COL, SERIES_COL, "forecast"]].copy()
    out["forecast"] = out["forecast"].clip(lower=0).round(2)
    out.to_csv(FORECAST_PATH, index=False)
    logger.info("Forecast written: %s (%d rows)", FORECAST_PATH, len(out))
    return out

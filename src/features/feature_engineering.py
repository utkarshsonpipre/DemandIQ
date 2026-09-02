"""Feature engineering pipeline. All features are leakage-safe: anything
derived from the target uses only values strictly before the current row
(shift(1) before rolling/expanding).

Feature groups (toggled via config['features']):
  temporal, lag, rolling, decomposition, business, hierarchical
"""
import numpy as np
import pandas as pd

from src.constants import DATE_COL, FEATURES_PATH, SERIES_COL, TARGET_COL
from src.logger import get_logger

logger = get_logger(__name__)


def add_temporal(df: pd.DataFrame, holiday_country: str = "US") -> pd.DataFrame:
    d = df[DATE_COL].dt
    df["day_of_week"] = d.dayofweek
    df["month"] = d.month
    df["quarter"] = d.quarter
    df["day_of_month"] = d.day
    df["week_of_year"] = d.isocalendar().week.astype(int)
    df["is_weekend"] = (d.dayofweek >= 5).astype(int)
    # cyclical encodings so models see Sunday next to Monday
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    try:
        import holidays as hol
        years = range(df[DATE_COL].dt.year.min(), df[DATE_COL].dt.year.max() + 2)
        hdates = pd.to_datetime(sorted(hol.country_holidays(holiday_country, years=years).keys()))
        df["is_holiday"] = df[DATE_COL].isin(hdates).astype(int)
        # days since the most recent holiday on or before each date
        idx = np.searchsorted(hdates.values, df[DATE_COL].values, side="right") - 1
        last_hol = np.where(idx >= 0, hdates.values[np.clip(idx, 0, None)], df[DATE_COL].values)
        df["days_since_last_holiday"] = ((df[DATE_COL].values - last_hol)
                                         / np.timedelta64(1, "D")).astype(int)
    except Exception as e:
        logger.warning("Holiday features skipped: %s", e)
        df["is_holiday"] = 0
        df["days_since_last_holiday"] = 0
    return df


def add_lags(df: pd.DataFrame, lookbacks: list[int]) -> pd.DataFrame:
    g = df.groupby(SERIES_COL)[TARGET_COL]
    for k in lookbacks:
        df[f"lag_{k}"] = g.shift(k)
    return df


def add_rolling(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    # shift(1) first: window covers [t-w, t-1], never the current value
    shifted = df.groupby(SERIES_COL)[TARGET_COL].shift(1)
    for w in windows:
        r = shifted.groupby(df[SERIES_COL]).rolling(w, min_periods=max(2, w // 2))
        df[f"rolling_mean_{w}"] = r.mean().reset_index(level=0, drop=True)
        df[f"rolling_std_{w}"] = r.std().reset_index(level=0, drop=True)
        df[f"rolling_min_{w}"] = r.min().reset_index(level=0, drop=True)
        df[f"rolling_max_{w}"] = r.max().reset_index(level=0, drop=True)
    if 14 in windows:
        r14 = shifted.groupby(df[SERIES_COL]).rolling(14, min_periods=7)
        df["rolling_median_14"] = r14.median().reset_index(level=0, drop=True)
    return df


def add_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Past-only trend/seasonal/residual proxies (no future leakage)."""
    shifted = df.groupby(SERIES_COL)[TARGET_COL].shift(1)
    df["trend_component"] = (shifted.groupby(df[SERIES_COL])
                             .rolling(30, min_periods=7).mean()
                             .reset_index(level=0, drop=True))
    # seasonal: expanding mean of the target per (series, day-of-week), past-only
    key = df[SERIES_COL].astype(str) + "_" + df[DATE_COL].dt.dayofweek.astype(str)
    df["seasonal_component"] = (shifted.groupby(key).expanding().mean()
                                .reset_index(level=0, drop=True)) - df["trend_component"]
    df["residual_component"] = shifted - df["trend_component"] - df["seasonal_component"]
    return df


def add_business(df: pd.DataFrame) -> pd.DataFrame:
    if "sell_price" in df.columns:
        mean_p = df.groupby(SERIES_COL)["sell_price"].transform("mean")
        df["price_norm"] = df["sell_price"] / (mean_p + 1e-9)
        df["price_change_pct"] = (df.groupby(SERIES_COL)["sell_price"]
                                  .pct_change().fillna(0))
    if "promotion_flag" not in df.columns:
        df["promotion_flag"] = 0
    return df


def add_hierarchical(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("store_id", "item_id"):
        if col in df.columns:
            df[f"{col}_encoded"] = df[col].astype("category").cat.codes
    shifted = df.groupby(SERIES_COL)[TARGET_COL].shift(1)
    for col, name in (("store_id", "store_avg_sales"), ("item_id", "product_avg_sales")):
        if col in df.columns:
            df[name] = (shifted.groupby(df[col]).expanding().mean()
                        .reset_index(level=0, drop=True))
    return df


def build_features(df: pd.DataFrame, config: dict, save: bool = True) -> pd.DataFrame:
    """Run all enabled feature groups; drop warm-up rows lacking the longest
    lag; write features.parquet unless save=False."""
    fcfg = config["features"]
    df = df.sort_values([SERIES_COL, DATE_COL]).reset_index(drop=True)

    if fcfg.get("include_temporal", True):
        df = add_temporal(df, fcfg.get("holiday_country", "US"))
    df = add_lags(df, fcfg.get("lag_lookbacks", [1, 7, 14, 30]))
    df = add_rolling(df, fcfg.get("rolling_windows", [7, 14, 30]))
    if fcfg.get("include_decomposition", True):
        df = add_decomposition(df)
    if fcfg.get("include_business_features", True):
        df = add_business(df)
    if fcfg.get("include_hierarchical", True):
        df = add_hierarchical(df)

    max_lag = max(fcfg.get("lag_lookbacks", [30]))
    before = len(df)
    df = df.dropna(subset=[f"lag_{max_lag}"]).reset_index(drop=True)
    df[feature_columns(df)] = df[feature_columns(df)].fillna(0)
    if save:
        logger.info("Features built: %d columns, dropped %d warm-up rows",
                    len(feature_columns(df)), before - len(df))
        df.to_parquet(FEATURES_PATH, index=False)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model input columns = everything except identifiers, raw target, raw price."""
    exclude = {DATE_COL, SERIES_COL, TARGET_COL, "item_id", "store_id", "sell_price"}
    return [c for c in df.columns if c not in exclude]

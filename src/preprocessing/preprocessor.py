"""Cleaning pipeline: frequency detection, gap reindexing, imputation,
type fixes. Operates per series on the canonical long format."""
import pandas as pd

from src.constants import CLEAN_DATA_PATH, DATE_COL, SERIES_COL, TARGET_COL
from src.logger import get_logger

logger = get_logger(__name__)


def detect_frequency(df: pd.DataFrame) -> str:
    """Infer temporal frequency from the modal date diff of the first series."""
    g = df[df[SERIES_COL] == df[SERIES_COL].iloc[0]]
    diff = g[DATE_COL].sort_values().diff().mode()
    freq_days = int(diff.iloc[0].days) if len(diff) else 1
    freq = {1: "D", 7: "W", 30: "MS", 31: "MS"}.get(freq_days, f"{freq_days}D")
    logger.info("Detected frequency: %s", freq)
    return freq


def clean(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Clean the raw long-format frame:

    1. Drop exact duplicates on (series, date).
    2. Reindex each series to a complete date range at detected frequency.
    3. Impute: sales gaps -> 0 (a missing retail day is a no-sale day);
       price -> forward/backward fill; promo flag -> 0.
    4. Clip negative sales to 0.

    Writes the result to data/processed/clean.parquet.
    """
    logger.info("Cleaning %d rows", len(df))
    df = df.drop_duplicates(subset=[SERIES_COL, DATE_COL], keep="first").copy()
    freq = detect_frequency(df) if config["data"].get("granularity", "auto") == "auto" else config["data"]["granularity"]

    static_cols = [c for c in ("item_id", "store_id") if c in df.columns]
    out = []
    for sid, g in df.groupby(SERIES_COL):
        g = g.set_index(DATE_COL).sort_index()
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq=freq)
        g = g.reindex(full_idx)
        g.index.name = DATE_COL
        g[SERIES_COL] = sid
        for c in static_cols:
            g[c] = g[c].ffill().bfill()
        g[TARGET_COL] = g[TARGET_COL].fillna(0).clip(lower=0)
        if "sell_price" in g.columns:
            g["sell_price"] = g["sell_price"].ffill().bfill()
        if "promotion_flag" in g.columns:
            g["promotion_flag"] = g["promotion_flag"].fillna(0).astype(int)
        out.append(g.reset_index())

    cleaned = pd.concat(out, ignore_index=True)
    cleaned.attrs["frequency"] = freq
    cleaned.to_parquet(CLEAN_DATA_PATH, index=False)
    logger.info("Cleaned data written: %s (%d rows, freq=%s)", CLEAN_DATA_PATH, len(cleaned), freq)
    return cleaned

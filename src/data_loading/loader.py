"""Load demand data into canonical long format:
columns = [date, series_id, item_id, store_id, sales, (sell_price, promotion_flag)].

Supports M5 (kaggle), generic CSV, and synthetic fallback.
"""
from pathlib import Path

import pandas as pd

from src.constants import DATE_COL, SERIES_COL, TARGET_COL
from src.data_loading import downloader
from src.logger import get_logger

logger = get_logger(__name__)


def load_m5(m5_dir: Path, n_series: int = 10) -> pd.DataFrame:
    """Load M5 wide sales file, keep the top-N series by total volume,
    melt to long format, and join calendar dates + sell prices.

    M5 full melt is ~58M rows; top-N subset keeps training under 15 min.
    """
    sales = pd.read_csv(m5_dir / "sales_train_validation.csv")
    cal = pd.read_csv(m5_dir / "calendar.csv", parse_dates=["date"])
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    totals = sales[day_cols].sum(axis=1)
    top = sales.loc[totals.nlargest(n_series).index]
    logger.info("Selected top %d of %d M5 series by volume", len(top), len(sales))

    long = top.melt(
        id_vars=["id", "item_id", "store_id"],
        value_vars=day_cols, var_name="d", value_name=TARGET_COL,
    )
    long = long.merge(cal[["d", "date", "wm_yr_wk", "snap_CA", "snap_TX", "snap_WI"]], on="d")

    prices_path = m5_dir / "sell_prices.csv"
    if prices_path.exists():
        prices = pd.read_csv(prices_path)
        long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    long[SERIES_COL] = long["item_id"] + "_" + long["store_id"]
    # SNAP flag for the row's own state doubles as a promotion-like signal
    state = long["store_id"].str[:2]
    long["promotion_flag"] = 0
    for st in ("CA", "TX", "WI"):
        mask = state == st
        long.loc[mask, "promotion_flag"] = long.loc[mask, f"snap_{st}"]

    cols = [DATE_COL, SERIES_COL, "item_id", "store_id", TARGET_COL, "sell_price", "promotion_flag"]
    long = long[[c for c in cols if c in long.columns]]
    return long.sort_values([SERIES_COL, DATE_COL]).reset_index(drop=True)


def load_csv(path: Path) -> pd.DataFrame:
    """Load a generic long-format CSV. Needs date + sales columns; series_id
    is synthesized from any identifier columns present (or set to 'series_0')."""
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    if TARGET_COL not in df.columns or DATE_COL not in df.columns:
        raise ValueError(f"CSV must contain '{DATE_COL}' and '{TARGET_COL}' columns")
    if SERIES_COL not in df.columns:
        id_cols = [c for c in ("item_id", "product_id", "store_id") if c in df.columns]
        if id_cols:
            df[SERIES_COL] = df[id_cols].astype(str).agg("_".join, axis=1)
        else:
            df[SERIES_COL] = "series_0"
    return df.sort_values([SERIES_COL, DATE_COL]).reset_index(drop=True)


def load_data(config: dict) -> pd.DataFrame:
    """Entry point: load data per config, falling back to synthetic when
    Kaggle is unavailable."""
    dcfg = config["data"]
    source = dcfg.get("source", "kaggle")
    if source == "csv" and dcfg.get("csv_path"):
        return load_csv(Path(dcfg["csv_path"]))
    if source == "kaggle":
        try:
            m5_dir = downloader.download_m5()
            return load_m5(m5_dir, n_series=dcfg.get("n_series", 10))
        except Exception as e:
            logger.warning("Kaggle load failed (%s); falling back to synthetic data.", e)
    if not downloader.SYNTHETIC_PATH.exists():
        downloader.generate_synthetic(n_series=dcfg.get("n_series", 10),
                                      seed=config.get("random_seed", 42))
    return load_csv(downloader.SYNTHETIC_PATH)

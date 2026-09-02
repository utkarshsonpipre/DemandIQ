"""Download the M5 dataset via kagglehub, with a synthetic fallback.

Run directly: ``python -m src.data_loading.downloader``
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import RAW_DIR
from src.logger import get_logger

logger = get_logger(__name__)

SYNTHETIC_PATH = RAW_DIR / "synthetic_sales.csv"


def download_m5() -> Path:
    """Download M5 competition files with kagglehub; returns the dataset dir.

    Requires Kaggle credentials (~/.kaggle/kaggle.json) and accepted
    competition rules.
    """
    import kagglehub
    logger.info("Downloading M5 dataset from Kaggle...")
    path = Path(kagglehub.competition_download("m5-forecasting-accuracy"))
    logger.info("M5 downloaded to %s", path)
    return path


def generate_synthetic(n_series: int = 10, n_days: int = 1095,
                       seed: int = 42) -> Path:
    """Generate realistic synthetic demand data (trend + weekly/yearly
    seasonality + promos + noise) in M5-like long format. Used when Kaggle
    credentials are unavailable so the pipeline runs out of the box."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="D")
    frames = []
    for i in range(n_series):
        base = rng.uniform(50, 200)
        trend = np.linspace(0, rng.uniform(-20, 60), n_days)
        weekly = 15 * np.sin(2 * np.pi * np.arange(n_days) / 7 + rng.uniform(0, 6))
        yearly = 25 * np.sin(2 * np.pi * np.arange(n_days) / 365.25 + rng.uniform(0, 6))
        promo = rng.random(n_days) < 0.05
        noise = rng.normal(0, base * 0.08, n_days)
        sales = np.clip(base + trend + weekly + yearly + promo * base * 0.4 + noise, 0, None)
        price = np.round(rng.uniform(2, 20) * (1 - promo * 0.2), 2)
        frames.append(pd.DataFrame({
            "date": dates,
            "series_id": f"ITEM_{i + 1:02d}_STORE_{i % 3 + 1}",
            "item_id": f"ITEM_{i + 1:02d}",
            "store_id": f"STORE_{i % 3 + 1}",
            "sales": np.round(sales).astype(int),
            "sell_price": price,
            "promotion_flag": promo.astype(int),
        }))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(SYNTHETIC_PATH, index=False)
    logger.info("Synthetic data written: %s (%d rows)", SYNTHETIC_PATH, len(df))
    return SYNTHETIC_PATH


if __name__ == "__main__":
    try:
        download_m5()
    except Exception as e:  # no kaggle creds / rules not accepted
        logger.warning("Kaggle download failed (%s); generating synthetic data.", e)
        generate_synthetic()

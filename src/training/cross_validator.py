"""Walk-forward time-series cross-validation with expanding train windows.

Layout (fractions of the unique-date timeline), for n_folds=3:

    test  = last 10%                      (held out from all folds)
    fold 1: train [0, 70%)  val [70%, 80%)
    fold 2: train [0, 75%)  val [75%, 85%)
    fold 3: train [0, 80%)  val [80%, 90%)

Each fold's validation window slides forward; train always expands from 0.
"""
from dataclasses import dataclass

import pandas as pd

from src.constants import DATE_COL
from src.logger import get_logger

logger = get_logger(__name__)

VAL_FRAC = 0.10
TEST_FRAC = 0.10
STEP_FRAC = 0.05


@dataclass
class Fold:
    fold: int
    train_mask: pd.Series
    val_mask: pd.Series


def make_folds(df: pd.DataFrame, n_folds: int = 3) -> tuple[list[Fold], pd.Series]:
    """Return (folds, test_mask). Masks are boolean, aligned to df.index and
    based purely on date position — every series splits at the same dates."""
    dates = pd.Series(sorted(df[DATE_COL].unique()))
    n = len(dates)

    def at(frac: float) -> pd.Timestamp:
        return dates.iloc[min(int(n * frac), n - 1)]

    test_start = at(1 - TEST_FRAC)
    test_mask = df[DATE_COL] >= test_start

    folds = []
    for i in range(n_folds):
        val_end_frac = (1 - TEST_FRAC) - (n_folds - 1 - i) * STEP_FRAC
        val_start_frac = val_end_frac - VAL_FRAC
        v0, v1 = at(val_start_frac), at(val_end_frac)
        folds.append(Fold(
            fold=i + 1,
            train_mask=df[DATE_COL] < v0,
            val_mask=(df[DATE_COL] >= v0) & (df[DATE_COL] < v1),
        ))
        logger.info("Fold %d: train<%s, val [%s, %s)", i + 1, v0.date(), v0.date(), v1.date())
    logger.info("Test window: >= %s", pd.Timestamp(test_start).date())
    return folds, test_mask

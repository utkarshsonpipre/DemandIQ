"""Abstract base class all forecasting models implement, so the trainer,
evaluator, and forecaster treat every model identically."""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Common interface.

    fit(train, feature_cols)
        Train on the long-format feature frame.
    predict(future, feature_cols, history)
        Return one prediction per row of ``future``. ``future`` rows must be
        time-ordered per series. ``history`` is all data strictly before
        ``future`` — required by sequence/statistical models, ignored by
        feature-based ones.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, train: pd.DataFrame, feature_cols: list[str]) -> None: ...

    @abstractmethod
    def predict(self, future: pd.DataFrame, feature_cols: list[str],
                history: pd.DataFrame | None = None) -> np.ndarray: ...

    def feature_importance(self) -> pd.Series | None:
        """Importance per feature (None if not applicable)."""
        return None

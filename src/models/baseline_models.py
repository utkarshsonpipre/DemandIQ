"""Baselines: moving-average ensemble and linear regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.constants import TARGET_COL
from src.models.base_model import BaseModel


class MovingAverageModel(BaseModel):
    """Ensemble of past rolling means (7/14/30-day), read straight from the
    precomputed leakage-safe rolling_mean_* features."""

    name = "MovingAverage"

    def __init__(self, windows: list[int] | None = None):
        self.windows = windows or [7, 14, 30]

    def fit(self, train: pd.DataFrame, feature_cols: list[str]) -> None:
        pass  # nothing to learn

    def predict(self, future: pd.DataFrame, feature_cols: list[str],
                history: pd.DataFrame | None = None) -> np.ndarray:
        cols = [f"rolling_mean_{w}" for w in self.windows if f"rolling_mean_{w}" in future.columns]
        return future[cols].mean(axis=1).to_numpy()


class LinearRegressionModel(BaseModel):
    name = "LinearRegression"

    def __init__(self):
        self.pipe = make_pipeline(StandardScaler(), LinearRegression())
        self._cols: list[str] = []

    def fit(self, train: pd.DataFrame, feature_cols: list[str]) -> None:
        self._cols = feature_cols
        self.pipe.fit(train[feature_cols], train[TARGET_COL])

    def predict(self, future: pd.DataFrame, feature_cols: list[str],
                history: pd.DataFrame | None = None) -> np.ndarray:
        return self.pipe.predict(future[self._cols])

    def feature_importance(self) -> pd.Series:
        coefs = self.pipe.named_steps["linearregression"].coef_
        return pd.Series(np.abs(coefs), index=self._cols).sort_values(ascending=False)

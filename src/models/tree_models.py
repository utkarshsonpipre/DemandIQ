"""Tree-based models: LightGBM (primary), XGBoost, Random Forest."""
import numpy as np
import pandas as pd

from src.constants import TARGET_COL
from src.models.base_model import BaseModel


class _SklearnLike(BaseModel):
    """Shared fit/predict for any sklearn-API regressor."""

    def __init__(self, estimator):
        self.est = estimator
        self._cols: list[str] = []

    def fit(self, train: pd.DataFrame, feature_cols: list[str]) -> None:
        self._cols = feature_cols
        self.est.fit(train[feature_cols], train[TARGET_COL])

    def predict(self, future: pd.DataFrame, feature_cols: list[str],
                history: pd.DataFrame | None = None) -> np.ndarray:
        return self.est.predict(future[self._cols])

    def feature_importance(self) -> pd.Series:
        imp = getattr(self.est, "feature_importances_", None)
        if imp is None:
            return None
        return pd.Series(imp, index=self._cols).sort_values(ascending=False)


class LightGBMModel(_SklearnLike):
    name = "LightGBM"

    def __init__(self, params: dict, seed: int = 42):
        from lightgbm import LGBMRegressor
        super().__init__(LGBMRegressor(random_state=seed, verbose=-1, **params))


class XGBoostModel(_SklearnLike):
    name = "XGBoost"

    def __init__(self, params: dict, seed: int = 42):
        from xgboost import XGBRegressor
        super().__init__(XGBRegressor(random_state=seed, verbosity=0, **params))


class RandomForestModel(_SklearnLike):
    name = "RandomForest"

    def __init__(self, params: dict, seed: int = 42):
        from sklearn.ensemble import RandomForestRegressor
        super().__init__(RandomForestRegressor(random_state=seed, n_jobs=-1, **params))

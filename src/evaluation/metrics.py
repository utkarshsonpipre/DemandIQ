"""Forecast evaluation metrics: MAE, RMSE, R2, MAPE, SMAPE."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error on a 0-1 scale; zero-actual rows skipped."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE (0-1 scale); rows where both are zero contribute 0."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.where(denom == 0, 0.0, np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom))
    return float(np.mean(ratio))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """All required metrics in one dict (MAPE/SMAPE as percentages)."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2_score": float(r2_score(y_true, y_pred)),
        "mape": round(mape(y_true, y_pred) * 100, 4),
        "smape": round(smape(y_true, y_pred) * 100, 4),
    }

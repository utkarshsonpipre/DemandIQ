"""Training orchestrator: run every enabled model through the same
walk-forward CV folds, log to MLflow, pick the best by mean MAPE
(RMSE tiebreak), refit on train+val, evaluate on the held-out test window,
register to the model registry, and persist comparison artifacts."""
import time

import numpy as np
import pandas as pd

from src.constants import (BEST_MODEL_META_PATH, BEST_MODEL_PATH, COMPARISON_PATH,
                           DATE_COL, FEATURE_IMPORTANCE_PATH, TARGET_COL)
from src.evaluation.metrics import evaluate
from src.features.feature_engineering import feature_columns
from src.logger import get_logger
from src.mlflow_utils import tracker
from src.models.model_factory import create_models
from src.training.cross_validator import make_folds
from src.utils.serialization import save_json, save_pickle

logger = get_logger(__name__)


def _model_params(model) -> dict:
    est = getattr(model, "est", None)
    if est is not None:
        return {k: v for k, v in est.get_params().items()
                if isinstance(v, (int, float, str, bool)) and v is not None}
    return {}


def run_training(features: pd.DataFrame, config: dict) -> dict:
    """Full train/compare/select/register flow. Returns summary dict."""
    fcols = feature_columns(features)
    folds, test_mask = make_folds(features, config["forecasting"]["cv_folds"])
    mlflow_on = tracker.setup(config)
    models = create_models(config)
    horizon = config["forecasting"]["forecast_horizon"]

    rows = []
    for model in models:
        for fold in folds:
            train = features[fold.train_mask]
            val = features[fold.val_mask]
            try:
                t0 = time.time()
                model.fit(train, fcols)
                train_time = time.time() - t0
                t0 = time.time()
                preds = model.predict(val, fcols, history=train)
                infer_ms = (time.time() - t0) / max(len(val), 1) * 1000
                metrics = evaluate(val[TARGET_COL].to_numpy(), preds)
                metrics |= {"training_time_seconds": round(train_time, 2),
                            "inference_time_ms": round(infer_ms, 4)}
                rows.append({"model": model.name, "fold": fold.fold, **metrics})
                logger.info("%s fold %d: MAPE=%.2f%% RMSE=%.2f",
                            model.name, fold.fold, metrics["mape"], metrics["rmse"])
                if mlflow_on:
                    params = _model_params(model) | {
                        "feature_count": len(fcols),
                        "training_set_size": len(train),
                        "validation_set_size": len(val),
                        "forecast_horizon": horizon,
                    }
                    tracker.log_run(model.name, fold.fold, params, metrics,
                                    tags=config["mlflow"].get("run_tags"))
            except Exception as e:
                logger.error("%s fold %d failed: %s", model.name, fold.fold, e,
                             exc_info=True)

    if not rows:
        raise RuntimeError("Every model failed on every fold — check the logs")

    comparison = pd.DataFrame(rows)
    avg = (comparison.groupby("model")[["mape", "rmse", "mae", "r2_score", "smape",
                                        "training_time_seconds", "inference_time_ms"]]
           .mean().sort_values(["mape", "rmse"]))
    avg["rank"] = range(1, len(avg) + 1)
    comparison.to_csv(COMPARISON_PATH, index=False)
    logger.info("Model comparison (avg over folds):\n%s", avg.round(3).to_string())

    # Refit winner on train+val, evaluate once on the untouched test window
    best_name = avg.index[0]
    best_model = next(m for m in models if m.name == best_name)
    trainval = features[~test_mask]
    test = features[test_mask]
    best_model.fit(trainval, fcols)
    test_preds = best_model.predict(test, fcols, history=trainval)
    test_metrics = evaluate(test[TARGET_COL].to_numpy(), test_preds)
    logger.info("Best model %s test MAPE=%.2f%%", best_name, test_metrics["mape"])

    save_pickle({"model": best_model, "feature_cols": fcols}, BEST_MODEL_PATH)
    imp = best_model.feature_importance()
    if imp is not None:
        imp.rename("importance").to_csv(FEATURE_IMPORTANCE_PATH)

    registry_info = {}
    if mlflow_on:
        try:
            registry_info = tracker.register_best(
                config, BEST_MODEL_PATH, best_name,
                avg_mape=float(avg.loc[best_name, "mape"]),
                test_mape=test_metrics["mape"])
        except Exception as e:
            logger.error("Model registration failed: %s", e, exc_info=True)

    summary = {
        "best_model": best_name,
        "avg_cv_mape": float(avg.loc[best_name, "mape"]),
        "test_metrics": test_metrics,
        "comparison": avg.reset_index().to_dict(orient="records"),
        "registry": registry_info,
        "trained_at": pd.Timestamp.now().isoformat(),
        "n_features": len(fcols),
        "train_rows": int(len(trainval)),
        "test_rows": int(len(test)),
    }
    save_json(summary, BEST_MODEL_META_PATH)
    return summary

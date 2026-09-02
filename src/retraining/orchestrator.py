"""Automated retraining: trigger checks (drift / performance / manual) and
the end-to-end retraining pipeline, with an append-only JSONL event log."""
import json
import time

import pandas as pd

from src.constants import RETRAIN_LOG_PATH
from src.logger import get_logger
from src.monitoring.drift_detector import detect_drift
from src.monitoring.performance_monitor import check_performance

logger = get_logger(__name__)


def check_triggers(features: pd.DataFrame, config: dict) -> dict:
    """Evaluate automatic trigger conditions. Returns dict with
    should_retrain flag and the reasons."""
    rcfg = config["retraining"]
    reasons = []
    drift = detect_drift(features, config)
    perf = check_performance(features, config)

    drifted = sum(f["drift"] for f in drift.get("features", []))
    if drifted >= rcfg.get("drift_feature_count_trigger", 3):
        reasons.append(f"Data drift on {drifted} features")
    if perf.get("status") == "DEGRADED":
        reasons.append(f"MAPE degraded {perf['degradation_pct']}% "
                       f"(threshold {perf['threshold_pct']}%)")
    return {"should_retrain": bool(reasons), "reasons": reasons,
            "drift": drift, "performance": perf}


def retrain(config: dict, trigger: str = "manual") -> dict:
    """Full retraining run: reload data, re-validate, rebuild features,
    retrain all models, register if better. Logs the event to JSONL."""
    # imported here to avoid a circular import with main-pipeline modules
    from src.data_loading.loader import load_data
    from src.features.feature_engineering import build_features
    from src.preprocessing.preprocessor import clean
    from src.training.train_pipeline import run_training
    from src.validation.validator import run_validation

    logger.info("Retraining triggered: %s", trigger)
    t0 = time.time()
    raw = load_data(config)
    report = run_validation(raw)
    if not report["valid"]:
        raise ValueError(f"Retraining aborted, data invalid: {report['schema_errors']}")
    cleaned = clean(raw, config)
    features = build_features(cleaned, config)
    summary = run_training(features, config)

    event = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "trigger": trigger,
        "data_points": len(cleaned),
        "n_features": summary["n_features"],
        "cv_folds": config["forecasting"]["cv_folds"],
        "best_model": summary["best_model"],
        "avg_cv_mape": summary["avg_cv_mape"],
        "test_mape": summary["test_metrics"]["mape"],
        "registry": summary.get("registry", {}),
        "duration_seconds": round(time.time() - t0, 1),
    }
    with open(RETRAIN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")
    logger.info("Retraining complete in %.0fs: best=%s test MAPE=%.2f%%",
                event["duration_seconds"], event["best_model"], event["test_mape"])
    return event


def read_retrain_log() -> list[dict]:
    if not RETRAIN_LOG_PATH.exists():
        return []
    with open(RETRAIN_LOG_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

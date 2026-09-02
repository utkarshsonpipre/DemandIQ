"""Model performance monitoring: recompute MAPE on the most recent window of
ground truth and compare against the MAPE recorded at training time."""
import pandas as pd

from src.constants import BEST_MODEL_META_PATH, BEST_MODEL_PATH, DATE_COL, TARGET_COL
from src.evaluation.metrics import evaluate
from src.logger import get_logger
from src.utils.serialization import load_json, load_pickle

logger = get_logger(__name__)


def check_performance(features: pd.DataFrame, config: dict) -> dict:
    """One-step-ahead predictions on the last `recent_window_days` of data,
    compared with training-time test MAPE. Flags degradation beyond the
    configured threshold."""
    if not (BEST_MODEL_PATH.exists() and BEST_MODEL_META_PATH.exists()):
        return {"status": "NO_MODEL"}
    bundle = load_pickle(BEST_MODEL_PATH)
    meta = load_json(BEST_MODEL_META_PATH)
    threshold = config["retraining"].get("mape_degradation_threshold", 0.15)
    window = config["drift_detection"].get("recent_window_days", 30)

    cutoff = features[DATE_COL].max() - pd.Timedelta(days=window)
    recent = features[features[DATE_COL] > cutoff]
    history = features[features[DATE_COL] <= cutoff]
    if len(recent) < 5:
        return {"status": "INSUFFICIENT_DATA"}

    preds = bundle["model"].predict(recent, bundle["feature_cols"], history=history)
    metrics = evaluate(recent[TARGET_COL].to_numpy(), preds)
    baseline = meta["test_metrics"]["mape"]
    degradation = (metrics["mape"] - baseline) / max(baseline, 1e-9)
    degraded = degradation > threshold
    result = {
        "status": "DEGRADED" if degraded else "OK",
        "recent_mape": metrics["mape"],
        "baseline_mape": baseline,
        "degradation_pct": round(degradation * 100, 2),
        "threshold_pct": threshold * 100,
        "checked_at": pd.Timestamp.now().isoformat(),
        **{f"recent_{k}": v for k, v in metrics.items()},
    }
    log = logger.warning if degraded else logger.info
    log("Performance check: %s (recent MAPE %.2f%% vs baseline %.2f%%)",
        result["status"], metrics["mape"], baseline)
    return result

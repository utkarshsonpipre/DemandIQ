"""Data drift detection.

Core: two-sample Kolmogorov-Smirnov test (scipy) on the top-N most important
features + the target, reference = training window vs current = recent window.
Optionally renders an Evidently HTML report when the library is importable.
"""
import pandas as pd
from scipy.stats import ks_2samp

from src.constants import (DATE_COL, DRIFT_HTML_PATH, DRIFT_REPORT_PATH,
                           FEATURE_IMPORTANCE_PATH, TARGET_COL)
from src.logger import get_logger
from src.utils.serialization import save_json

logger = get_logger(__name__)


def top_features(n: int = 5) -> list[str]:
    if not FEATURE_IMPORTANCE_PATH.exists():
        return []
    imp = pd.read_csv(FEATURE_IMPORTANCE_PATH, index_col=0)
    return imp.head(n).index.tolist()


def detect_drift(features: pd.DataFrame, config: dict) -> dict:
    """Compare recent window vs the rest (training reference). Returns and
    persists a report: per-feature p-values, means, drift flags, overall status."""
    dcfg = config["drift_detection"]
    recent_days = dcfg.get("recent_window_days", 30)
    p_thresh = dcfg.get("p_value_threshold", 0.05)
    cols = top_features(dcfg.get("features_to_monitor", 5)) or ["lag_7", "rolling_mean_7"]
    cols = [c for c in cols if c in features.columns] + [TARGET_COL]

    cutoff = features[DATE_COL].max() - pd.Timedelta(days=recent_days)
    reference = features[features[DATE_COL] <= cutoff]
    current = features[features[DATE_COL] > cutoff]
    if len(current) < 10 or len(reference) < 10:
        logger.warning("Not enough data for drift detection")
        return {"status": "UNKNOWN", "features": []}

    results = []
    for c in cols:
        stat, p = ks_2samp(reference[c].dropna(), current[c].dropna())
        results.append({
            "feature": c,
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "reference_mean": round(float(reference[c].mean()), 3),
            "current_mean": round(float(current[c].mean()), 3),
            "drift": bool(p < p_thresh),
        })

    n_drifted = sum(r["drift"] for r in results)
    target_drift = next((r["drift"] for r in results if r["feature"] == TARGET_COL), False)
    status = ("HIGH" if target_drift and n_drifted >= 3
              else "MEDIUM" if n_drifted >= 2
              else "LOW")
    report = {
        "checked_at": pd.Timestamp.now().isoformat(),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "n_drifted": n_drifted,
        "status": status,
        "recommendation": {
            "LOW": "No action needed",
            "MEDIUM": "Monitor closely, consider retraining",
            "HIGH": "Retraining recommended",
        }[status],
        "features": results,
    }
    save_json(report, DRIFT_REPORT_PATH)
    logger.info("Drift check: %s (%d/%d features drifted)", status, n_drifted, len(cols))

    _evidently_report(reference, current, cols)
    return report


def _evidently_report(reference: pd.DataFrame, current: pd.DataFrame,
                      cols: list[str]) -> None:
    """Optional Evidently HTML report; skipped gracefully if unavailable."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        report = Report([DataDriftPreset()])
        snapshot = report.run(current[cols], reference[cols])
        snapshot.save_html(str(DRIFT_HTML_PATH))
        logger.info("Evidently report saved: %s", DRIFT_HTML_PATH)
    except Exception as e:
        try:  # evidently < 0.7 legacy API
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference[cols], current_data=current[cols])
            report.save_html(str(DRIFT_HTML_PATH))
            logger.info("Evidently (legacy) report saved: %s", DRIFT_HTML_PATH)
        except Exception:
            logger.warning("Evidently report skipped: %s", e)

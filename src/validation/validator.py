"""Data validation: schema check, quality metrics, outlier detection,
and a JSON validation report with an overall quality score."""
import numpy as np
import pandas as pd

from src.constants import DATE_COL, SERIES_COL, TARGET_COL, VALIDATION_REPORT_PATH
from src.logger import get_logger
from src.utils.serialization import save_json

logger = get_logger(__name__)

REQUIRED_COLUMNS = {DATE_COL: "datetime", SERIES_COL: "object", TARGET_COL: "numeric"}


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Return a list of schema errors (empty = valid)."""
    errors = []
    for col, kind in REQUIRED_COLUMNS.items():
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
            continue
        if kind == "datetime" and not pd.api.types.is_datetime64_any_dtype(df[col]):
            errors.append(f"Column {col} must be datetime, got {df[col].dtype}")
        if kind == "numeric" and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column {col} must be numeric, got {df[col].dtype}")
    return errors


def quality_metrics(df: pd.DataFrame) -> dict:
    """Missing values, duplicates, negatives, and date-gap statistics."""
    missing = (df.isna().mean() * 100).round(2).to_dict()
    dupes = int(df.duplicated(subset=[DATE_COL, SERIES_COL]).sum())
    negatives = int((df[TARGET_COL] < 0).sum()) if TARGET_COL in df else 0
    gaps = 0
    for _, g in df.groupby(SERIES_COL):
        d = g[DATE_COL].sort_values()
        expected = (d.max() - d.min()).days + 1
        gaps += expected - d.nunique()
    return {"missing_pct": missing, "duplicate_rows": dupes,
            "negative_sales": negatives, "date_gaps": gaps}


def detect_outliers(df: pd.DataFrame, method: str = "iqr", z_thresh: float = 4.0) -> pd.Series:
    """Boolean mask of outlier rows in the target, per series.

    IQR: outside [Q1 - 3*IQR, Q3 + 3*IQR] (3x — demand spikes are often real).
    Z-score: |z| > z_thresh.
    """
    mask = pd.Series(False, index=df.index)
    for _, g in df.groupby(SERIES_COL):
        x = g[TARGET_COL]
        if method == "iqr":
            q1, q3 = x.quantile(0.25), x.quantile(0.75)
            iqr = q3 - q1
            mask.loc[g.index] = (x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)
        else:
            z = (x - x.mean()) / (x.std() + 1e-9)
            mask.loc[g.index] = z.abs() > z_thresh
    return mask


def run_validation(df: pd.DataFrame) -> dict:
    """Full validation pipeline; writes and returns the report dict."""
    logger.info("Starting data validation on %d rows", len(df))
    schema_errors = validate_schema(df)
    quality = quality_metrics(df)
    outliers = detect_outliers(df)

    # Quality score: start at 100, subtract weighted penalties
    target_missing = quality["missing_pct"].get(TARGET_COL, 0)
    score = 100.0
    score -= min(30, target_missing * 3)
    score -= min(20, quality["duplicate_rows"] / max(len(df), 1) * 100 * 2)
    score -= min(20, quality["date_gaps"] / max(len(df), 1) * 100 * 2)
    score -= min(10, outliers.mean() * 100)
    score -= 30 if schema_errors else 0

    report = {
        "n_rows": len(df),
        "n_series": int(df[SERIES_COL].nunique()) if SERIES_COL in df else 0,
        "date_range": [str(df[DATE_COL].min()), str(df[DATE_COL].max())] if DATE_COL in df else None,
        "schema_errors": schema_errors,
        **quality,
        "outlier_count": int(outliers.sum()),
        "outlier_pct": round(float(outliers.mean() * 100), 2),
        "quality_score": round(max(score, 0), 1),
        "valid": not schema_errors,
    }
    save_json(report, VALIDATION_REPORT_PATH)
    if schema_errors:
        logger.error("Schema validation failed: %s", schema_errors)
    else:
        logger.info("Validation passed. Quality score: %.1f", report["quality_score"])
    return report

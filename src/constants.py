"""Global constants and canonical project paths."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"

CLEAN_DATA_PATH = PROCESSED_DIR / "clean.parquet"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
VALIDATION_REPORT_PATH = REPORTS_DIR / "validation_report.json"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
FORECAST_PATH = REPORTS_DIR / "forecast.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
DRIFT_REPORT_PATH = REPORTS_DIR / "drift_report.json"
DRIFT_HTML_PATH = REPORTS_DIR / "drift_report.html"
RETRAIN_LOG_PATH = REPORTS_DIR / "retraining_log.jsonl"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
BEST_MODEL_META_PATH = MODELS_DIR / "best_model_meta.json"

# Canonical long-format column names used across the whole pipeline
DATE_COL = "date"
SERIES_COL = "series_id"
TARGET_COL = "sales"

for _d in (RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

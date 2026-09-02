"""DemandIQ end-to-end pipeline:
load -> validate -> clean -> features -> train/CV/MLflow -> forecast -> drift.

Usage: python main.py [--config config.yaml]
"""
import argparse
import random
from pathlib import Path

import numpy as np

from src.api.forecast_api import generate_forecast
from src.data_loading.loader import load_data
from src.features.feature_engineering import build_features
from src.logger import get_logger
from src.monitoring.drift_detector import detect_drift
from src.preprocessing.preprocessor import clean
from src.training.train_pipeline import run_training
from src.utils.config_loader import load_config
from src.validation.validator import run_validation

logger = get_logger(__name__)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def main(config_path: str = "config.yaml") -> None:
    config = load_config(Path(config_path))
    set_seeds(config.get("random_seed", 42))

    logger.info("=== DemandIQ pipeline start ===")
    raw = load_data(config)
    report = run_validation(raw)
    if not report["valid"]:
        raise SystemExit(f"Data validation failed: {report['schema_errors']}")

    cleaned = clean(raw, config)
    features = build_features(cleaned, config)
    summary = run_training(features, config)
    generate_forecast(cleaned, config)
    if config["drift_detection"].get("enabled", True):
        detect_drift(features, config)

    logger.info("=== Pipeline done. Best model: %s (test MAPE %.2f%%) ===",
                summary["best_model"], summary["test_metrics"]["mape"])
    logger.info("Launch the dashboard: uvicorn dashboard.server:app --port 8000, "
                "then cd frontend && npm run dev")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DemandIQ pipeline")
    parser.add_argument("--config", default="config.yaml")
    main(parser.parse_args().config)

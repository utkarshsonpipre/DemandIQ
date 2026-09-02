"""Load and validate config.yaml. Loaded once, passed around as a dict."""
from pathlib import Path

import yaml

from src.constants import CONFIG_PATH

_REQUIRED_SECTIONS = ("data", "forecasting", "features", "models", "mlflow",
                      "drift_detection", "retraining")


def load_config(path: Path = CONFIG_PATH) -> dict:
    """
    Load YAML config and check required sections exist.

    Raises
    ------
    FileNotFoundError
        If the config file is missing.
    ValueError
        If a required top-level section is absent.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    missing = [s for s in _REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"config.yaml missing sections: {missing}")
    return cfg

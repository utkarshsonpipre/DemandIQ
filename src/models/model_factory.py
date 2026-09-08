"""Factory: instantiate all enabled models from config."""
from src.logger import get_logger
from src.models.base_model import BaseModel
from src.models.baseline_models import LinearRegressionModel, MovingAverageModel
from src.models.statistical_models import ARIMAModel, ExpSmoothingModel
from src.models.tree_models import LightGBMModel, RandomForestModel, XGBoostModel

logger = get_logger(__name__)


def create_models(config: dict) -> list[BaseModel]:
    """Build the enabled model roster. LSTM import is deferred so the rest of
    the platform works without torch installed."""
    m = config["models"]
    seed = config.get("random_seed", 42)
    horizon = config["forecasting"]["forecast_horizon"]
    models: list[BaseModel] = []

    if m.get("moving_average", {}).get("enabled", True):
        models.append(MovingAverageModel(m["moving_average"].get("windows")))
    if m.get("linear_regression", {}).get("enabled", True):
        models.append(LinearRegressionModel())
    for key, cls in (("lightgbm", LightGBMModel), ("xgboost", XGBoostModel),
                     ("random_forest", RandomForestModel)):
        cfg = m.get(key, {})
        if cfg.get("enabled", False):
            params = {k: v for k, v in cfg.items() if k != "enabled"}
            models.append(cls(params, seed=seed))
    if m.get("arima", {}).get("enabled", False):
        models.append(ARIMAModel(auto_order=m["arima"].get("auto_order", True),
                                 seasonal=m["arima"].get("seasonal", False)))
    if m.get("exponential_smoothing", {}).get("enabled", False):
        models.append(ExpSmoothingModel())
    if m.get("lstm", {}).get("enabled", False):
        try:
            from src.models.lstm_model import LSTMModel
            models.append(LSTMModel(m["lstm"], horizon=horizon, seed=seed))
        except ImportError as e:
            logger.warning("LSTM skipped (torch unavailable): %s", e)

    logger.info("Models enabled: %s", [x.name for x in models])
    return models

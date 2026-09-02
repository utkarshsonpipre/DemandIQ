"""MLflow experiment tracking + model registry operations.

All functions no-op gracefully when mlflow is disabled in config.
"""
import mlflow
from mlflow.tracking import MlflowClient

from src.logger import get_logger

logger = get_logger(__name__)


def setup(config: dict) -> bool:
    """Point mlflow at the configured backend and experiment. Returns enabled flag."""
    mcfg = config["mlflow"]
    if not mcfg.get("enabled", True):
        return False
    mlflow.set_tracking_uri(mcfg["tracking_uri"])
    mlflow.set_experiment(mcfg["experiment_name"])
    return True


def log_run(model_name: str, fold: int, params: dict, metrics: dict,
            tags: dict | None = None, artifacts: list | None = None) -> str:
    """Log one training run; returns the run id."""
    with mlflow.start_run(run_name=f"{model_name}_fold{fold}") as run:
        mlflow.log_params({"model_name": model_name, "cv_fold": fold, **params})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        for k, v in (tags or {}).items():
            mlflow.set_tag(k, v)
        for path in artifacts or []:
            mlflow.log_artifact(str(path))
        return run.info.run_id


def register_best(config: dict, model_path, model_name: str, avg_mape: float,
                  test_mape: float) -> dict:
    """Register the winning model. Promote to Production when it beats the
    current production model's test MAPE by more than the configured margin
    (or when no production model exists); otherwise leave as Staging.
    Previous production versions get archived on promotion."""
    mcfg = config["mlflow"]
    reg_name = mcfg["registry_model_name"]
    threshold = mcfg.get("promotion_improvement_threshold", 0.02)
    client = MlflowClient()

    with mlflow.start_run(run_name=f"register_{model_name}"):
        mlflow.log_params({"model_name": model_name})
        mlflow.log_metrics({"avg_cv_mape": avg_mape, "test_mape": test_mape})
        mlflow.log_artifact(str(model_path))
        run_id = mlflow.active_run().info.run_id

    mv = client.create_model_version(
        name=reg_name, source=f"runs:/{run_id}/artifacts",
        run_id=run_id, tags={"model_type": model_name, "test_mape": f"{test_mape:.4f}"},
    ) if _ensure_registered(client, reg_name) else None

    prod = [v for v in client.search_model_versions(f"name='{reg_name}'")
            if v.current_stage == "Production" and v.version != mv.version]
    prod_mape = min((float(v.tags.get("test_mape", "inf")) for v in prod), default=float("inf"))

    promote = test_mape < prod_mape * (1 - threshold) or not prod
    stage = "Production" if promote else "Staging"
    client.transition_model_version_stage(reg_name, mv.version, stage,
                                          archive_existing_versions=promote)
    logger.info("Registered %s v%s -> %s (test MAPE %.2f%% vs prod %.2f%%)",
                reg_name, mv.version, stage, test_mape, prod_mape)
    return {"version": mv.version, "stage": stage, "run_id": run_id,
            "previous_prod_mape": None if prod_mape == float("inf") else prod_mape}


def _ensure_registered(client: MlflowClient, name: str) -> bool:
    try:
        client.get_registered_model(name)
    except Exception:
        client.create_registered_model(name)
    return True


def get_model_versions(config: dict) -> list[dict]:
    """All registry versions for the dashboard."""
    try:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        client = MlflowClient()
        versions = client.search_model_versions(
            f"name='{config['mlflow']['registry_model_name']}'")
        return [{"version": v.version, "stage": v.current_stage,
                 "model_type": v.tags.get("model_type", "?"),
                 "test_mape": v.tags.get("test_mape", "?"),
                 "created": v.creation_timestamp} for v in versions]
    except Exception as e:
        logger.warning("Registry query failed: %s", e)
        return []

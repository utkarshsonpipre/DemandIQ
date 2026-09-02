"""FastAPI backend for the React dashboard. Local dev API only — reads the
pipeline's artifact files and exposes the retrain/drift actions. Not a model
serving layer (see project scope in ARCHITECTURE.md).

Run: uvicorn dashboard.server:app --reload --port 8000
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src import constants as C
from src.mlflow_utils.tracker import get_model_versions
from src.monitoring.performance_monitor import check_performance
from src.retraining.orchestrator import check_triggers, read_retrain_log, retrain
from src.utils.config_loader import load_config
from src.utils.serialization import load_json

app = FastAPI(title="DemandIQ API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


def _json(path_name: str):
    path = getattr(C, path_name)
    return load_json(path) if path.exists() else None


def _csv(path_name: str) -> pd.DataFrame | None:
    path = getattr(C, path_name)
    return pd.read_csv(path) if path.exists() else None


def _clean_nan(obj):
    """Recursively replace float NaN with None — JSON has no NaN literal and
    Starlette's encoder rejects it outright."""
    if isinstance(obj, float) and pd.isna(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan(v) for v in obj]
    return obj


def _require(df, name: str):
    if df is None:
        raise HTTPException(404, f"{name} not found — run the pipeline first")
    return df


@app.get("/api/status")
def status():
    meta = _json("BEST_MODEL_META_PATH")
    drift = _json("DRIFT_REPORT_PATH")
    clean = pd.read_parquet(C.CLEAN_DATA_PATH) if C.CLEAN_DATA_PATH.exists() else None
    return {
        "has_pipeline_run": clean is not None,
        "best_model": meta["best_model"] if meta else None,
        "test_mape": meta["test_metrics"]["mape"] if meta else None,
        "avg_cv_mape": meta["avg_cv_mape"] if meta else None,
        "trained_at": meta["trained_at"] if meta else None,
        "n_features": meta["n_features"] if meta else None,
        "drift_status": drift["status"] if drift else None,
        "n_series": int(clean["series_id"].nunique()) if clean is not None else None,
    }


@app.get("/api/data-overview")
def data_overview():
    clean = _require(pd.read_parquet(C.CLEAN_DATA_PATH) if C.CLEAN_DATA_PATH.exists() else None,
                     "clean data")
    report = _json("VALIDATION_REPORT_PATH")
    missing = (clean.isna().mean() * 100).round(2)
    return {
        "n_rows": len(clean),
        "n_series": int(clean["series_id"].nunique()),
        "date_min": str(clean["date"].min().date()),
        "date_max": str(clean["date"].max().date()),
        "missing_pct": {k: float(v) for k, v in missing.items() if v > 0},
        "validation_report": report,
        "sample": _clean_nan(clean.head(50).assign(date=lambda d: d["date"].astype(str))
                            .to_dict("records")),
    }


@app.get("/api/series")
def list_series():
    clean = _require(pd.read_parquet(C.CLEAN_DATA_PATH) if C.CLEAN_DATA_PATH.exists() else None,
                     "clean data")
    return sorted(clean["series_id"].unique().tolist())


@app.get("/api/eda/{series_id}")
def eda(series_id: str):
    clean = _require(pd.read_parquet(C.CLEAN_DATA_PATH) if C.CLEAN_DATA_PATH.exists() else None,
                     "clean data")
    g = clean[clean["series_id"] == series_id].sort_values("date")
    if g.empty:
        raise HTTPException(404, f"series {series_id} not found")

    history = [{"date": str(d.date()), "sales": float(s)}
              for d, s in zip(g["date"], g["sales"])]

    decomposition = None
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        dec = seasonal_decompose(g.set_index("date")["sales"], period=7, model="additive")
        decomposition = {
            name: [{"date": str(d.date()), "value": None if pd.isna(v) else float(v)}
                  for d, v in series.items()]
            for name, series in (("trend", dec.trend), ("seasonal", dec.seasonal),
                                 ("residual", dec.resid))
        }
    except Exception:
        pass

    acf_pacf = None
    try:
        from statsmodels.tsa.stattools import acf, pacf
        nlags = min(40, len(g) // 3)
        acf_pacf = {"acf": acf(g["sales"], nlags=nlags).tolist(),
                    "pacf": pacf(g["sales"], nlags=nlags).tolist()}
    except Exception:
        pass

    return {"history": history, "decomposition": decomposition, "acf_pacf": acf_pacf}


@app.get("/api/model-comparison")
def model_comparison():
    comp = _require(_csv("COMPARISON_PATH"), "model comparison")
    metric_cols = ["mape", "rmse", "mae", "r2_score", "smape",
                  "training_time_seconds", "inference_time_ms"]
    avg = comp.groupby("model")[metric_cols].mean().sort_values("mape")
    avg["rank"] = range(1, len(avg) + 1)
    return {"average": avg.reset_index().to_dict("records"),
            "per_fold": comp.to_dict("records")}


@app.get("/api/forecast")
def forecast():
    fc = _require(_csv("FORECAST_PATH"), "forecast")
    clean = pd.read_parquet(C.CLEAN_DATA_PATH)
    imp = _csv("FEATURE_IMPORTANCE_PATH")
    meta = _json("BEST_MODEL_META_PATH")

    def series_payload(sid: str):
        hist = clean[clean["series_id"] == sid].sort_values("date").tail(120)
        fut = fc[fc["series_id"] == sid].sort_values("date")
        resid_std = clean[clean["series_id"] == sid].sort_values("date")["sales"].tail(90).diff().std()
        band = 1.96 * (resid_std if pd.notna(resid_std) else 0)
        return {
            "history": [{"date": str(d.date()), "sales": float(s)}
                       for d, s in zip(hist["date"], hist["sales"])],
            "forecast": [{"date": str(pd.Timestamp(d).date()), "forecast": round(float(f), 2),
                         "lower": round(max(float(f) - band, 0), 2),
                         "upper": round(float(f) + band, 2)}
                        for d, f in zip(fut["date"], fut["forecast"])],
        }

    return {
        "series": sorted(fc["series_id"].unique().tolist()),
        "default_series": series_payload(sorted(fc["series_id"].unique())[0]),
        "test_metrics": meta["test_metrics"] if meta else None,
        "best_model": meta["best_model"] if meta else None,
        "feature_importance": (imp.head(15).rename(columns={imp.columns[0]: "feature"})
                               .to_dict("records") if imp is not None else []),
    }


@app.get("/api/forecast/{series_id}")
def forecast_series(series_id: str):
    fc = _require(_csv("FORECAST_PATH"), "forecast")
    clean = pd.read_parquet(C.CLEAN_DATA_PATH)
    if series_id not in fc["series_id"].unique():
        raise HTTPException(404, f"series {series_id} not found in forecast")
    hist = clean[clean["series_id"] == series_id].sort_values("date").tail(120)
    fut = fc[fc["series_id"] == series_id].sort_values("date")
    resid_std = clean[clean["series_id"] == series_id].sort_values("date")["sales"].tail(90).diff().std()
    band = 1.96 * (resid_std if pd.notna(resid_std) else 0)
    return {
        "history": [{"date": str(d.date()), "sales": float(s)}
                   for d, s in zip(hist["date"], hist["sales"])],
        "forecast": [{"date": str(pd.Timestamp(d).date()), "forecast": round(float(f), 2),
                     "lower": round(max(float(f) - band, 0), 2),
                     "upper": round(float(f) + band, 2)}
                    for d, f in zip(fut["date"], fut["forecast"])],
    }


@app.get("/api/mlflow")
def mlflow_info():
    config = load_config()
    meta = _json("BEST_MODEL_META_PATH")
    versions = get_model_versions(config)
    runs = []
    try:
        import mlflow
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        exp = mlflow.get_experiment_by_name(config["mlflow"]["experiment_name"])
        if exp:
            df = mlflow.search_runs([exp.experiment_id], max_results=100)
            keep = [c for c in df.columns if c in ("start_time", "status")
                    or c.startswith(("metrics.mape", "metrics.rmse", "metrics.mae",
                                     "params.model_name", "params.cv_fold"))]
            df = df[keep].rename(columns=lambda c: c.replace("metrics.", "").replace("params.", ""))
            df["start_time"] = df["start_time"].astype(str)
            runs = _clean_nan(df.to_dict("records"))
    except Exception:
        pass
    return _clean_nan({"best_run": meta, "versions": versions, "runs": runs})


@app.get("/api/drift")
def drift():
    report = _json("DRIFT_REPORT_PATH")
    if report is None:
        return {"status": None, "features": []}
    return report


@app.get("/api/drift/html", response_class=HTMLResponse)
def drift_html():
    if not C.DRIFT_HTML_PATH.exists():
        raise HTTPException(404, "no evidently report")
    return C.DRIFT_HTML_PATH.read_text(encoding="utf-8")


@app.get("/api/retrain-log")
def retrain_log():
    return read_retrain_log()


@app.post("/api/retrain")
def do_retrain():
    config = load_config()
    try:
        return retrain(config, trigger="manual (react dashboard)")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/drift-check")
def do_drift_check():
    from src.monitoring.drift_detector import detect_drift
    config = load_config()
    if not C.FEATURES_PATH.exists():
        raise HTTPException(404, "no features — run the pipeline first")
    features = pd.read_parquet(C.FEATURES_PATH)
    return detect_drift(features, config)


@app.get("/api/performance-check")
def performance_check():
    config = load_config()
    if not C.FEATURES_PATH.exists():
        raise HTTPException(404, "no features — run the pipeline first")
    features = pd.read_parquet(C.FEATURES_PATH)
    return check_performance(features, config)


@app.get("/api/triggers")
def triggers():
    config = load_config()
    if not C.FEATURES_PATH.exists():
        raise HTTPException(404, "no features — run the pipeline first")
    features = pd.read_parquet(C.FEATURES_PATH)
    return check_triggers(features, config)


@app.post("/api/upload")
async def upload(file: UploadFile):
    dest = C.RAW_DIR / "uploaded.csv"
    dest.write_bytes(await file.read())
    return {"path": str(dest), "message": "Set data.source='csv' and data.csv_path in "
                                          "config.yaml to this path, then retrain."}

# DemandIQ — Automated Demand Forecasting & ML Experimentation Platform

An internal-ML-platform-style system covering the full forecasting lifecycle:
data validation → feature engineering → multi-model training with time-series
cross-validation → MLflow tracking & registry → drift detection → automated
retraining → interactive dashboard.

## Quick Start

```bash
pip install -r requirements.txt

# 1. Run the full pipeline (downloads M5 from Kaggle if credentials exist,
#    otherwise auto-generates realistic synthetic demand data)
python main.py

# 2. Launch the dashboard API (FastAPI, local dev only — reads pipeline artifacts,
#    exposes retrain/drift actions; not a model-serving layer)
uvicorn dashboard.server:app --port 8000

# 3. Launch the React dashboard (separate terminal)
cd frontend && npm install && npm run dev   # http://localhost:5173

# 4. (optional) MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The React app under `frontend/` is the dashboard.

Kaggle setup (optional): put `kaggle.json` in `~/.kaggle/` and accept the
[M5 competition rules](https://www.kaggle.com/c/m5-forecasting-accuracy). Without
credentials the pipeline falls back to synthetic data so everything still works.

## Key Features

- **Walk-forward time-series CV** — expanding train windows, sliding validation, untouched final test window
- **7 models on identical folds** — Moving Average, Linear Regression, LightGBM, XGBoost, Random Forest, ARIMA (ADF + AIC order search), PyTorch seq2seq LSTM
- **5 metrics** — MAPE (primary), RMSE (tiebreak), MAE, R², SMAPE
- **MLflow** — every fold logged (params/metrics/timing); best model registered, promoted to Production only when it beats the incumbent by >2% test MAPE
- **Drift detection** — KS test on the top-5 most important features + target; optional Evidently HTML report
- **Performance monitoring** — recent-window MAPE vs training baseline, alert at >15% degradation
- **Automated retraining** — manual dashboard trigger or automatic (drift on 3+ features / MAPE degradation), with JSONL event log
- **Leakage-safe features** — ~40 features (temporal, lag, rolling, decomposition, business, hierarchical), all target-derived features use strictly-past values

## Project Structure

```
config.yaml              # every knob: data, features, models, thresholds
main.py                  # end-to-end pipeline entrypoint
src/
  data_loading/          # Kaggle M5 downloader, long-format loaders, synthetic fallback
  validation/            # schema, quality, outliers, quality score report
  preprocessing/         # frequency detection, gap reindexing, imputation
  features/              # feature engineering pipeline (leakage-safe)
  models/                # BaseModel + baseline / tree / statistical / LSTM
  training/              # walk-forward CV + training orchestrator
  evaluation/            # MAPE, SMAPE, RMSE, MAE, R²
  mlflow_utils/          # tracking + registry (staging/production promotion)
  monitoring/            # KS drift detection + performance degradation
  retraining/            # trigger checks + retraining pipeline + event log
  api/                   # forecast generation (recursive / direct multi-step)
dashboard/server.py      # FastAPI dev API — reads artifacts, exposes retrain/drift actions
frontend/                # React dashboard (Vite, Recharts)
tests/                   # pytest: metrics, features (leakage), CV, validation
notebooks/               # exploration / training / evaluation walkthroughs
```

## Configuration

Everything is driven by `config.yaml`: data source, series count, forecast
horizon, CV folds, per-model hyperparameters, drift thresholds, retraining
triggers. To use your own data:

```yaml
data:
  source: 'csv'
  csv_path: 'data/raw/my_sales.csv'   # columns: date, sales (+ optional ids)
```

## Testing

```bash
python -m pytest tests -q
```

## Troubleshooting

| Problem | Fix |
|---|---|
| Kaggle 403 / no credentials | Pipeline auto-falls back to synthetic data; add `~/.kaggle/kaggle.json` + accept M5 rules for the real dataset |
| Training too slow | Lower `data.n_series`, `models.*.n_estimators`, or `models.lstm.epochs` |
| Dashboard shows "No pipeline artifacts" | Run `python main.py` first |
| Evidently HTML missing | KS-test drift core still runs; Evidently report is optional |

See [ARCHITECTURE.md](ARCHITECTURE.md) for design decisions.

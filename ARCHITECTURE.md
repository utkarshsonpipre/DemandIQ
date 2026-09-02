# DemandIQ Architecture

## Pipeline flow

```
load (Kaggle M5 / CSV / synthetic)
  → validate (schema, quality, outliers → quality score)
  → clean (frequency detection, gap reindex, imputation)
  → features (~40 leakage-safe features → features.parquet)
  → train (7 models × 3 walk-forward folds → MLflow)
  → select (lowest mean MAPE, RMSE tiebreak)
  → test (refit on train+val, evaluate once on held-out last 10%)
  → register (MLflow registry; promote if >2% better than production)
  → forecast (next 30 periods per series)
  → monitor (KS drift + MAPE degradation → retraining triggers)
```

## Canonical data format

Everything downstream of the loader consumes one long format:
`date, series_id, item_id, store_id, sales, sell_price, promotion_flag`.
New data sources only need a loader that emits this shape.

## Key design decisions

| Decision | Rationale |
|---|---|
| Walk-forward CV, expanding windows | Shuffled CV leaks the future; expanding windows mirror how a production model actually retrains over time. Test window (last 10%) is excluded from every fold. |
| MAPE primary, RMSE tiebreak | Industry standard for demand; interpretable across series of different scale. Zero-sales rows are skipped in MAPE (SMAPE reported for that regime). |
| One global model across series (trees) | Cross-series learning + one artifact to manage; `series/store/item` encodings + per-series lag features give the model per-series context. |
| Per-series ARIMA / scaling for LSTM | Statistical models and NN scaling are scale-sensitive; each series gets its own ARIMA fit and min-max range. |
| Leakage-safe features everywhere | Every target-derived feature (`lag_*`, `rolling_*`, decomposition, hierarchy averages) is computed on values strictly before the row (`shift(1)` before rolling/expanding). A unit test asserts it. |
| Evaluation protocol | Feature models are scored one-step-ahead on validation rows; ARIMA/LSTM produce true multi-step forecasts over the same rows. This slightly favors feature models and mirrors the trade-off you'd see in production (documented, not hidden). |
| ADF + small AIC grid instead of pmdarima | ADF picks d; 4 candidate orders by AIC. pmdarima adds a fragile dependency for marginal gain at this scale. |
| Direct multi-horizon LSTM (60 in → 30 out) | One forward pass yields the whole horizon; no compounding recursion error inside the horizon. Rolls forward in 30-step chunks for longer windows. |
| Recursive forecasting for tree models | Trees predict one step; the forecaster appends each prediction and regenerates features so day t+2 sees day t+1's forecast as `lag_1`. |
| KS-test drift core, Evidently optional | scipy KS is stable across environments and versions; Evidently renders the pretty HTML report when importable (both its ≥0.7 and legacy APIs are tried). |
| Registry promotion gate | New model must beat the current Production version's **test** MAPE by >2% — avoids churn from noise-level improvements. Losing candidates park in Staging. |
| Top-N series subset of M5 | Full M5 melt is ~58M rows; top-N by volume keeps the full lifecycle demonstrable in <15 min. `data.n_series` scales it up. |
| Synthetic fallback | Trend + weekly/yearly seasonality + promos + noise. The platform is demonstrable with zero external credentials. |
| SQLite MLflow backend | Zero-infra local tracking; swap `mlflow.tracking_uri` for a server later. |

## Retraining state machine

```
manual button ─────────────┐
drift ≥ 3 top features ────┤→ retrain() → full pipeline rerun on all data
recent MAPE +15% vs base ──┘      → register if better → JSONL event log
```

## Artifacts contract (dashboard reads these)

| File | Producer |
|---|---|
| `data/processed/clean.parquet` | preprocessing |
| `data/processed/features.parquet` | features |
| `reports/validation_report.json` | validation |
| `reports/model_comparison.csv` | training (per-fold metrics) |
| `reports/feature_importance.csv` | training (best model) |
| `reports/forecast.csv` | forecast API |
| `reports/drift_report.json` / `.html` | monitoring |
| `reports/retraining_log.jsonl` | retraining |
| `models/best_model.pkl` + `best_model_meta.json` | training |
| `mlflow.db` + registry | MLflow |

The dashboard is a pure reader of this contract (plus two action buttons that
invoke `retrain()` and `detect_drift()`), so pipeline and UI stay decoupled.

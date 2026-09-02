// Plain-language explanations shown in the (i) tooltip beside every metric label.
export const METRIC_INFO = {
  mape: "Mean Absolute Percentage Error — average forecast error as a percent of actual sales. Lower is better. This is the main number used to pick the best model.",
  smape: "Symmetric MAPE — same idea as MAPE, but fairer when actual sales are close to zero. Lower is better.",
  rmse: "Root Mean Squared Error — average error size in sales units, with big misses counted extra. Lower is better.",
  mae: "Mean Absolute Error — average error size in raw sales units, every miss weighted the same. Lower is better.",
  r2_score: "R-squared — how much of the ups and downs in sales the model explains, from 0 (none) to 1 (perfect). Higher is better.",
  training_time_seconds: "How long the model took to train, in seconds.",
  inference_time_ms: "How long the model takes to produce one prediction, in milliseconds.",
  avg_cv_mape: "Average MAPE across all cross-validation folds — accuracy during training, before touching the final held-out test set.",
  test_mape: "MAPE measured once on the untouched final 10% of data — the most honest estimate of real-world accuracy.",
  quality_score: "Overall data health score (0-100), combining missing values, duplicates, date gaps and outliers. Higher is better.",
  ks_statistic: "Kolmogorov-Smirnov statistic — how different the recent data's distribution is from training data, from 0 (identical) to 1 (completely different).",
  p_value: "Statistical significance of the drift test. Below 0.05 means the difference is unlikely to be random chance — the feature has likely drifted.",
  drift_status: "Overall verdict: LOW (no action needed), MEDIUM (keep an eye on it), HIGH (retraining recommended).",
  feature_importance: "How much each input feature influenced the model's predictions, relative to the others.",
  confidence_band: "The range we're 95% confident the actual value will land in, based on how much the recent forecast has varied.",
  rank: "Model's position when every model is sorted by MAPE, lowest (best) first.",
  outlier_pct: "Share of rows flagged as unusually high or low sales (3x the interquartile range) compared to the rest of that series.",
};

export const METRIC_LABEL = {
  mape: "MAPE", smape: "SMAPE", rmse: "RMSE", mae: "MAE", r2_score: "R²",
  training_time_seconds: "Training time (s)", inference_time_ms: "Inference (ms)",
  avg_cv_mape: "Avg CV MAPE", test_mape: "Test MAPE", quality_score: "Quality score",
  ks_statistic: "KS statistic", p_value: "p-value", drift_status: "Drift status",
  rank: "Rank", outlier_pct: "Outliers",
};

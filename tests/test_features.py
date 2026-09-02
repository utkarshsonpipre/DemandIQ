import numpy as np
import pandas as pd

from src.features.feature_engineering import build_features, feature_columns

CFG = {
    "features": {"lag_lookbacks": [1, 7], "rolling_windows": [7],
                 "include_temporal": True, "include_decomposition": True,
                 "include_business_features": True, "include_hierarchical": True,
                 "holiday_country": "US"},
}


def _df(n=100):
    dates = pd.date_range("2023-01-01", periods=n)
    return pd.DataFrame({
        "date": list(dates) * 2,
        "series_id": ["A"] * n + ["B"] * n,
        "item_id": ["A"] * n + ["B"] * n,
        "store_id": ["S1"] * (2 * n),
        "sales": list(np.arange(n, dtype=float)) * 2,
    })


def test_lag_is_exactly_previous_value():
    feats = build_features(_df(), CFG, save=False)
    a = feats[feats["series_id"] == "A"].sort_values("date")
    assert (a["lag_1"].to_numpy()[1:] == a["sales"].to_numpy()[:-1]).all()


def test_rolling_mean_excludes_current_row():
    feats = build_features(_df(), CFG, save=False)
    a = feats[feats["series_id"] == "A"].sort_values("date").iloc[-1]
    # sales = 0..99; rolling_mean_7 at t=99 covers 92..98, mean 95
    assert a["rolling_mean_7"] == 95


def test_no_nans_in_feature_matrix():
    feats = build_features(_df(), CFG, save=False)
    assert not feats[feature_columns(feats)].isna().any().any()

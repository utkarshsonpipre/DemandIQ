import pandas as pd

from src.validation.validator import detect_outliers, validate_schema


def _df():
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=50),
        "series_id": "A",
        "sales": [100.0] * 49 + [10000.0],  # one wild spike
    })


def test_schema_valid():
    assert validate_schema(_df()) == []


def test_schema_catches_missing_and_wrong_type():
    bad = _df().rename(columns={"sales": "qty"})
    assert any("sales" in e for e in validate_schema(bad))
    bad2 = _df()
    bad2["date"] = bad2["date"].astype(str)
    assert any("datetime" in e for e in validate_schema(bad2))


def test_outlier_detected():
    mask = detect_outliers(_df())
    assert mask.sum() == 1 and mask.iloc[-1]

import numpy as np

from src.evaluation.metrics import evaluate, mape, smape


def test_mape_skips_zeros():
    y = np.array([0, 100, 200])
    p = np.array([50, 110, 190])
    assert abs(mape(y, p) - 0.075) < 1e-9  # only nonzero rows counted


def test_smape_zero_both():
    assert smape(np.array([0, 0]), np.array([0, 0])) == 0.0


def test_evaluate_keys_and_perfect_fit():
    y = np.array([10.0, 20.0, 30.0])
    m = evaluate(y, y)
    assert set(m) == {"mae", "rmse", "r2_score", "mape", "smape"}
    assert m["mape"] == 0 and m["rmse"] == 0 and m["r2_score"] == 1

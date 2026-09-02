import pandas as pd

from src.training.cross_validator import make_folds


def _df(n=200):
    return pd.DataFrame({"date": pd.date_range("2023-01-01", periods=n),
                         "sales": range(n)})


def test_walk_forward_order_and_no_overlap():
    df = _df()
    folds, test_mask = make_folds(df, n_folds=3)
    assert len(folds) == 3
    for f in folds:
        train_max = df.loc[f.train_mask, "date"].max()
        val_min = df.loc[f.val_mask, "date"].min()
        assert train_max < val_min                      # temporal order
        assert not (f.train_mask & f.val_mask).any()    # disjoint
        assert not (f.val_mask & test_mask).any()       # val never touches test
    # expanding: later folds have strictly more training data
    sizes = [f.train_mask.sum() for f in folds]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


def test_test_window_is_last_10pct():
    df = _df()
    _, test_mask = make_folds(df, 3)
    assert 15 <= test_mask.sum() <= 25
    assert df.loc[test_mask, "date"].min() > df.loc[~test_mask, "date"].max()

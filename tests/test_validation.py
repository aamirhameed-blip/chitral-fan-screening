"""Unit tests for the validation and statistics helpers. Run with: pytest"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fanscreen.stats import benjamini_hochberg, morans_i
from fanscreen.validation import (
    block_ids,
    bootstrap_index,
    classification_metrics,
    out_of_fold,
    partition,
    spatial_cv,
    summarise,
)


@pytest.fixture
def sample():
    """60 catchments on a 6 x 10 grid with 12 km spacing; the target follows x."""
    rng = np.random.default_rng(0)
    xx, yy = np.meshgrid(np.arange(6) * 12000.0, np.arange(10) * 12000.0)
    df = pd.DataFrame({"cx": xx.ravel(), "cy": yy.ravel()})
    df["x"] = df.cx / 1000 + rng.normal(0, 10, len(df))
    y = (df.cx > df.cx.median()).astype(int).values
    return df, y


def test_block_ids_group_by_square(sample):
    df, _ = sample
    ids = block_ids(df, 30)
    assert len(np.unique(ids)) == 3 * 4  # x from 0 to 60 km -> 3 blocks, y from 0 to 108 km -> 4 blocks
    assert ids[0] == ids[1] == ids[2]  # 0, 12 and 24 km share the first block


def test_partition_keeps_blocks_whole(sample):
    df, _ = sample
    groups = block_ids(df, 30)
    folds = partition(groups, seed=1, k=4)
    assert sorted(np.concatenate([test for _, test in folds])) == list(range(len(df)))
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])


def test_partition_is_reproducible(sample):
    groups = block_ids(sample[0], 30)
    a, b = partition(groups, seed=3, k=4), partition(groups, seed=3, k=4)
    assert all(np.array_equal(ta, tb) for (_, ta), (_, tb) in zip(a, b))


def test_out_of_fold_uses_training_prevalence(sample):
    df, y = sample
    folds = partition(block_ids(df, 30), seed=0, k=4)
    prob, thr = out_of_fold(LogisticRegression(), df[["x"]].values, y, folds)
    assert np.isfinite(prob).all()
    for train, test in folds:
        assert np.allclose(thr[test], y[train].mean())


def test_metrics_for_perfect_ranking():
    y = np.array([0, 0, 0, 1, 1, 1])
    m = classification_metrics(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]), np.full(6, 0.5))
    assert m["auc"] == 1 and m["bal_acc"] == 1 and m["mcc"] == 1


def test_bootstrap_draws_whole_blocks():
    groups = np.repeat([1, 2, 3], [2, 3, 4])
    idx = bootstrap_index(groups, np.random.default_rng(0))
    counts = pd.Series(groups[idx]).value_counts()
    sizes = {1: 2, 2: 3, 3: 4}
    assert all(n % sizes[b] == 0 for b, n in counts.items())


def test_summary_and_paired_difference(sample):
    df, y = sample
    P, T, groups = spatial_cv(LogisticRegression(), df, ["x"], y, km=30, n_repeats=3, n_jobs=1)
    s = summarise("LR", P, T, y, groups, baseline=P, n_boot=200)
    assert P.shape == (3, len(y))
    assert s["auc_lo"] <= s["auc"] <= s["auc_hi"]
    assert s["dauc"] == 0  # a model compared with itself gains nothing


def test_morans_i_detects_clustering():
    xy = np.c_[np.arange(40) * 1000.0, np.zeros(40)]
    clustered = (np.arange(40) >= 20).astype(float)
    alternating = (np.arange(40) % 2).astype(float)
    assert morans_i(clustered, xy, band=1500) > 0.8
    assert morans_i(alternating, xy, band=1500) < -0.8


def test_benjamini_hochberg_matches_reference():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    expected = np.array([0.04, 0.0533333, 0.0533333, 0.20])  # p * m / rank, made monotone
    assert np.allclose(benjamini_hochberg(p), expected, atol=1e-6)

"""Spatial-block cross-validation, skill metrics and block-bootstrap summaries.

Catchments are grouped into square blocks of side ``km`` by their centroids. Each
repeat assigns the blocks to folds in a different random order, so neighbouring
catchments never sit on both sides of a train/test split. Every random step takes an
explicit seed, which makes the results reproducible on any machine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupKFold

from .config import BLOCK_KM, N_BOOT, N_FOLDS, N_JOBS, N_REPEATS


def block_ids(df: pd.DataFrame, km: float) -> np.ndarray:
    """Integer id of the ``km`` x ``km`` block that holds each catchment centroid (``cx``, ``cy`` in metres)."""
    size = km * 1000
    return (np.floor(df["cx"] / size) * 100000 + np.floor(df["cy"] / size)).astype(int).values


def partition(groups: np.ndarray, seed: int, k: int = N_FOLDS) -> list:
    """Split the samples into ``k`` folds of whole blocks, with the block order shuffled by ``seed``."""
    order = np.random.default_rng(seed).permutation(np.unique(groups))
    rank = {b: i for i, b in enumerate(order)}
    return list(GroupKFold(k).split(np.zeros(len(groups)), groups=np.array([rank[b] for b in groups])))


def out_of_fold(model, X: np.ndarray, y: np.ndarray, folds: list) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold probabilities, and for each sample the training prevalence used as its decision threshold."""
    prob, threshold = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for train, test in folds:
        fitted = clone(model).fit(X[train], y[train])
        prob[test] = fitted.predict_proba(X[test])[:, 1]
        threshold[test] = y[train].mean()
    return prob, threshold


def _one_repeat(model, X, y, groups, seed):
    return out_of_fold(model, X, y, partition(groups, seed))


def spatial_cv(
    model,
    df: pd.DataFrame,
    features: list,
    y: np.ndarray,
    km: float = BLOCK_KM,
    n_repeats: int = N_REPEATS,
    n_jobs: int = N_JOBS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Repeated spatial-block cross-validation.

    Returns the out-of-fold probabilities and thresholds, both of shape
    (n_repeats, n_samples), and the block id of each sample.
    """
    X = df[features].values
    groups = block_ids(df, km)
    runs = Parallel(n_jobs=n_jobs)(delayed(_one_repeat)(model, X, y, groups, s) for s in range(n_repeats))
    return np.array([p for p, _ in runs]), np.array([t for _, t in runs]), groups


def classification_metrics(y: np.ndarray, prob: np.ndarray, threshold: np.ndarray) -> dict:
    """Discrimination, calibration and thresholded skill of one set of out-of-fold predictions."""
    hit = (prob >= threshold).astype(int)
    tp, tn = ((hit == 1) & (y == 1)).sum(), ((hit == 0) & (y == 0)).sum()
    sens, spec = tp / max(y.sum(), 1), tn / max((1 - y).sum(), 1)
    clipped = np.clip(prob, 1e-4, 1 - 1e-4)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    recal = LogisticRegression(C=1e6, max_iter=2000).fit(logit, y)
    return dict(
        auc=roc_auc_score(y, prob),
        ap=average_precision_score(y, prob),
        brier=brier_score_loss(y, prob),
        bal_acc=(sens + spec) / 2,
        sens=sens,
        spec=spec,
        mcc=matthews_corrcoef(y, hit),
        cal_slope=float(recal.coef_[0, 0]),
        cal_int=float(recal.intercept_[0]),
    )


def bootstrap_index(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample indices of one block bootstrap: whole blocks drawn with replacement."""
    blocks = np.unique(groups)
    picked = rng.choice(blocks, len(blocks), replace=True)
    return np.concatenate([np.flatnonzero(groups == b) for b in picked])


def summarise(
    name: str,
    P: np.ndarray,
    T: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    baseline: np.ndarray | None = None,
    n_boot: int = N_BOOT,
    seed: int = 7,
) -> dict:
    """Mean skill over repeats, with block-bootstrap 95% intervals.

    Each bootstrap draw takes one random repeat and one block resample. With a
    ``baseline`` (out-of-fold probabilities of the reference model, same shape as
    ``P``) the paired AUC difference is also returned, with the share of draws in
    which the model does not beat the baseline (``dauc_p_le0``).
    """
    per_repeat = pd.DataFrame([classification_metrics(y, P[r], T[r]) for r in range(len(P))])
    rng = np.random.default_rng(seed)
    auc, brier, dauc = [], [], []
    for _ in range(n_boot):
        r = rng.integers(len(P))
        i = bootstrap_index(groups, rng)
        if len(np.unique(y[i])) < 2:
            continue
        a = roc_auc_score(y[i], P[r][i])
        auc.append(a)
        brier.append(brier_score_loss(y[i], P[r][i]))
        if baseline is not None:
            dauc.append(a - roc_auc_score(y[i], baseline[r][i]))
    out = dict(
        name=name,
        **per_repeat.mean().round(4).to_dict(),
        auc_partition_sd=round(per_repeat.auc.std(), 4),
        auc_lo=round(np.percentile(auc, 2.5), 3),
        auc_hi=round(np.percentile(auc, 97.5), 3),
        brier_lo=round(np.percentile(brier, 2.5), 3),
        brier_hi=round(np.percentile(brier, 97.5), 3),
    )
    if baseline is not None:
        out.update(
            dauc=round(float(np.mean(dauc)), 3),
            dauc_lo=round(np.percentile(dauc, 2.5), 3),
            dauc_hi=round(np.percentile(dauc, 97.5), 3),
            dauc_p_le0=round(float(np.mean(np.array(dauc) <= 0)), 4),
        )
    return out

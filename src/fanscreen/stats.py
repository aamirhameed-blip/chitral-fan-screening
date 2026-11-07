"""Spatial autocorrelation and multiple-testing helpers."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def morans_i(values: np.ndarray, xy: np.ndarray, band: float) -> float:
    """Global Moran's I with binary weights for all pairs closer than ``band`` (same units as ``xy``)."""
    z = values - values.mean()
    dist = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    w = ((dist > 0) & (dist <= band)).astype(float)
    return len(values) / w.sum() * (z @ w @ z) / (z @ z)


def morans_i_test(
    values: np.ndarray, xy: np.ndarray, band: float, n_perm: int = 999, rng: np.random.Generator | None = None
) -> dict:
    """Moran's I with a one-sided permutation p-value."""
    rng = rng or np.random.default_rng()
    observed = morans_i(values, xy, band)
    null = np.array([morans_i(rng.permutation(values), xy, band) for _ in range(n_perm)])
    return dict(I=float(observed), p=float((1 + (null >= observed).sum()) / (n_perm + 1)))


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (false discovery rate)."""
    p = np.asarray(p, dtype=float)
    ranks = rankdata(p)
    order = np.argsort(-ranks)
    adjusted = np.minimum.accumulate((p * len(p) / ranks)[order])
    return adjusted[np.argsort(order)].clip(max=1)

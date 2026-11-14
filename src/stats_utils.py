"""
Statistical helpers for honest model reporting.

- bootstrap_auc_ci: percentile bootstrap 95% CI for ROC-AUC.
- delong_test: paired DeLong test comparing two correlated ROC-AUCs on the same patients
  (DeLong, DeLong & Clarke-Pearson 1988; fast implementation after Sun & Xu 2014).
"""

from typing import Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def bootstrap_auc_ci(
    y_true, scores, n_boot: int = 2000, alpha: float = 0.05, random_state: int = 42
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for ROC-AUC."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue
        aucs.append(roc_auc_score(y_true[idx], scores[idx]))
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _delong_components(y_true: np.ndarray, scores: np.ndarray):
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    m, n = len(pos), len(neg)
    tx, ty, tz = _midrank(pos), _midrank(neg), _midrank(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:m] - tx) / n          # structural components for positives
    v10 = 1.0 - (tz[m:] - ty) / m    # structural components for negatives
    return auc, v01, v10


def delong_test(y_true, scores_a, scores_b) -> Tuple[float, float, float]:
    """
    Paired two-sided DeLong test for AUC(a) vs AUC(b) on the same samples.

    Returns
    -------
    (auc_a, auc_b, p_value)
    """
    y_true = np.asarray(y_true).astype(int)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    auc_a, v01_a, v10_a = _delong_components(y_true, a)
    auc_b, v01_b, v10_b = _delong_components(y_true, b)
    m, n = len(v01_a), len(v10_a)
    s01 = np.cov(np.vstack([v01_a, v01_b]))
    s10 = np.cov(np.vstack([v10_a, v10_b]))
    cov = s01 / m + s10 / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(auc_a), float(auc_b), 1.0
    z = (auc_a - auc_b) / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return float(auc_a), float(auc_b), float(p)

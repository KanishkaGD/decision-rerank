"""Calibration of P(relevant) against human labels.

Binary metrics score the probability of the positive class (a forecast of "this pair has
rel >= t"), not the top-label confidence Laya's own ECE uses:
- ECE: 15 equal-width bins, sum_b |B|/n * |mean(y) - mean(p)|;
- adaptive ECE: the same with 15 equal-mass bins;
- Brier, NLL (p clipped to [1e-7, 1 - 1e-7]) and AUROC (discrimination, not calibration).

Graded metrics score Laya's distribution over grades 0-3: the ranked probability score
RPS = 1/(K-1) * sum_k (F_pred(k) - F_true(k))^2, argmax accuracy, and Spearman rho of E[y] vs
the grade. Confidence intervals resample queries (all of a query's pairs together), 1,000 times.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

BINS = 15
EPS = 1e-7


def _ece(p: np.ndarray, y: np.ndarray, edges_of) -> float:
    idx = edges_of(p)
    total = 0.0
    for b in np.unique(idx):
        m = idx == b
        total += m.sum() * abs(y[m].mean() - p[m].mean())
    return total / len(p)


def ece(p: np.ndarray, y: np.ndarray, bins: int = BINS) -> float:
    return _ece(p, y, lambda x: np.minimum((x * bins).astype(int), bins - 1))


def adaptive_ece(p: np.ndarray, y: np.ndarray, bins: int = BINS) -> float:
    order = np.argsort(p, kind="stable")
    idx = np.empty(len(p), int)
    idx[order] = np.arange(len(p)) * bins // len(p)
    return _ece(p, y, lambda _x: idx)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def nll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auroc(p: np.ndarray, y: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


BINARY = {"ECE": ece, "aECE": adaptive_ece, "Brier": brier, "NLL": nll, "AUROC": auroc}


def rps(P: np.ndarray, grade: np.ndarray) -> float:
    K = P.shape[1]
    F = np.cumsum(P, axis=1)[:, :-1]
    T = (grade[:, None] <= np.arange(K - 1)[None, :]).astype(float)
    return float(np.mean(((F - T) ** 2).sum(axis=1) / (K - 1)))


def graded(P: np.ndarray, grade: np.ndarray) -> dict:
    return {"RPS": rps(P, grade), "Acc": float((P.argmax(1) == grade).mean()),
            "Spearman": float(stats.spearmanr(P @ np.arange(P.shape[1]), grade).statistic)}


def reliability(p: np.ndarray, y: np.ndarray, bins: int = BINS) -> pd.DataFrame:
    """Equal-width bins: bin, lo, hi, n, mean_p, frac_pos (empty bins dropped)."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    df = pd.DataFrame({"bin": idx, "p": p, "y": y})
    g = df.groupby("bin").agg(n=("p", "size"), mean_p=("p", "mean"), frac_pos=("y", "mean")).reset_index()
    g["lo"], g["hi"] = g["bin"] / bins, (g["bin"] + 1) / bins
    return g


def query_groups(frame: pd.DataFrame) -> list[np.ndarray]:
    """Row positions of each (qset, query_id), for `bootstrap`."""
    codes = frame.groupby(["qset", "query_id"], sort=True).ngroup().to_numpy()
    order = np.argsort(codes, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)


def bootstrap(fn, groups: list[np.ndarray], n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile CI of fn(row positions), resampling queries with replacement."""
    rng = np.random.default_rng(seed)
    vals = [fn(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
            for _ in range(n)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)

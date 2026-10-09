"""
RQ2, Part C (docs/notes/calibration-proposal.md): reranker fine-tunes calibrated after the fact.

The objection to RQ2 is "just calibrate the reranker". This module fits the two standard one-model scalings on the
reranker's logit z -- temperature scaling, p = sigmoid(z / T), and Platt scaling, p = sigmoid(a z + b) -- on
msmarco-dev only (never on a test split, CLAUDE.md), and evaluates the scaled probabilities on the test collections:
ECE and Brier on the judged pairs of DL19-23, the fixed cutoff of Part A (src/cdm/cutoff.py), and ECE with unjudged
passages as non-relevant on every collection (on BEIR the shift check, since BEIR judges few non-relevant passages).

Two facts shape the reading:
- Both scalings are monotone, so the ranking (nDCG) never changes; temperature scaling keeps z = 0 at p = 0.5, so the
  set kept at p >= 0.5 does not change either -- only Platt's bias b moves the cutoff.
- msmarco-dev judges about one relevant passage per query and no non-relevant ones; the fit counts unjudged
  candidates as non-relevant, which pulls the probabilities down. The in-sample row (dataset msmarco-dev) and the
  test rows show how much that matters.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

from cdm.cutoff import QueryScores, cutoff_analysis
from cdm.evaluation import calibration

METHODS = ("native", "temperature", "platt")
EPS = 1e-7


def logits(queries: Sequence[QueryScores]) -> np.ndarray:
    """
    The logit of every candidate, concatenated over queries. Where the stored score is the logit itself
    (prob = sigmoid(score), true for the cross-encoders and yes/no readers we scale) it is used directly, which
    avoids the saturation of probabilities near 0 and 1; otherwise the logit of the clipped probability.
    """
    scores = np.concatenate([q.scores for q in queries])
    probs = np.concatenate([q.probs for q in queries])
    if np.allclose(1 / (1 + np.exp(-scores)), probs, atol=1e-4):
        return scores
    p = np.clip(probs, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def fit_logistic(z: np.ndarray, y: np.ndarray, bias: bool, iterations: int = 100) -> Tuple[float, float]:
    """
    Minimum-NLL (a, b) of p = sigmoid(a z + b) by Newton's method with a backtracking line search (plain Newton
    overshoots from a poor start); b = 0 without bias (temperature scaling, T = 1/a). One or two parameters, so it
    converges in a few steps; a tiny ridge keeps the Hessian invertible when the classes separate.
    """
    X = np.stack([z, np.ones_like(z)], axis=1) if bias else z[:, None]
    w = np.zeros(X.shape[1])
    w[0] = 1.0  # start at the native probabilities

    def loss(v: np.ndarray) -> float:
        return nll(_sigmoid(X @ v), y) + 0.5e-8 * float(v @ v) / len(y)

    current = loss(w)
    for _ in range(iterations):
        p = _sigmoid(X @ w)
        grad = X.T @ (p - y) + 1e-8 * w
        hess = (X * (p * (1 - p))[:, None]).T @ X + 1e-8 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while t > 1e-10 and loss(w - t * step) > current:
            t /= 2
        w = w - t * step
        new = loss(w)
        if abs(current - new) < 1e-12 or np.max(np.abs(t * step)) < 1e-10:
            break
        current = new
    return float(w[0]), float(w[1]) if bias else 0.0


def nll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit(dev: Sequence[QueryScores]) -> Dict[str, Tuple[float, float]]:
    """method -> (a, b) fitted on the dev queries, every candidate labelled (unjudged = non-relevant)."""
    z = logits(dev)
    y = np.concatenate([q.relevant for q in dev]).astype(float)
    return {
        "native": (1.0, 0.0),
        "temperature": fit_logistic(z, y, bias=False),
        "platt": fit_logistic(z, y, bias=True),
    }


def scaled(queries: Sequence[QueryScores], a: float, b: float) -> List[QueryScores]:
    """The queries with probabilities sigmoid(a z + b); order, labels and scores unchanged."""
    out, z, start = [], logits(queries), 0
    for q in queries:
        n = len(q.probs)
        out.append(QueryScores(q.qid, _sigmoid(a * z[start : start + n] + b), q.labels, q.scores))
        start += n
    return out


def evaluate_scaled(queries: Sequence[QueryScores], with_cutoff: bool) -> Dict[str, float]:
    """
    ECE/Brier on the judged pairs (only where judged non-relevant pairs exist), ECE/Brier with unjudged as
    non-relevant, NLL of the latter, and -- on the collections with judged non-relevant passages -- Part A's cutoff
    values.
    """
    p = np.concatenate([q.probs for q in queries])
    labels = np.concatenate([q.labels for q in queries])
    judged = labels >= 0
    out: Dict[str, float] = {}
    if (labels == 0).any():
        out.update(calibration(p[judged], labels[judged]))
    everything = calibration(p, (labels == 1).astype(int))
    out.update(ECE_unjudged0=everything["ECE"], Brier_unjudged0=everything["Brier"])
    out["NLL_unjudged0"] = nll(p, (labels == 1).astype(float))
    out["mean_prob"] = float(p.mean())
    if with_cutoff:
        out.update(cutoff_analysis(queries))
    return out

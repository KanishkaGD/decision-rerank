"""
RQ2, Part A (docs/notes/calibration-proposal.md): one fixed probability cutoff across collections.

Per query, keep every reranked candidate whose relevance probability is at least tau and score the kept set against
the judgments: precision, recall and F1 of the set, its size, and whether it is empty. A calibrated model should do
well with tau = 0.5 on every collection without tuning; the regret is how much F1 the per-collection oracle cutoff
(chosen afterwards on that collection: an upper bound, never used as a setting) gains over tau = 0.5. Fixed top-k
sets of the same ranking are the baseline that needs no probability at all.

Conventions (docs/roadmap.md, "RQ2 evidence"):
- relevant = graded label >= the dataset's rel_threshold (TREC DL: 2); unjudged passages count as non-relevant, and
  the share of unjudged passages in the kept sets is reported so this can be judged;
- recall is relative to the relevant passages among the candidates (what a reranker can find);
- a query without any relevant candidate scores 1 for an empty set (correct abstention) and 0 otherwise; an empty
  set for a query with relevant candidates scores 0;
- means are macro averages over the judged queries.

No model runs: everything comes from the stored outputs (scores.jsonl.gz).
"""

import gzip
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cdm.runs import Qrels

TAUS = (0.3, 0.5, 0.7)
TOP_KS = (5, 10, 20)
# Oracle search grid 0.000, 0.001, ..., 1.000: fine enough for every probability we store (Jev's have two decimals),
# and it contains TAUS.
ORACLE_GRID = np.round(np.linspace(0.0, 1.0, 1001), 3)


@dataclass
class QueryScores:
    """
    One query's candidates in ranking order: probabilities, labels (1 relevant, 0 not, -1 unjudged) and the raw
    ranking scores (for the post-hoc calibration of src/cdm/posthoc.py; None where not read).
    """

    qid: str
    probs: np.ndarray
    labels: np.ndarray
    scores: Optional[np.ndarray] = None

    @property
    def relevant(self) -> np.ndarray:
        return self.labels == 1

    @property
    def unjudged(self) -> np.ndarray:
        return self.labels == -1


def read_scores(path: str, qrels: Qrels, rel_threshold: int) -> List[QueryScores]:
    """scores.jsonl.gz of a rerank output -> the judged queries that have probabilities, in file order."""
    by_query: Dict[str, List[Tuple[float, float, int, int]]] = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r.get("prob") is None or r["qid"] not in qrels:
                continue
            rel = qrels[r["qid"]].get(r["docid"])
            label = -1 if rel is None else int(rel >= rel_threshold)
            by_query.setdefault(r["qid"], []).append((r["prob"], r["score"], r.get("first_stage_rank") or 0, label))
    out = []
    for qid, rows in by_query.items():
        probs, scores, ranks, labels = (np.asarray(x, dtype=float) for x in zip(*rows))
        # Ranking order (for top-k): the reranker's score, ties broken by the first-stage rank.
        order = np.lexsort((ranks, -scores))
        out.append(QueryScores(qid, probs[order], labels[order].astype(int), scores[order]))
    return out


def set_metrics(kept: np.ndarray, relevant: np.ndarray, unjudged: np.ndarray) -> Dict[str, np.ndarray]:
    """
    kept: (n_candidates, n_cutoffs) booleans, one column per cutoff. Per-cutoff precision, recall, F1, size, empty
    and the share of unjudged passages in the kept set (0 for an empty set), following the module's conventions.
    """
    size = kept.sum(axis=0).astype(float)
    hits = (kept & relevant[:, None]).sum(axis=0).astype(float)
    n_rel = int(relevant.sum())
    nonempty = size > 0
    safe_size = np.where(nonempty, size, 1.0)
    unjudged_share = np.where(nonempty, (kept & unjudged[:, None]).sum(axis=0) / safe_size, 0.0)
    if n_rel == 0:  # nothing to find: abstaining is right, keeping anything is wrong
        f1 = (~nonempty).astype(float)
        precision = recall = f1
    else:
        precision = np.where(nonempty, hits / safe_size, 0.0)
        recall = hits / n_rel
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / np.where(denom > 0, denom, 1.0), 0.0)
    return {
        "P": precision,
        "R": recall,
        "F1": f1,
        "size": size,
        "empty": (~nonempty).astype(float),
        "unjudged": unjudged_share,
    }


def _macro(per_query: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {m: np.mean([pq[m] for pq in per_query], axis=0) for m in per_query[0]} if per_query else {}


def threshold_metrics(queries: Sequence[QueryScores], taus: Sequence[float]) -> Dict[str, np.ndarray]:
    """Macro means over queries of the set metrics for every cutoff in taus (arrays aligned with taus)."""
    t = np.asarray(taus, dtype=float)[None, :]
    return _macro([set_metrics(q.probs[:, None] >= t, q.relevant, q.unjudged) for q in queries])


def topk_metrics(queries: Sequence[QueryScores], ks: Sequence[int]) -> Dict[str, np.ndarray]:
    """Macro means of the set metrics of each query's top-k (arrays aligned with ks)."""
    k = np.asarray(ks)[None, :]
    return _macro([set_metrics(np.arange(len(q.probs))[:, None] < k, q.relevant, q.unjudged) for q in queries])


def cutoff_analysis(
    queries: Sequence[QueryScores], taus: Sequence[float] = TAUS, ks: Sequence[int] = TOP_KS
) -> Dict[str, float]:
    """
    The values Part A stores per (system, dataset): set metrics at each fixed tau (cut_<metric>@<tau>), the oracle
    cutoff and its F1 (cut_oracle_tau, cut_F1@oracle), the regret of tau = 0.5 (cut_regret) and top-k F1
    (topk_F1@<k>).
    """
    if not queries:
        return {}
    out: Dict[str, float] = {}
    fixed = threshold_metrics(queries, taus)
    for i, tau in enumerate(taus):
        for metric, values in fixed.items():
            out[f"cut_{metric}@{tau:g}"] = float(values[i])
    grid = threshold_metrics(queries, ORACLE_GRID)["F1"]
    best = int(np.argmax(grid))  # the lowest cutoff among ties
    out["cut_oracle_tau"] = float(ORACLE_GRID[best])
    out["cut_F1@oracle"] = float(grid[best])
    out["cut_regret"] = float(grid[best] - grid[int(np.argmin(np.abs(ORACLE_GRID - 0.5)))])
    top = topk_metrics(queries, ks)
    for i, k in enumerate(ks):
        out[f"topk_F1@{k}"] = float(top["F1"][i])
    out["cut_n_queries"] = len(queries)
    out["cut_n_queries_no_rel"] = int(sum(not q.relevant.any() for q in queries))
    return out


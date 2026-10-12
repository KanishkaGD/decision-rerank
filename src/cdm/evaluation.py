"""
Effectiveness (ir_measures), calibration (ECE, Brier) and significance (paired bootstrap, Holm) -- the metrics of
the research plan's "Experimental setup".
"""

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from cdm.config import ECE_BINS, NDCG_CUTOFFS, RECALL_CUTOFFS
from cdm.runs import Qrels, Run


def effectiveness_measures(rel_threshold: int) -> Dict[str, object]:
    """
    Measures by the column name they get in results.csv. Binary measures count labels >= rel_threshold as
    relevant (TREC DL: 2, BEIR: 1), so the names stay the same across datasets while the threshold is per dataset.
    """
    from ir_measures import parse_measure

    t = rel_threshold
    measures = {f"nDCG@{k}": parse_measure(f"nDCG@{k}") for k in NDCG_CUTOFFS}
    measures.update({f"R@{k}": parse_measure(f"R(rel={t})@{k}") for k in RECALL_CUTOFFS})
    measures["RR@10"] = parse_measure(f"RR(rel={t})@10")
    measures["AP"] = parse_measure(f"AP(rel={t})")
    return measures


def evaluate(run: Run, qrels: Qrels, rel_threshold: int) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Returns (mean over the judged queries, per-query table with one column per measure). Only queries that have
    judgments are evaluated; per-query values are what the significance tests need.
    """
    import ir_measures

    measures = effectiveness_measures(rel_threshold)
    by_measure = {m: name for name, m in measures.items()}
    judged_run = {qid: docs for qid, docs in run.items() if qid in qrels}
    # Only the queries that were ranked: ir_measures scores a judged query missing from the run as 0, which would
    # drag the mean down whenever the run covers a subset (--max-queries).
    ranked_qrels = {qid: qrels[qid] for qid in judged_run}
    rows: Dict[str, Dict[str, float]] = {}
    for metric in ir_measures.iter_calc(list(measures.values()), ranked_qrels, judged_run):
        rows.setdefault(metric.query_id, {})[by_measure[metric.measure]] = metric.value
    per_query = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=list(measures))
    per_query.index.name = "qid"
    means: Dict[str, float] = {name: float(per_query[name].mean()) for name in measures}
    means["n_queries"] = len(per_query)
    return means, per_query.sort_index()


def calibration(probs: Sequence[float], labels: Sequence[int], n_bins: int = ECE_BINS) -> Dict[str, float]:
    """
    Expected calibration error (equal-width bins, weighted by bin size) and Brier score of predicted relevance
    probabilities against binary labels. Callers pass only judged pairs: unjudged passages have no label.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if len(p) == 0:
        return {"ECE": float("nan"), "Brier": float("nan"), "n_judged_pairs": 0}
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(p[in_bin].mean() - y[in_bin].mean())
    return {"ECE": float(ece), "Brier": float(np.mean((p - y) ** 2)), "n_judged_pairs": int(len(p))}


def judged_pairs(
    scored: Sequence[Tuple[str, str, float]], qrels: Qrels, rel_threshold: int, unjudged_as_nonrelevant: bool = False
) -> Tuple[List[float], List[int]]:
    """
    (qid, docid, prob) triples -> (probs, binary labels). By default only the pairs that have a judgment; with
    unjudged_as_nonrelevant, every pair of a judged query, unjudged ones labelled 0 (the BEIR convention, needed where
    the qrels list only relevant documents). Main calibration results
    come from judged pairs on sets that judge non-relevant documents too; the unjudged-as-0 variant is secondary.
    """
    probs, labels = [], []
    for qid, docid, prob in scored:
        if qid not in qrels:
            continue
        rel = qrels[qid].get(docid)
        if rel is None and not unjudged_as_nonrelevant:
            continue
        probs.append(prob)
        labels.append(int(rel is not None and rel >= rel_threshold))
    return probs, labels


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_resamples: int = 10_000, seed: int = 42) -> float:
    """
    Two-sided p-value of the paired bootstrap test (shift method) for mean(a) != mean(b), where a and b are
    per-query values of two systems on the same queries, in the same order.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) == 0:
        return float("nan")
    observed = d.mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_resamples, len(d)))
    boot = (d - observed)[idx].mean(axis=1)
    return float(np.mean(np.abs(boot) >= abs(observed)))


def holm(p_values: Mapping[str, float], alpha: float = 0.05) -> Dict[str, bool]:
    """Holm-Bonferroni correction across e.g. datasets: name -> whether the difference is significant."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    significant: Dict[str, bool] = {}
    still_rejecting = True
    for i, (name, p) in enumerate(ordered):
        still_rejecting = still_rejecting and p <= alpha / (m - i)
        significant[name] = still_rejecting
    return significant

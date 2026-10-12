"""
RQ1 significance (docs/roadmap.md, "Significance"): paired t-tests over queries on the stored per-query nDCG@10
(perquery.csv of each rerank output), Holm-corrected across the ten collections per comparison. Comparisons, per
backbone of the triplet table (cdm.paper_tables.TRIPLETS): each reranker fine-tune against the decision fine-tune with
the best average (its better mode), that decision fine-tune against the base model, and joint against isolated for
every decision model run in both modes (plus Jev). Literature cells have no per-query runs and are not tested.
"""

import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

METRIC = "nDCG@10"
ALPHA = 0.05

Pair = Tuple[str, str, str]  # system a, system b, comparison


def comparisons(averages: Dict[str, Optional[float]]) -> List[Pair]:
    """The tested pairs, from the triplet table and each system's average over the ten collections."""
    from cdm.paper_tables import TRIPLETS

    pairs: List[Pair] = []
    for _, rows in TRIPLETS:
        decision = [s for s, _, objective, _ in rows if objective == "decision" and averages.get(s) is not None]
        best = max(decision, key=lambda s: averages[s]) if decision else None
        if best is not None:
            pairs += [(s, best, "reranker-vs-decision") for s, _, objective, _ in rows if objective == "reranker"]
            pairs += [(best, s, "decision-vs-base") for s, _, objective, _ in rows if objective == "base"]
        names = {(name, mode): s for s, name, objective, mode in rows if objective == "decision"}
        for (name, mode), system in names.items():
            if mode == "joint" and (name, "isolated") in names:
                pairs.append((system, names[(name, "isolated")], "joint-vs-isolated"))
    pairs.append(("jev-joint", "jev-isolated", "joint-vs-isolated"))
    return pairs


def read_perquery(output: str, metric: str = METRIC) -> Dict[str, float]:
    """qid -> metric from an output directory's perquery.csv."""
    with open(os.path.join(output, "perquery.csv"), newline="") as f:
        return {row["qid"]: float(row[metric]) for row in csv.DictReader(f)}


def paired_t(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    """Two-sided paired t-test on the queries both systems scored: mean difference a - b, t, p, n."""
    from scipy.stats import ttest_rel

    qids = sorted(set(a) & set(b))
    x = np.array([a[q] for q in qids])
    y = np.array([b[q] for q in qids])
    if len(qids) < 2 or np.allclose(x, y):
        return {"delta": float(np.mean(x - y)) if qids else float("nan"), "t": 0.0, "p": 1.0, "n": len(qids)}
    test = ttest_rel(x, y)
    return {"delta": float(np.mean(x - y)), "t": float(test.statistic), "p": float(test.pvalue), "n": len(qids)}


def holm_flags(p_values: Sequence[float], alpha: float = ALPHA) -> List[bool]:
    """Holm-Bonferroni across the given p-values (here: the collections of one comparison)."""
    from cdm.evaluation import holm

    flags = holm({str(i): p for i, p in enumerate(p_values)}, alpha)
    return [flags[str(i)] for i in range(len(p_values))]

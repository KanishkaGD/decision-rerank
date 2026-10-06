"""Per-query effectiveness of a reranked official TREC DL pool, and paired significance tests.

Measures (ir_measures / pytrec_eval, trec_eval semantics, unless noted):
- nDCG@10 with trec_eval's linear gain (primary);
- nDCG@10 (exp): exponential gain 2^rel - 1, ideal DCG over every judged passage (computed here:
  pytrec_eval ignores ir_measures' custom gain map and returns 0);
- nDCG'@10: the condensed-list variant (Sakai 2007), unjudged passages removed before ranking.
  New systems surface passages the DL pools never judged; this bounds how much that costs them;
- RR@10 and AP@1000 at rel >= 2 (TREC DL passage convention);
- Judged@10, the share of the top 10 that has a label (unjudged passages count as non-relevant).

Ties in a reranker's score are broken by BM25 rank, then the list is handed to pytrec_eval as
descending integer scores, so trec_eval's own doc-id tie-break never applies.
"""

from __future__ import annotations

import ir_measures
import numpy as np
import pandas as pd
from ir_measures import AP, RR, Judged, nDCG
from scipy import stats

from . import data

IRM = {
    "nDCG@10": nDCG @ 10,
    "RR@10": RR(rel=data.REL_MIN) @ 10,
    "AP@1000": AP(rel=data.REL_MIN) @ 1000,
    "Judged@10": Judged @ 10,
}
MEASURES = ["nDCG@10", "nDCG@10 (exp)", "nDCG'@10", "RR@10", "AP@1000", "Judged@10"]
DISCOUNT = 1 / np.log2(np.arange(2, 12))


def ranked_run(frame: pd.DataFrame, col: str, tie: str = "bm25_rank") -> dict:
    """{qid: {doc: score}} over the pool pairs, ordered by `col` desc, then `tie` ascending."""
    top = frame[frame.in_pool].sort_values(["query_id", col, tie], ascending=[True, False, True])
    top = top.assign(pos=top.groupby("query_id").cumcount())
    run: dict = {}
    for q, d, pos in zip(top.query_id, top.doc_id, top.pos):
        run.setdefault(q, {})[d] = float(-pos)
    return run


def ndcg10_exp(run: dict, qr: dict) -> pd.Series:
    out = {}
    for q, docs in run.items():
        rel = qr.get(q, {})
        top = sorted(docs, key=docs.get, reverse=True)[:10]
        dcg = sum((2.0 ** rel.get(d, 0) - 1) * DISCOUNT[i] for i, d in enumerate(top))
        ideal = sorted((2.0 ** r - 1 for r in rel.values()), reverse=True)[:10]
        idcg = sum(g * DISCOUNT[i] for i, g in enumerate(ideal))
        out[q] = dcg / idcg if idcg > 0 else 0.0
    return pd.Series(out)


def _irm(run: dict, qr: dict, measures: dict) -> pd.DataFrame:
    names = {repr(m): n for n, m in measures.items()}
    rows = [(m.query_id, names[repr(m.measure)], m.value)
            for m in ir_measures.iter_calc(list(measures.values()), qr, run)]
    return pd.DataFrame(rows, columns=["query_id", "measure", "value"]).pivot(
        index="query_id", columns="measure", values="value")


def per_query(frame: pd.DataFrame, col: str, qrels: pd.DataFrame, tie: str = "bm25_rank") -> pd.DataFrame:
    """Rows: query_id; columns: MEASURES. `frame` needs in_pool, rel, bm25_rank and `col`.

    Only queries present in `frame` are scored (pytrec_eval would score a qrels-only query as 0).
    """
    qrels = qrels[qrels.query_id.isin(set(frame.query_id))]
    qr = {}
    for q, d, r in zip(qrels.query_id, qrels.doc_id, qrels.rel):
        qr.setdefault(q, {})[d] = int(r)
    run = ranked_run(frame, col, tie)
    out = _irm(run, qr, IRM)
    out["nDCG@10 (exp)"] = ndcg10_exp(run, qr)
    condensed = _irm(ranked_run(frame[frame.rel.notna()], col, tie), qr, {"nDCG'@10": nDCG @ 10})
    out["nDCG'@10"] = condensed["nDCG'@10"].reindex(out.index).fillna(0.0)
    return out[MEASURES]


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, (len(p) - i) * p[j])
        adj[j] = min(1.0, running)
    return adj


def paired_t(a: pd.Series, b: pd.Series) -> float:
    a, b = a.align(b, join="inner")
    if np.allclose(a.to_numpy(), b.to_numpy()):
        return 1.0
    return float(stats.ttest_rel(a, b).pvalue)


def tie_rate(frame: pd.DataFrame, col: str) -> float:
    """Share of pool pairs whose score ties another pair of the same query."""
    top = frame[frame.in_pool]
    return float(top.duplicated(["query_id", col], keep=False).mean())

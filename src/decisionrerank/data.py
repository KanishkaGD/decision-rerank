"""Frozen inputs under ``data/`` (written once by ``scripts/prepare_data.py``).

- ``queries.parquet``    qset, query_id, text: the judged DL19 (43) and DL20 (54) queries.
- ``qrels.parquet``      qset, query_id, doc_id, rel: human grades 0-3.
- ``candidates.parquet`` qset, query_id, doc_id, bm25, bm25_rank: the official TREC DL passage
  rerank-subtask pool (msmarco-passagetest{2019,2020}-top1000; up to 1000 per query). The files are
  unranked, so ``bm25``/``bm25_rank`` are our bm25s scores (Lucene variant, k1=1.2, b=0.75, the
  rerank-depth index) over that pool: the BM25 baseline and every reranker's tie-break.
- ``bm25_std.parquet``   qset, query_id, doc_id, std_rank, std_score: the standard BM25 run (Anserini default,
  k1=0.9, b=0.4; reproduces its DL19/20 regression scores), top 1000, for QPP; its top 10 joins the pair set.
- ``passages.parquet``   doc_id, text: every passage in the pool, the qrels or the standard run's top 10.

Every model scores the same pair set: the union of the pool and the judged pairs. Pool pairs feed
the effectiveness evaluation; judged pairs feed calibration, so the calibration results do not
depend on the candidate pool.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA, RUNS, RESULTS, FIGURES = ROOT / "data", ROOT / "runs", ROOT / "results", ROOT / "figures"
QSETS = ("dl19", "dl20")
REL_MIN = 2  # TREC DL passage convention for binary relevance
KEY = ["qset", "query_id", "doc_id"]


def queries() -> pd.DataFrame:
    return pd.read_parquet(DATA / "queries.parquet")


def qrels() -> pd.DataFrame:
    return pd.read_parquet(DATA / "qrels.parquet")


def candidates() -> pd.DataFrame:
    return pd.read_parquet(DATA / "candidates.parquet")


def bm25_std() -> pd.DataFrame | None:
    path = DATA / "bm25_std.parquet"
    return pd.read_parquet(path) if path.exists() else None


def passages() -> pd.Series:
    return pd.read_parquet(DATA / "passages.parquet").set_index("doc_id")["text"]


def pairs() -> pd.DataFrame:
    """qset, query_id, doc_id, query, passage, in_pool, in_std10, rel (NaN if unjudged), sorted."""
    pool = candidates()[KEY].assign(in_pool=True)
    p = pool.merge(qrels(), on=KEY, how="outer")
    std = bm25_std()
    if std is not None:
        p = p.merge(std[std.std_rank <= 10][KEY].assign(in_std10=True), on=KEY, how="outer")
    for flag in ("in_pool", "in_std10"):
        p[flag] = p[flag].astype("boolean").fillna(False).astype(bool) if flag in p else False
    q = queries().set_index(["qset", "query_id"])["text"]
    p["query"] = q.loc[list(zip(p.qset, p.query_id))].to_numpy()
    p["passage"] = passages().loc[p.doc_id].to_numpy()
    return p.sort_values(KEY).reset_index(drop=True)


def load_run(model: str) -> pd.DataFrame:
    return pd.read_parquet(RUNS / f"{model}.parquet")

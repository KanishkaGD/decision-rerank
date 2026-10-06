"""Frozen inputs under ``data/`` (written once by ``scripts/prepare_data.py``).

- ``queries.parquet``    qset, query_id, text: the judged DL19 (43) and DL20 (54) queries.
- ``qrels.parquet``      qset, query_id, doc_id, rel: human grades 0-3.
- ``candidates.parquet`` qset, query_id, doc_id, bm25, bm25_rank: the official TREC DL passage
  rerank-subtask pool (msmarco-passagetest{2019,2020}-top1000; up to 1000 per query). The files are
  unranked, so ``bm25``/``bm25_rank`` are our bm25s scores (Lucene variant, k1=1.2, b=0.75, the
  rerank-depth index) over that pool: the BM25 baseline and every reranker's tie-break.
- ``passages.parquet``   doc_id, text: every passage in the pool or the qrels.

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


def passages() -> pd.Series:
    return pd.read_parquet(DATA / "passages.parquet").set_index("doc_id")["text"]


def pairs() -> pd.DataFrame:
    """qset, query_id, doc_id, query, passage, in_pool, rel (NaN if unjudged), sorted."""
    pool = candidates()[KEY].assign(in_pool=True)
    p = pool.merge(qrels(), on=KEY, how="outer")
    p["in_pool"] = p["in_pool"].astype("boolean").fillna(False).astype(bool)
    q = queries().set_index(["qset", "query_id"])["text"]
    p["query"] = q.loc[list(zip(p.qset, p.query_id))].to_numpy()
    p["passage"] = passages().loc[p.doc_id].to_numpy()
    return p.sort_values(KEY).reset_index(drop=True)


def load_run(model: str) -> pd.DataFrame:
    return pd.read_parquet(RUNS / f"{model}.parquet")

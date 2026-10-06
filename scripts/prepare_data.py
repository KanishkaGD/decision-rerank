"""Freeze the DL19/20 inputs into data/.

Sources (read-only):
- qrels and query texts: ir_datasets msmarco-passage/trec-dl-{2019,2020} files
- candidate pool: the official rerank-subtask files msmarco-passagetest{2019,2020}-top1000.tsv.gz
  (downloaded to data/raw/ from https://msmarco.z22.web.core.windows.net/msmarcoranking/)
- passages: qpp-pruning-study/data/prepared/msmarco/corpus.parquet (MS MARCO v1, 8.8M)
- BM25 scores over the pool: the rerank-depth bm25s index (Lucene variant, k1=1.2, b=0.75)
"""

import hashlib
import json

import bm25s
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import Stemmer

from decisionrerank.data import DATA

IRDS = "/root/qpp-pruning-study/data/ir_datasets/msmarco-passage"
CORPUS = "/root/qpp-pruning-study/data/prepared/msmarco/corpus.parquet"
INDEX = "/root/rerank-depth/data/bm25_index"
URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/"
YEARS = {"dl19": "2019", "dl20": "2020"}


def bm25_scores(pool: pd.DataFrame, queries: pd.DataFrame) -> np.ndarray:
    """Score of every pool pair under the bm25s index (same tokenizer and stemmer as built)."""
    ret = bm25s.BM25.load(INDEX, mmap=True)
    doc_ids = np.load(f"{INDEX}/doc_ids.npy", allow_pickle=True)
    pos = pd.Series(np.arange(len(doc_ids)), index=doc_ids)
    stem = Stemmer.Stemmer("english")
    text = queries.set_index("query_id")["text"]
    out = np.empty(len(pool), np.float32)
    for qid, idx in pool.groupby("query_id").indices.items():
        tok = bm25s.tokenize([text[qid]], stopwords="en", stemmer=stem, show_progress=False, return_ids=False)[0]
        out[idx] = ret.get_scores(tok)[pos.loc[pool.doc_id.iloc[idx]].to_numpy()]
    return out


def main():
    DATA.mkdir(exist_ok=True)
    qrels, queries, pools = [], [], []
    for qset, year in YEARS.items():
        q = pd.read_csv(f"{IRDS}/trec-dl-{year}/qrels", sep=r"\s+", header=None, dtype=str,
                        names=["query_id", "it", "doc_id", "rel"])
        qrels.append(q.assign(qset=qset, rel=q.rel.astype(int))[["qset", "query_id", "doc_id", "rel"]])
        t = pd.read_csv(f"{IRDS}/trec-dl-{year}/queries.tsv", sep="\t", header=None, dtype=str,
                        names=["query_id", "text"], quoting=3)
        queries.append(t[t.query_id.isin(set(q.query_id))].assign(qset=qset))
        f = pd.read_csv(DATA / "raw" / f"msmarco-passagetest{year}-top1000.tsv.gz", sep="\t", header=None,
                        dtype=str, names=["query_id", "doc_id"], usecols=[0, 1], quoting=3)
        pools.append(f[f.query_id.isin(set(q.query_id))].drop_duplicates().assign(qset=qset))
    qrels = pd.concat(qrels, ignore_index=True)
    queries = pd.concat(queries, ignore_index=True)[["qset", "query_id", "text"]]
    pool = pd.concat(pools, ignore_index=True)[["qset", "query_id", "doc_id"]]
    assert queries.groupby("qset").size().to_dict() == {"dl19": 43, "dl20": 54}
    assert pool.groupby("qset").query_id.nunique().to_dict() == {"dl19": 43, "dl20": 54}

    pool["bm25"] = bm25_scores(pool, queries)
    pool = pool.sort_values(["qset", "query_id", "bm25", "doc_id"], ascending=[True, True, False, True])
    pool["bm25_rank"] = pool.groupby("query_id").cumcount() + 1

    # sanity: the index reproduces rerank-depth's stored top-1000 scores where the pool overlaps them
    ref = pd.read_parquet("/root/rerank-depth/runs/bm25.parquet").merge(pool, on=["query_id", "doc_id"])
    assert len(ref) > 1000 and np.allclose(ref.score, ref.bm25, atol=1e-3), "bm25s scores do not reproduce"

    need = sorted(set(pool.doc_id) | set(qrels.doc_id))
    tab = pq.read_table(CORPUS, filters=pc.field("doc_id").isin(need))
    passages = tab.to_pandas().drop_duplicates("doc_id").sort_values("doc_id")
    assert len(passages) == len(need), (len(passages), len(need))

    out = {"queries": queries, "qrels": qrels, "candidates": pool, "passages": passages}
    manifest = {"sources": {"irds": IRDS, "corpus": CORPUS, "pool": URL, "bm25_index": INDEX},
                "bm25_checked_pairs": len(ref), "files": {}}
    for name, df in out.items():
        path = DATA / f"{name}.parquet"
        df.reset_index(drop=True).to_parquet(path, index=False)
        manifest["files"][path.name] = {"rows": len(df),
                                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for stale in ("bm25.parquet",):
        (DATA / stale).unlink(missing_ok=True)
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

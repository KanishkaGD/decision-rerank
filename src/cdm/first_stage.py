"""
The candidates every reranker sees, cached once per dataset (src/cdm/runs.py): BM25 top-100 with Pyserini's prebuilt
indexes (retrieve_bm25), read from the track's official top-100 file where it publishes exactly that set
(official_pool, DL 2021-2023; see src/cdm/datasets.py).
"""

import gzip
import json
import os
import shutil
import urllib.request
from typing import Dict, Iterator, List, Optional, Tuple

from cdm.config import POOLS_FOLDER
from cdm.datasets import Dataset
from cdm.runs import Candidate, Qrels, QueryCandidates

# Pyserini's defaults for its prebuilt MS MARCO and BEIR (flat) BM25 runs; recorded in every manifest.
BM25_PARAMS = {"k1": 0.9, "b": 0.4}


def doc_text(raw: str) -> str:
    """
    Passage text from a Pyserini stored document: MS MARCO v1 {"contents"}, MS MARCO v2 {"passage", "pid", ...}, BEIR
    {"title", "text"}, or plain text.
    """
    try:
        d = json.loads(raw)
    except (TypeError, ValueError):
        return raw or ""
    if not isinstance(d, dict):
        return str(d)
    if "contents" in d:
        return d["contents"]
    if "passage" in d:
        return d["passage"]
    title, text = (d.get("title") or "").strip(), (d.get("text") or "").strip()
    return f"{title}\n{text}" if title else text


def load_topics_and_qrels(dataset: Dataset) -> Tuple[Dict[str, str], Qrels]:
    """Queries (only those with judgments, as TREC DL topic files also list unjudged ones) and qrels, ids as str."""
    from pyserini.search import get_qrels_file, get_topics

    # Parsed from the file rather than with get_qrels(), which turns numeric ids into ints and so would drop
    # leading zeros -- and with them the match against the docids of the index.
    qrels: Qrels = {}
    with open(get_qrels_file(dataset.qrels)) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4:
                qid, _, docid, rel = parts
                qrels.setdefault(qid, {})[docid] = int(rel)
    topics = {str(q): t["title"] for q, t in get_topics(dataset.topics).items()}
    queries = {qid: text for qid, text in topics.items() if qid in qrels}
    return queries, qrels


def retrieve_bm25(
    dataset: Dataset, depth: int, threads: int = 8, max_queries: Optional[int] = None
) -> Tuple[List[QueryCandidates], Qrels, Dict]:
    from pyserini.search.lucene import LuceneSearcher

    queries, qrels = load_topics_and_qrels(dataset)
    qids = sorted(queries)
    if max_queries:
        qids = qids[:max_queries]
    searcher = LuceneSearcher.from_prebuilt_index(dataset.index)
    searcher.set_bm25(**BM25_PARAMS)
    hits = searcher.batch_search([queries[q] for q in qids], qids, k=depth, threads=threads)

    results = []
    for qid in qids:
        cands = []
        for rank, hit in enumerate(hits.get(qid, []), start=1):
            doc = searcher.doc(hit.docid)
            cands.append(
                Candidate(
                    docid=hit.docid,
                    rank=rank,
                    score=float(hit.score),
                    text=doc_text(doc.raw()) if doc is not None else "",
                )
            )
        results.append(QueryCandidates(qid=qid, query=queries[qid], candidates=cands))
    params = {
        "retriever": "bm25",
        "index": dataset.index,
        "topics": dataset.topics,
        "qrels": dataset.qrels,
        "depth": depth,
        **BM25_PARAMS,
        "n_queries": len(qids),
    }
    return results, {q: qrels[q] for q in qids}, params


def pool_path(dataset: Dataset) -> str:
    """Where `cdm prefetch` keeps a dataset's official pool: data/pools/<dataset>/<file>."""
    assert dataset.pool is not None
    return os.path.join(POOLS_FOLDER, dataset.name, os.path.basename(dataset.pool.url))


def download_pool(dataset: Dataset) -> str:
    """Downloads the official pool once (in the prefetch step; compute nodes may have no internet)."""
    path = pool_path(dataset)
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with urllib.request.urlopen(dataset.pool.url) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out)
    os.replace(tmp, path)
    return path


def read_pool(path: str) -> Iterator[Tuple[str, str, float]]:
    """(qid, docid, track score) per line of an official pool, a TREC run file (qid Q0 docid rank score tag)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            qid, _, docid, _, score, _ = line.split()
            yield qid, docid, float(score)


def official_pool(dataset: Dataset, max_queries: Optional[int] = None) -> Tuple[List[QueryCandidates], Qrels, Dict]:
    """
    The official pool of the judged queries, in the track's order (file order; a stable sort by score keeps its tie
    order), with the passage texts from the index.
    """
    from pyserini.search.lucene import LuceneSearcher

    queries, qrels = load_topics_and_qrels(dataset)
    by_query: Dict[str, Dict[str, float]] = {}
    for qid, docid, score in read_pool(pool_path(dataset)):
        if qid in queries:
            by_query.setdefault(qid, {})[docid] = score
    qids = sorted(by_query)
    if max_queries:
        qids = qids[:max_queries]
    searcher = LuceneSearcher.from_prebuilt_index(dataset.index)
    results = []
    for qid in qids:
        cands = []
        for rank, (docid, score) in enumerate(sorted(by_query[qid].items(), key=lambda kv: -kv[1]), start=1):
            doc = searcher.doc(docid)
            cands.append(Candidate(docid=docid, rank=rank, score=score, text=doc_text(doc.raw()) if doc else ""))
        results.append(QueryCandidates(qid=qid, query=queries[qid], candidates=cands))
    params = {
        "retriever": "official",
        "pool_url": dataset.pool.url,
        "pool_depth": dataset.pool.depth,
        "index": dataset.index,
        "topics": dataset.topics,
        "qrels": dataset.qrels,
        "n_queries": len(qids),
        "n_pairs": sum(len(q.candidates) for q in results),
        "judged_queries_without_pool": len(set(queries) - set(by_query)),
    }
    return results, {q: qrels[q] for q in qids}, params

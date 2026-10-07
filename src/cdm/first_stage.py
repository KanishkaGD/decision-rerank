"""
The candidates every reranker sees, cached once per dataset (src/cdm/runs.py): the track's official rerank pool where
there is one (official_pool), else BM25 top-100 with Pyserini's prebuilt indexes (retrieve_bm25).
"""

import gzip
import json
import os
import shutil
import tarfile
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
    """Where `cdm prefetch` keeps a dataset's official pool: data/pools/<dataset>/<file>, gzipped."""
    assert dataset.pool is not None
    name = dataset.pool.member + ".gz" if dataset.pool.member else os.path.basename(dataset.pool.url)
    return os.path.join(POOLS_FOLDER, dataset.name, name)


def download_pool(dataset: Dataset) -> str:
    """Downloads the official pool once (login node: compute nodes may have no internet). A tar member is re-gzipped."""
    path = pool_path(dataset)
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with urllib.request.urlopen(dataset.pool.url) as response:
        if dataset.pool.member:
            with tarfile.open(fileobj=response, mode="r|gz") as tar:
                for member in tar:
                    if os.path.basename(member.name) == dataset.pool.member:
                        with gzip.open(tmp, "wb") as out:
                            shutil.copyfileobj(tar.extractfile(member), out)
                        break
                else:
                    raise FileNotFoundError(f"{dataset.pool.member} not in {dataset.pool.url}")
        else:
            with open(tmp, "wb") as out:
                shutil.copyfileobj(response, out)
    os.replace(tmp, path)
    return path


def read_pool(path: str, fmt: str) -> Iterator[Tuple[str, str, Optional[float], Optional[str]]]:
    """(qid, docid, track score or None, passage text or None) per line of an official pool file."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if fmt == "tsv":  # qid, pid, query, passage
                qid, docid, _, text = line.rstrip("\n").split("\t", 3)
                yield qid, docid, None, text
            elif fmt == "trec":  # qid Q0 docid rank score tag
                qid, _, docid, _, score, _ = line.split()
                yield qid, docid, float(score), None
            else:
                raise ValueError(f"Unknown pool format '{fmt}'")


def official_pool(
    dataset: Dataset, threads: int = 8, max_queries: Optional[int] = None
) -> Tuple[List[QueryCandidates], Qrels, Dict]:
    """
    The official rerank pool of the judged queries. An unranked pool ("tsv") is ordered by our BM25 score of each pair
    (Pyserini, BM25_PARAMS; ties by docid, as in runs.write_run). The scores come from a BM25 search of the same depth
    plus, for the few pool passages outside its results, Lucene's score of that single pair; the two agree on every
    pair both cover (checked here on up to 20 per query and recorded in the manifest). A ranked pool ("trec") keeps
    the track's order and scores.
    """
    from pyserini.search.lucene import LuceneSearcher

    pool = dataset.pool
    queries, qrels = load_topics_and_qrels(dataset)
    by_query: Dict[str, Dict[str, Tuple[Optional[float], Optional[str]]]] = {}
    for qid, docid, score, text in read_pool(pool_path(dataset), pool.fmt):
        if qid in queries:
            by_query.setdefault(qid, {})[docid] = (score, text)
    qids = sorted(by_query)
    if max_queries:
        qids = qids[:max_queries]
    searcher = LuceneSearcher.from_prebuilt_index(dataset.index)
    searcher.set_bm25(**BM25_PARAMS)

    checked, max_diff = 0, 0.0
    if pool.fmt == "tsv":
        from pyserini.index.lucene import LuceneIndexReader

        reader = LuceneIndexReader.from_prebuilt_index(dataset.index)
        hits = searcher.batch_search([queries[q] for q in qids], qids, k=pool.depth, threads=threads)
        for qid in qids:
            searched = {h.docid: float(h.score) for h in hits.get(qid, [])}
            docs = by_query[qid]
            for docid, (_, text) in docs.items():
                score = searched.get(docid)
                if score is None:
                    score = float(reader.compute_query_document_score(docid, queries[qid]))
                elif checked < 20 * len(qids):
                    diff = abs(score - float(reader.compute_query_document_score(docid, queries[qid])))
                    checked, max_diff = checked + 1, max(max_diff, diff)
                docs[docid] = (score, text)
        if max_diff > 1e-3:
            raise RuntimeError(f"BM25 search and single-pair scores differ by {max_diff:.4f}: different parameters?")

    results = []
    for qid in qids:
        if pool.fmt == "tsv":  # our BM25 scores; ties by docid
            ranked = sorted(by_query[qid].items(), key=lambda kv: (-kv[1][0], kv[0]))
        else:  # the track's ranking as given (file order; a stable sort keeps its tie order)
            ranked = sorted(by_query[qid].items(), key=lambda kv: -kv[1][0])
        cands = []
        for rank, (docid, (score, text)) in enumerate(ranked, start=1):
            if text is None:
                doc = searcher.doc(docid)
                text = doc_text(doc.raw()) if doc is not None else ""
            cands.append(Candidate(docid=docid, rank=rank, score=score, text=text))
        results.append(QueryCandidates(qid=qid, query=queries[qid], candidates=cands))
    params = {
        "retriever": "official",
        "pool_url": pool.url,
        "pool_format": pool.fmt,
        "pool_depth": pool.depth,
        "order": "bm25" if pool.fmt == "tsv" else "track",
        "index": dataset.index,
        "topics": dataset.topics,
        "qrels": dataset.qrels,
        **(BM25_PARAMS if pool.fmt == "tsv" else {}),
        "n_queries": len(qids),
        "n_pairs": sum(len(q.candidates) for q in results),
        "judged_queries_without_pool": len(set(queries) - set(by_query)),
        "bm25_pairs_checked": checked,
        "bm25_max_abs_diff": max_diff,
    }
    return results, {q: qrels[q] for q in qids}, params

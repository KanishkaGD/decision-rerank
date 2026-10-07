"""First-stage retrieval with Pyserini's prebuilt BM25 indexes (research plan: BM25 top-100, cached once)."""

import json
from typing import Dict, List, Optional, Tuple

from cdm.datasets import Dataset
from cdm.runs import Candidate, Qrels, QueryCandidates

# Pyserini's defaults for its prebuilt MS MARCO and BEIR (flat) BM25 runs; recorded in every manifest.
BM25_PARAMS = {"k1": 0.9, "b": 0.4}


def doc_text(raw: str) -> str:
    """Passage text from a Pyserini stored document: MS MARCO {"contents"}, BEIR {"title", "text"}, or plain text."""
    try:
        d = json.loads(raw)
    except (TypeError, ValueError):
        return raw or ""
    if not isinstance(d, dict):
        return str(d)
    if "contents" in d:
        return d["contents"]
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

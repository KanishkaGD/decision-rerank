"""
The first-stage candidate cache, so that every reranker sees exactly the same candidates (research plan:
"Every reranker sees the same cached first-stage candidates"). One directory per dataset and first stage:

  runs/<dataset>/<first-stage>/          e.g. runs/dl19/bm25-d100/
    candidates.jsonl.gz   one line per query: {"qid", "query", "candidates": [{"docid", "rank", "score", "text"}]}
    run.trec.gz           the same ranking as a TREC run file
    qrels.tsv             the judgments (qid, docid, relevance), so evaluation never needs Pyserini/Java
    manifest.json         how it was made (src/cdm/provenance.py)
"""

import gzip
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List

from cdm.config import RUNS_FOLDER


@dataclass
class Candidate:
    docid: str
    rank: int  # 1-based first-stage rank
    score: float  # first-stage score
    text: str


@dataclass
class QueryCandidates:
    qid: str
    query: str
    candidates: List[Candidate] = field(default_factory=list)


Qrels = Dict[str, Dict[str, int]]  # qid -> docid -> graded relevance


def first_stage_name(retriever: str, depth: int) -> str:
    return f"{retriever}-d{depth}"


def run_dir(dataset: str, first_stage: str) -> str:
    return os.path.join(RUNS_FOLDER, dataset, first_stage)


def write_candidates(path: str, queries: Iterable[QueryCandidates]) -> None:
    tmp = f"{path}.tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps({
                "qid": q.qid,
                "query": q.query,
                "candidates": [c.__dict__ for c in q.candidates],
            }, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_candidates(path: str) -> Iterator[QueryCandidates]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            yield QueryCandidates(d["qid"], d["query"], [Candidate(**c) for c in d["candidates"]])


# TREC run files: qid Q0 docid rank score tag
Run = Dict[str, Dict[str, float]]  # qid -> docid -> score


def write_run(path: str, run: Run, tag: str) -> None:
    tmp = f"{path}.tmp"
    opener = gzip.open if path.endswith(".gz") else open
    with opener(tmp, "wt", encoding="utf-8") as f:
        for qid, docs in run.items():
            ranked = sorted(docs.items(), key=lambda kv: (-kv[1], kv[0]))
            for rank, (docid, score) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {tag}\n")
    os.replace(tmp, path)


def read_run(path: str) -> Run:
    opener = gzip.open if path.endswith(".gz") else open
    run: Run = {}
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            qid, _, docid, _, score, _ = line.split()
            run.setdefault(qid, {})[docid] = float(score)
    return run


def write_qrels(path: str, qrels: Qrels) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for qid, docs in qrels.items():
            for docid, rel in docs.items():
                f.write(f"{qid}\t{docid}\t{rel}\n")


def read_qrels(path: str) -> Qrels:
    qrels: Qrels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, docid, rel = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels

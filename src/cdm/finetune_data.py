"""
Training data for our own reranker fine-tunes (docs/notes/reranker-27b-plan.md): MS MARCO passage train queries, each
with one judged positive and hard negatives from several public negative sets, written as training groups

  data/finetune/<name>/groups.jsonl.gz   one line per query: {"qid", "query", "pos": {"docid", "text"},
                                         "negs": [{"docid", "text", "source"}, ...]}
  data/finetune/<name>/manifest.json     sources (repo, file, revision), seed, sizes, what the leakage filter removed

Sources are Tevatron-format JSONL files on the Hugging Face Hub (query_id, query, positive_passages,
negative_passages), downloaded with `cdm prefetch --training-data ...` and read offline from HF_HOME. Passages are used
without their titles: the test passages (Pyserini's msmarco-v1/v2 and BEIR raw text as cached in runs/) are scored
from their text, and the reader sees exactly what training saw.

Leakage filter: a training query is dropped if its id or its normalised text (or its bag of words) equals that of any
query of a test collection or of msmarco-dev (docs/roadmap.md: never tune on test splits; msmarco-dev is for
selection). Ids are compared across all collections at once -- a coincidental id match in another id space costs one
training query, which is cheaper than reasoning about which id spaces overlap.
"""

import gzip
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from cdm.config import DATA_FOLDER


@dataclass(frozen=True)
class TrainingSource:
    name: str
    repo: str  # Hugging Face dataset repo
    filename: str
    revision: str  # pinned commit
    description: str


# Read 2026-10-09 from the Hub API. MS MARCO itself is under Microsoft's terms (non-commercial research use).
TRAINING_SOURCES: Dict[str, TrainingSource] = {
    s.name: s
    for s in [
        TrainingSource(
            "tevatron-aug",
            "Tevatron/msmarco-passage-aug",
            "train.jsonl.gz",
            "d2f82a9006454b081292437f8587cc00b5de024d",
            "MS MARCO passage train, 30 hard negatives per query mined with a dense retriever (coCondenser, per "
            "Tevatron's docs); 2.2 GB. The repo also has dl19/dl20 files: never read them.",
        ),
        TrainingSource(
            "tevatron-bm25",
            "Tevatron/msmarco-passage",
            "train.jsonl.gz",
            "330a160b40a739d4c07d52c7ce562e2c549214c0",
            "MS MARCO passage train, up to 30 BM25 negatives per query (the candidate distribution we rerank); "
            "1.7 GB, apache-2.0 card.",
        ),
    ]
}

FINETUNE_FOLDER = os.path.join(DATA_FOLDER, "finetune")


def groups_path(name: str) -> str:
    return os.path.join(FINETUNE_FOLDER, name, "groups.jsonl.gz")


def source_path(source: TrainingSource, local_only: bool = True) -> str:
    """The cached file (offline by default; `cdm prefetch --training-data` downloads it)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        source.repo,
        source.filename,
        repo_type="dataset",
        revision=source.revision,
        local_files_only=local_only,
    )


# ---------------------------------------------------------------------------------------------------------- leakage


def normalize_query(text: str) -> str:
    """Lower case, every run of non-alphanumeric characters -> one space, stripped."""
    return re.sub(r"[^0-9a-z]+", " ", text.lower()).strip()


def bag_of_words(text: str) -> str:
    """The normalised query's distinct words, sorted: catches reorderings ("cost of x" / "x cost of")."""
    return " ".join(sorted(set(normalize_query(text).split())))


@dataclass
class LeakageFilter:
    """Ids and texts of the queries no training example may share (the test collections and msmarco-dev)."""

    qids: Set[str]
    texts: Set[str]
    bags: Set[str]
    origin: Dict[str, str]  # normalised text or "qid:<id>" -> dataset, for the report

    @classmethod
    def from_queries(cls, queries: Iterable[Tuple[str, str, str]]) -> "LeakageFilter":
        """queries: (dataset, qid, text)."""
        f = cls(set(), set(), set(), {})
        for dataset, qid, text in queries:
            f.qids.add(str(qid))
            f.texts.add(normalize_query(text))
            f.bags.add(bag_of_words(text))
            f.origin.setdefault(f"qid:{qid}", dataset)
            f.origin.setdefault(normalize_query(text), dataset)
            f.origin.setdefault(f"bow:{bag_of_words(text)}", dataset)
        return f

    def reason(self, qid: str, text: str) -> Optional[str]:
        """Why this training query must go ("qid:<dataset>", "text:<dataset>", "bow:<dataset>"), or None."""
        if str(qid) in self.qids:
            return f"qid:{self.origin.get(f'qid:{qid}', '?')}"
        norm = normalize_query(text)
        if norm in self.texts:
            return f"text:{self.origin.get(norm, '?')}"
        bow = bag_of_words(text)
        if bow in self.bags:
            return f"bow:{self.origin.get(f'bow:{bow}', '?')}"
        return None


def evaluation_queries(datasets: Sequence[str]) -> Iterator[Tuple[str, str, str]]:
    """
    (dataset, qid, text) of every query of the given datasets: all topics of the dataset's Pyserini topic file, judged
    or not (msmarco-dev: all 6,980), plus any query in its cached candidate sets (runs/<dataset>/*/) -- so the filter
    covers more than we evaluate on, never less. Fails if a topic file can't be read: no dataset is skipped silently.
    """
    from pyserini.search import get_topics

    from cdm.datasets import get_dataset
    from cdm.runs import RUNS_FOLDER, read_candidates

    for name in datasets:
        ds = get_dataset(name)
        seen = set()
        for qid, topic in get_topics(ds.topics).items():
            seen.add(str(qid))
            yield name, str(qid), topic["title"]
        parent = os.path.join(RUNS_FOLDER, name)
        for d in sorted(os.listdir(parent)) if os.path.isdir(parent) else []:
            path = os.path.join(parent, d, "candidates.jsonl.gz")
            if os.path.exists(path):
                for q in read_candidates(path):
                    if q.qid not in seen:
                        seen.add(q.qid)
                        yield name, q.qid, q.query


# ---------------------------------------------------------------------------------------------------------- groups


_QID = re.compile(r'"query_id"\s*:\s*"([^"]*)"')


def line_qid(line: str) -> Optional[str]:
    """The query id of a Tevatron JSONL line without parsing the whole line (its passages are most of it)."""
    m = _QID.search(line[:200])
    return m.group(1) if m else None


def select_qids(qids: Iterable[str], n: Optional[int], seed: int) -> List[str]:
    """
    A deterministic sample of n ids (all if n is None), ranked by a seeded hash: the same seed picks the same queries
    whatever the file order, and a larger n is a superset of a smaller one (nested training sets for a learning
    curve).
    """

    def key(qid: str) -> str:
        return hashlib.sha1(f"{seed}:{qid}".encode()).hexdigest()

    ranked = sorted(set(qids), key=key)
    return ranked if n is None else ranked[:n]


def build_group(
    qid: str,
    query: str,
    positives: List[Dict[str, str]],
    negatives: Dict[str, List[Dict[str, str]]],
    quota: Dict[str, int],
    rng: random.Random,
) -> Optional[Dict]:
    """
    One training group: one positive drawn from the judged ones, and per source `quota[source]` negatives drawn
    without replacement. A negative that is a positive of the query, or already drawn from another source, is skipped;
    a source short of negatives is topped up from the others (in quota order), so every group has the same size.
    Returns None if the query has no positive or too few distinct negatives overall.
    """
    if not positives:
        return None
    pos = rng.choice(positives)
    pos_ids = {p["docid"] for p in positives}
    want = sum(quota.values())
    taken: List[Dict[str, str]] = []
    used = set(pos_ids)
    leftovers: List[Dict[str, str]] = []
    for source, k in quota.items():
        pool = [p for p in negatives.get(source, []) if p["docid"] not in pos_ids]
        rng.shuffle(pool)
        got = 0
        for p in pool:
            if p["docid"] in used:
                continue
            if got < k:
                taken.append({"docid": p["docid"], "text": p["text"], "source": source})
                used.add(p["docid"])
                got += 1
            else:
                leftovers.append({"docid": p["docid"], "text": p["text"], "source": source})
    for p in leftovers:
        if len(taken) >= want:
            break
        if p["docid"] not in used:
            taken.append(p)
            used.add(p["docid"])
    if len(taken) < want:
        return None
    return {"qid": qid, "query": query, "pos": {"docid": pos["docid"], "text": pos["text"]}, "negs": taken}


def parse_quota(spec: str) -> Dict[str, int]:
    """'tevatron-aug:8,tevatron-bm25:7' -> {'tevatron-aug': 8, 'tevatron-bm25': 7} (order kept)."""
    quota: Dict[str, int] = {}
    for part in filter(None, spec.split(",")):
        name, _, k = part.partition(":")
        if name not in TRAINING_SOURCES:
            raise KeyError(f"Unknown training source '{name}'. Known: {', '.join(TRAINING_SOURCES)}")
        quota[name] = int(k)
    if not quota or any(k < 0 for k in quota.values()):
        raise ValueError(f"Bad negative quota '{spec}'")
    return quota


def _passages(entries: List[Dict]) -> List[Dict[str, str]]:
    return [{"docid": str(e["docid"]), "text": e["text"]} for e in entries]


def read_source(path: str, keep: Set[str]) -> Iterator[Dict]:
    """The parsed lines of a Tevatron JSONL(.gz) file whose query id is in `keep`."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            qid = line_qid(line)
            if qid is None or qid in keep:
                d = json.loads(line)
                if str(d["query_id"]) in keep:
                    yield d


def source_qids(path: str) -> Iterator[Tuple[str, str]]:
    """(qid, query) of every line; the query text is needed for the leakage filter, so the line is parsed."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            # The query comes before the passages: parse only the head of the line.
            head = line[: line.find('"positive_passages"')] if '"positive_passages"' in line else line
            m_id = _QID.search(head)
            m_q = re.search(r'"query"\s*:\s*("(?:[^"\\]|\\.)*")', head)
            if m_id and m_q:
                yield m_id.group(1), json.loads(m_q.group(1))
            else:
                d = json.loads(line)
                yield str(d["query_id"]), d["query"]


def build_groups(
    paths: Dict[str, str],
    quota: Dict[str, int],
    leakage: LeakageFilter,
    n_queries: Optional[int],
    seed: int,
) -> Tuple[List[Dict], Dict]:
    """
    Training groups from the sources in `paths` (source name -> file). Candidate queries are those of the first
    source in `quota` that pass the leakage filter and appear in every source with a quota > 0; n_queries of them are
    sampled (select_qids), then the groups built. Returns (groups, stats).
    """
    order = list(quota)
    first = order[0]
    queries: Dict[str, str] = {}
    removed: Dict[str, int] = {}
    removed_examples: List[str] = []
    for qid, text in source_qids(paths[first]):
        why = leakage.reason(qid, text)
        if why:
            removed[why] = removed.get(why, 0) + 1
            if len(removed_examples) < 20:
                removed_examples.append(f"{qid}\t{why}\t{text}")
            continue
        queries[qid] = text
    eligible = set(queries)
    n_first = len(queries) + sum(removed.values())
    for name in order[1:]:
        if quota[name] > 0:
            eligible &= {qid for qid, _ in source_qids(paths[name])}
    # Oversample a little: a few queries fail build_group (no positive or too few negatives).
    picked = select_qids(eligible, None if n_queries is None else int(n_queries * 1.02) + 10, seed)
    keep = set(picked)
    positives: Dict[str, List[Dict[str, str]]] = {}
    negatives: Dict[str, Dict[str, List[Dict[str, str]]]] = {qid: {} for qid in keep}
    for name in order:
        for d in read_source(paths[name], keep):
            qid = str(d["query_id"])
            if name == first:
                positives[qid] = _passages(d["positive_passages"])
            negatives[qid][name] = _passages(d["negative_passages"])
    rng = random.Random(seed)
    groups = []
    failed = 0
    for qid in picked:
        g = build_group(qid, queries[qid], positives.get(qid, []), negatives[qid], quota, rng)
        if g is None:
            failed += 1
            continue
        groups.append(g)
        if n_queries is not None and len(groups) == n_queries:
            break
    stats = {
        "queries_in_first_source": n_first,
        "removed_by_leakage_filter": removed,
        "removed_examples": removed_examples,
        "eligible": len(eligible),
        "groups": len(groups),
        "failed_groups": failed,
        "group_size": 1 + sum(quota.values()),
    }
    return groups, stats


def write_groups(path: str, groups: Iterable[Dict]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    n = 0
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for g in groups:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def read_groups(path: str) -> List[Dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

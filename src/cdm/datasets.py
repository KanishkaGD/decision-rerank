"""
The datasets of the study, by short name, with the Pyserini resources that provide their corpus index, queries and
judgments, and the official rerank pool where the track published one. Adding a dataset: add it here, give it a
--time budget in slurm/config.sh, prefetch it.

Candidates (owner's decision, 2026-10-07): rerank the official pool where one exists, so the numbers compare with the
track's reranking runs and with Kanishka's DL19/20 results; build our own BM25 top-100 only where none exists.
Official pools exist only for the MS MARCO-based TREC DL tracks and MS MARCO dev; BEIR publishes none (its de-facto
standard is Pyserini's BM25 flat top-100, which is what `cdm first-stage --retriever bm25` builds).

Pyserini names are from its prebuilt-index and topic registries (`python -m pyserini.search.lucene --help`,
pyserini.prebuilt_index_info); WP0/WP1 should confirm them against the installed version (`cdm prefetch` fails
loudly on an unknown name).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

MSMARCO_RANKING_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking"


@dataclass(frozen=True)
class OfficialPool:
    """
    The candidate set a track gave its reranking subtask. Formats: "tsv" = qid, pid, query, passage (MS MARCO v1
    top1000 files; unranked, so first_stage.official_pool orders them by our BM25 score); "trec" = a ranked TREC run
    (DL 2021-2023 top100; passage texts then come from the index).
    """

    url: str
    depth: int  # the most candidates per query the track gave
    fmt: str = "tsv"
    member: Optional[str] = None  # the file inside a .tar.gz download


@dataclass(frozen=True)
class Dataset:
    name: str
    index: str  # Pyserini prebuilt Lucene index (BM25, with stored raw documents)
    topics: str  # Pyserini topics
    qrels: str  # Pyserini qrels
    # Lowest graded label counted as relevant by binary measures (Recall, AP, RR) and by calibration (ECE, Brier).
    # TREC DL passage judgments are 0-3 with "relevant" from 2 on; BEIR's from 1 on.
    rel_threshold: int
    split: str  # "test" (reported) or "dev" (only for selecting configurations, never reported as a result)
    group: str  # "trec-dl" | "beir" | "msmarco"
    pool: Optional[OfficialPool] = None

    @property
    def default_first_stage(self) -> str:
        """The candidate set reranked unless --first-stage says otherwise: the official pool if there is one."""
        from cdm.config import FIRST_STAGE_DEPTH
        from cdm.runs import first_stage_name

        if self.pool:
            return first_stage_name("official", self.pool.depth)
        return first_stage_name("bm25", FIRST_STAGE_DEPTH)


def _beir(name: str) -> Dataset:
    return Dataset(
        name=name,
        index=f"beir-v1.0.0-{name}.flat",
        topics=f"beir-v1.0.0-{name}-test",
        qrels=f"beir-v1.0.0-{name}-test",
        rel_threshold=1,
        split="test",
        group="beir",
    )


def _dl_v2(year: int) -> Dataset:
    name = f"dl{year % 100}"
    return Dataset(
        name=name,
        index="msmarco-v2-passage",
        topics=name,
        qrels=f"{name}-passage",
        rel_threshold=2,
        split="test",
        group="trec-dl",
        pool=OfficialPool(f"{MSMARCO_RANKING_URL}/{year}_passage_top100.txt.gz", 100, fmt="trec"),
    )


DATASETS: Dict[str, Dataset] = {
    d.name: d
    for d in [
        # The official rerank-subtask pools of TREC DL 2019/2020 (up to 1000 BM25 candidates per query), as in
        # Kanishka's study; MS MARCO's top1000.dev the same for the dev queries (687 MB download).
        Dataset(
            "dl19",
            "msmarco-v1-passage",
            "dl19-passage",
            "dl19-passage",
            2,
            "test",
            "trec-dl",
            OfficialPool(f"{MSMARCO_RANKING_URL}/msmarco-passagetest2019-top1000.tsv.gz", 1000),
        ),
        Dataset(
            "dl20",
            "msmarco-v1-passage",
            "dl20",
            "dl20-passage",
            2,
            "test",
            "trec-dl",
            OfficialPool(f"{MSMARCO_RANKING_URL}/msmarco-passagetest2020-top1000.tsv.gz", 1000),
        ),
        Dataset(
            "msmarco-dev",
            "msmarco-v1-passage",
            "msmarco-passage-dev-subset",
            "msmarco-passage-dev-subset",
            1,
            "dev",
            "msmarco",
            OfficialPool(f"{MSMARCO_RANKING_URL}/top1000.dev.tar.gz", 1000, member="top1000.dev"),
        ),
        # TREC DL 2021-2023: MS MARCO v2 passage (138M passages), official top-100 pools (ranked TREC runs of BM25,
        # checked 2026-10-07). The passage texts come from Pyserini's msmarco-v2-passage index (39 GB, stores the
        # raw passages; prefetch it with slurm/prefetch.sbatch). Topic and qrels names as in Pyserini's 2CR.
        *[_dl_v2(year) for year in (2021, 2022, 2023)],
        _beir("trec-covid"),
        _beir("nfcorpus"),
        _beir("scifact"),
        _beir("fiqa"),
        _beir("scidocs"),
        # Planned, not yet wired up: MIRACL (multilingual checkpoint of Laya; open decision) and BRIGHT (RQ5).
    ]
}


def get_dataset(name: str) -> Dataset:
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(f"Unknown dataset '{name}'. Known: {', '.join(DATASETS)}") from None


def dataset_names(split: str = "test") -> List[str]:
    return [d.name for d in DATASETS.values() if d.split == split]

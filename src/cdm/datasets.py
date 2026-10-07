"""
The datasets of the study, by short name, with the Pyserini resources that provide their corpus index, queries and
judgments. Adding a dataset: add it here, give it a --time budget in slurm/config.sh, prefetch it.

Candidates (docs/roadmap.md, "Setup"): every dataset is reranked on the BM25 top-100 of Pyserini's default BM25
(k1 0.9, b 0.4; BEIR "flat" indexes) -- the candidate set S1Rank, RankGPT, RankZephyr and the comparison study of
Abdallah et al. (2025) rerank, so their published numbers sit in the same table as ours. For TREC DL 2021-2023 the
track's official top-100 files are exactly this set (their BM25 nDCG@10 equals Pyserini's 2CR scores); we read them
instead of searching the 39 GB index. Everywhere else `cdm first-stage` searches the index.

Pyserini names are from its prebuilt-index and topic registries (`python -m pyserini.search.lucene --help`,
pyserini.prebuilt_index_info, pyserini/2cr/*.yaml); `cdm prefetch` fails loudly on an unknown name.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

MSMARCO_RANKING_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking"


@dataclass(frozen=True)
class OfficialPool:
    """A track's official reranking candidates as a ranked TREC run file (DL 2021-2023 top100)."""

    url: str
    depth: int  # candidates per query


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
        """The candidate set reranked unless --first-stage says otherwise (always BM25 top-100, see above)."""
        from cdm.config import FIRST_STAGE_DEPTH
        from cdm.runs import first_stage_name

        if self.pool:
            return first_stage_name("official", self.pool.depth)
        return first_stage_name("bm25", FIRST_STAGE_DEPTH)


def _beir(name: str, beir_id: Optional[str] = None) -> Dataset:
    beir_id = beir_id or name
    return Dataset(
        name=name,
        index=f"beir-v1.0.0-{beir_id}.flat",
        topics=f"beir-v1.0.0-{beir_id}-test",
        qrels=f"beir-v1.0.0-{beir_id}-test",
        rel_threshold=1,
        split="test",
        group="beir",
    )


def _dl_v1(name: str, topics: str) -> Dataset:
    return Dataset(name, "msmarco-v1-passage", topics, f"{name}-passage", 2, "test", "trec-dl")


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
        pool=OfficialPool(f"{MSMARCO_RANKING_URL}/{year}_passage_top100.txt.gz", 100),
    )


DATASETS: Dict[str, Dataset] = {
    d.name: d
    for d in [
        # TREC DL 2019/2020 on MS MARCO v1 passage.
        _dl_v1("dl19", "dl19-passage"),
        _dl_v1("dl20", "dl20"),
        # TREC DL 2021-2023 on MS MARCO v2 passage (138M passages); passage texts from the msmarco-v2-passage index
        # (39 GB, prefetch it with slurm/prefetch.sbatch).
        *[_dl_v2(year) for year in (2021, 2022, 2023)],
        # The public BEIR sets with published RankGPT/RankZephyr/monoT5 numbers (docs/roadmap.md).
        _beir("trec-covid"),
        _beir("nfcorpus"),
        _beir("scifact"),
        _beir("touche", "webis-touche2020"),
        _beir("dbpedia", "dbpedia-entity"),
        # Development split (fine-tuning, docs/roadmap.md); never reported as a result.
        Dataset(
            "msmarco-dev",
            "msmarco-v1-passage",
            "msmarco-passage-dev-subset",
            "msmarco-passage-dev-subset",
            1,
            "dev",
            "msmarco",
        ),
    ]
}


def get_dataset(name: str) -> Dataset:
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(f"Unknown dataset '{name}'. Known: {', '.join(DATASETS)}") from None


def dataset_names(split: str = "test") -> List[str]:
    return [d.name for d in DATASETS.values() if d.split == split]

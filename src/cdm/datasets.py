"""
The datasets of the study, by short name, with the Pyserini resources that provide their corpus index, queries and
judgments. Adding a dataset: add it here, give it a --time budget in slurm/config.sh, prefetch it.

Pyserini names are from its prebuilt-index and topic registries (`python -m pyserini.search.lucene --help`,
pyserini.prebuilt_index_info); WP0/WP1 should confirm them against the installed version (`cdm prefetch` fails
loudly on an unknown name).
"""

from dataclasses import dataclass
from typing import Dict, List


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


DATASETS: Dict[str, Dataset] = {
    d.name: d
    for d in [
        Dataset("dl19", "msmarco-v1-passage", "dl19-passage", "dl19-passage", 2, "test", "trec-dl"),
        Dataset("dl20", "msmarco-v1-passage", "dl20", "dl20-passage", 2, "test", "trec-dl"),
        Dataset(
            "msmarco-dev",
            "msmarco-v1-passage",
            "msmarco-passage-dev-subset",
            "msmarco-passage-dev-subset",
            1,
            "dev",
            "msmarco",
        ),
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

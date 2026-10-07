"""Candidate sets: the official DL 2021-2023 pools, and the BM25 top-100 default everywhere else."""

from cdm.datasets import DATASETS
from cdm.first_stage import doc_text, pool_path, read_pool


def test_read_pool(tmp_path):
    trec = tmp_path / "pool.txt"
    trec.write_text("2082 Q0 msmarco_passage_45_623131157 1 19.820700 Anserini\n")
    assert list(read_pool(str(trec))) == [("2082", "msmarco_passage_45_623131157", 19.8207)]


def test_default_first_stage():
    for name in ["dl19", "dl20", "trec-covid", "nfcorpus", "scifact", "touche", "dbpedia", "msmarco-dev"]:
        assert DATASETS[name].default_first_stage == "bm25-d100", name
    for name in ["dl21", "dl22", "dl23"]:
        assert DATASETS[name].default_first_stage == "official-d100", name


def test_dl_v2_datasets():
    for year in ("21", "22", "23"):
        ds = DATASETS[f"dl{year}"]
        assert (ds.index, ds.topics, ds.qrels, ds.rel_threshold) == (
            "msmarco-v2-passage",
            f"dl{year}",
            f"dl{year}-passage",
            2,
        )
        assert ds.pool.url.endswith(f"20{year}_passage_top100.txt.gz")
        assert pool_path(ds).endswith(f"pools/dl{year}/20{year}_passage_top100.txt.gz")


def test_beir_ids():
    assert DATASETS["touche"].index == "beir-v1.0.0-webis-touche2020.flat"
    assert DATASETS["dbpedia"].qrels == "beir-v1.0.0-dbpedia-entity-test"
    assert DATASETS["dl20"].qrels == "dl20-passage"


def test_doc_text_v2():
    assert doc_text('{"pid": "msmarco_passage_00_0", "passage": "text", "spans": "(0,4)"}') == "text"

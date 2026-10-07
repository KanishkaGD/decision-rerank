"""Official rerank pools: file formats, the default first stage per dataset, and the rerank CLI's default cache."""

import gzip

from cdm.datasets import DATASETS
from cdm.first_stage import pool_path, read_pool


def test_read_pool_tsv_and_trec(tmp_path):
    tsv = tmp_path / "pool.tsv.gz"
    with gzip.open(tsv, "wt") as f:
        f.write("1\t7\tq one\tpassage\twith a tab\n2\t8\tq two\tother\n")
    assert list(read_pool(str(tsv), "tsv")) == [("1", "7", None, "passage\twith a tab"), ("2", "8", None, "other")]

    trec = tmp_path / "pool.txt"
    trec.write_text("2082 Q0 msmarco_passage_45_623131157 1 19.820700 Anserini\n")
    assert list(read_pool(str(trec), "trec")) == [("2082", "msmarco_passage_45_623131157", 19.8207, None)]


def test_default_first_stage():
    assert DATASETS["dl19"].default_first_stage == "official-d1000"
    assert DATASETS["dl20"].default_first_stage == "official-d1000"
    assert DATASETS["scifact"].default_first_stage == "bm25-d100"


def test_pool_path_of_tar_member():
    assert pool_path(DATASETS["msmarco-dev"]).endswith("pools/msmarco-dev/top1000.dev.gz")
    assert pool_path(DATASETS["dl19"]).endswith("pools/dl19/msmarco-passagetest2019-top1000.tsv.gz")


def test_dl_v2_datasets():
    for year in ("21", "22", "23"):
        ds = DATASETS[f"dl{year}"]
        assert (ds.index, ds.topics, ds.qrels, ds.rel_threshold) == (
            "msmarco-v2-passage",
            f"dl{year}",
            f"dl{year}-passage",
            2,
        )
        assert ds.pool.fmt == "trec" and ds.pool.url.endswith(f"20{year}_passage_top100.txt.gz")
        assert ds.default_first_stage == "official-d100"


def test_doc_text_v2():
    from cdm.first_stage import doc_text

    assert doc_text('{"pid": "msmarco_passage_00_0", "passage": "text", "spans": "(0,4)"}') == "text"

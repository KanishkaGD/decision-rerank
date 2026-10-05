import os

from cdm.first_stage import doc_text
from cdm.runs import Candidate, QueryCandidates, read_candidates, read_qrels, read_run, write_candidates, write_qrels, write_run


def test_roundtrip(tmp_path):
    qs = [QueryCandidates("1", "what is x", [Candidate("d1", 1, 12.5, "x is ü"), Candidate("d2", 2, 10.0, "y")])]
    path = os.path.join(tmp_path, "candidates.jsonl.gz")
    write_candidates(path, qs)
    assert list(read_candidates(path)) == qs

    run = {"1": {"d1": 0.2, "d2": 0.9}}
    write_run(os.path.join(tmp_path, "run.trec.gz"), run, tag="t")
    assert read_run(os.path.join(tmp_path, "run.trec.gz")) == {"1": {"d1": 0.2, "d2": 0.9}}

    qrels = {"1": {"d1": 2, "d3": 0}}
    write_qrels(os.path.join(tmp_path, "qrels.tsv"), qrels)
    assert read_qrels(os.path.join(tmp_path, "qrels.tsv")) == qrels


def test_doc_text_formats():
    assert doc_text('{"id": "1", "contents": "passage"}') == "passage"
    assert doc_text('{"_id": "1", "title": "T", "text": "body"}') == "T\nbody"
    assert doc_text('{"_id": "1", "title": "", "text": "body"}') == "body"
    assert doc_text("plain") == "plain"

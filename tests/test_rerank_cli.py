"""End-to-end `cdm rerank` on a tiny cached candidate set, with a fake reranker (no model download)."""

import importlib
import os

import pytest

pytest.importorskip("ir_measures")


def test_rerank_end_to_end(tmp_path, monkeypatch):
    for var, sub in [("CDM_RUNS", "runs"), ("CDM_OUTPUTS", "outputs"), ("CDM_RESULTS", "results")]:
        monkeypatch.setenv(var, str(tmp_path / sub))
    import cdm.config
    import cdm.results_csv
    import cdm.runs

    for m in (cdm.config, cdm.runs, cdm.results_csv):
        importlib.reload(m)
    import cdm.cli.rerank as rr

    importlib.reload(rr)
    from click.testing import CliRunner

    from cdm.rerankers.base import Reranker, Scores
    from cdm.runs import Candidate, QueryCandidates, write_candidates, write_qrels

    class Fake(Reranker):
        name, model_id, prob_kind = "minilm-l6", "fake", "sigmoid(logit)"

        def score(self, query, passages, phrasing):
            s = [1.0 if "good" in p else 0.0 for p in passages]
            return Scores(scores=s, probs=[0.9 if x else 0.1 for x in s])

    monkeypatch.setattr(rr, "load_reranker", lambda name, **kw: Fake())
    cache = tmp_path / "runs" / "scifact" / "bm25-d100"
    cache.mkdir(parents=True)
    write_candidates(str(cache / "candidates.jsonl.gz"), [
        QueryCandidates("q1", "query", [Candidate("bad", 1, 2.0, "bad"), Candidate("good", 2, 1.0, "good")]),
    ])
    write_qrels(str(cache / "qrels.tsv"), {"q1": {"good": 1, "bad": 0}})

    result = CliRunner().invoke(rr.rerank, ["minilm-l6", "scifact"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    df = cdm.results_csv.load_results()
    assert float(df["nDCG@10"].iloc[0]) == 1.0
    assert float(df["n_judged_nonrel_pairs"].iloc[0]) == 1
    assert os.path.exists(tmp_path / "outputs" / "scifact" / "bm25-d100" / "minilm-l6" / "none" / "scores.jsonl.gz")

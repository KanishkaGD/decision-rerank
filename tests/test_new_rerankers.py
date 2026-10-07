"""The rerankers added from Kanishka's study: registry entries, the LLM judge's readout, Kev's answer mapping."""

import math
import sys
import types

import pytest

from cdm.phrasings import get_phrasing
from cdm.rerankers import SYSTEMS, _factory
from cdm.rerankers.llm_judge import OpenRouterJudge, yes_probability


def test_registry():
    for name in ["monot5-3b", "qwen3-reranker-8b", "kev-4b-isolated", "qwen3.7-flash", "set-encoder-large"]:
        assert name in SYSTEMS
    assert SYSTEMS["kev-4b-isolated"].params == {"backend": "kev", "window": 1}
    assert SYSTEMS["kev-27b-joint"].params["window"] == 0
    for base in ["jev", "clef-flash", "clef", "kev-4b", "kev-27b"]:
        assert SYSTEMS[f"{base}-isolated"].params["window"] == 1 and SYSTEMS[f"{base}-joint"].params["window"] == 0
    assert not SYSTEMS["jev-joint"].hf and SYSTEMS["kev-4b-joint"].revision
    new = [s for s in SYSTEMS.values() if s.kind in ("monot5", "qwen3-reranker", "set-encoder") or "kev-" in s.name]
    assert all(s.revision for s in new), "the new Hugging Face models are pinned"
    for spec in SYSTEMS.values():
        assert callable(_factory(spec.kind))


def test_yes_probability():
    p = yes_probability([["yes", math.log(0.6)], [" Yes", math.log(0.1)], ["no", math.log(0.2)], ["maybe", -1.0]])
    assert p == pytest.approx(0.7 / 0.9)
    assert yes_probability([["maybe", -0.1]]) is None


def test_llm_judge_scores_and_breaks_ties(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    judge = OpenRouterJudge("qwen3.7-flash", "qwen/qwen3.7-flash")
    answers = {"a": [["yes", 0.0]], "b": [["yes", 0.0]], "c": [["other", 0.0]]}

    def fake_call(prompt):
        passage = prompt.split("Passage: ")[1].split("\n")[0]
        return {"top": answers[passage], "cost": 0.001, "input_tokens": 5}

    monkeypatch.setattr(judge, "_call", fake_call)
    out = judge.score("q", ["a", "b", "c"], get_phrasing("none"))
    assert out.probs == [1.0, 1.0, 0.5]
    assert out.scores[0] > out.scores[1]  # tie broken by first-stage rank
    assert judge.run_stats()["no_yes_no_answer"] == 1
    assert judge.run_stats()["api_cost_usd"] == pytest.approx(0.003)


def test_kev_backend_maps_noul_answers(monkeypatch):
    """KevBackend with a stand-in `kev` package: request encoding, the [no, yes] readout, too-long handling."""
    from cdm.rerankers import system_one

    class ContextOverflow(ValueError):
        pass

    class Model:
        def probs_batch(self, encs, prefixes, keep):
            return [[[0.3, 0.7] for _ in e["questions"]] for e in encs], [None] * len(encs)

    class Checkpoint:
        def __init__(self, ref):
            self.ref, self.path = ref, "/cache/kev"
            self.meta = types.SimpleNamespace(
                base="Qwen/base", base_revision="r", temperature=2.0, weights_dtype="bf16"
            )

        def load(self, device, opts):
            return "tok", Model()

    def to_record(req):
        return {"state": req.state, "questions": list(req.questions)}, [{"id": q} for q in req.questions]

    def admit(model, tok, rec, truncate=False):
        if "long" in str(rec["state"]) and not truncate:
            raise ContextOverflow()
        return {"ids": [0] * 10, "questions": rec["questions"], "state_truncated": "long" in str(rec["state"])}

    kev = types.ModuleType("kev")
    modules = {
        "kev": kev,
        "kev.checkpoint": types.SimpleNamespace(Checkpoint=Checkpoint, LoadOptions=lambda **kw: kw),
        "kev.api": types.SimpleNamespace(SystemOneRequest=types.SimpleNamespace, to_record=to_record),
        "kev.model": types.SimpleNamespace(ContextOverflow=ContextOverflow, admit=admit),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    pytest.importorskip("torch")

    r = system_one.SystemOneReranker("kev-4b-joint", "jaredpalmer/kev-4b", backend="kev", window=0, revision="abc")
    out = r.score("q", ["a", "b", "long"], get_phrasing("s1rank"))
    assert out.probs == [0.7, 0.7, 0.7]
    assert r.run_stats()["window_splits"] == 2  # 3 -> (1, 2) -> (1, 1): the long passage alone is cut
    assert r.count_truncated("q", []) == 1
    assert r.backend.describe()["checkpoint"] == "jaredpalmer/kev-4b@abc"

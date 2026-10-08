"""The rerankers added from Kanishka's study and the triplets: registry entries, Kev's answer mapping."""

import sys
import types

import pytest

from cdm.phrasings import get_phrasing
from cdm.rerankers import SYSTEMS, _factory


def test_registry():
    for name in ["monot5-3b", "qwen3-reranker-8b", "kev-4b-isolated", "set-encoder-large", "prism-9b",
                 "qwen3-8b-prompted", "kev-8b-joint"]:
        assert name in SYSTEMS
    assert SYSTEMS["kev-4b-isolated"].params == {"backend": "kev", "window": 1}
    assert SYSTEMS["kev-27b-joint"].params["window"] == 0
    for base in ["jev", "clef-flash", "clef", "kev-4b", "kev-27b"]:
        assert SYSTEMS[f"{base}-isolated"].params["window"] == 1 and SYSTEMS[f"{base}-joint"].params["window"] == 0
    assert not SYSTEMS["jev-joint"].hf and SYSTEMS["kev-4b-joint"].revision
    new = [
        s
        for s in SYSTEMS.values()
        if s.kind in ("monot5", "qwen3-reranker", "set-encoder", "prompted", "listwise") or "kev-" in s.name
    ]
    assert all(s.revision for s in new), "the new Hugging Face models are pinned"
    for spec in SYSTEMS.values():
        assert callable(_factory(spec.kind))


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

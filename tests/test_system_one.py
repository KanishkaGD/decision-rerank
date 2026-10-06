"""SystemOneReranker: request format, joint vs isolated, splitting, truncation; the TypeSafe backend with a fake SDK."""

import sys
import types

import pytest

from cdm.phrasings import get_phrasing
from cdm.rerankers import system_one


class FakeBackend:
    """Answers each Noul with 0.01 * its label number; too long above max_docs documents (cut if allowed)."""

    price_per_m_input = 0.0
    max_docs = 1000

    def __init__(self, model_id, **params):
        self.model_id = model_id
        self.params = params
        self.calls = []

    def answer(self, requests, truncate):
        self.calls.append((requests, truncate))
        out = []
        for (state, questions), cut in zip(requests, truncate):
            long = any("long" in d for d in state["documents"].values()) or len(questions) > self.max_docs
            if long and not cut:
                out.append(system_one.TOO_LONG)
                continue
            answers = {k: {"type": "noul", "noul": 0.01 * int(k[1:])} for k in questions}
            out.append(
                {
                    "model": self.model_id,
                    "answers": answers,
                    "usage": {"input_tokens": 10 * len(questions), "truncated": long},
                }
            )
        return out

    def describe(self):
        return {"backend": "fake"}


@pytest.fixture
def fake(monkeypatch):
    FakeBackend.max_docs = 1000
    monkeypatch.setitem(system_one.BACKENDS, "fake", FakeBackend)
    return lambda window: system_one.SystemOneReranker("s", "m", backend="fake", window=window)


def test_joint_request_is_s1rank_format(fake):
    r = fake(0)
    out = r.score("what is x", ["a", "b", "c"], get_phrasing("s1rank"))
    ((state, questions),), truncate = r.backend.calls[0]
    assert state == {"query": "what is x", "documents": {"D1": "a", "D2": "b", "D3": "c"}}
    assert questions["D2"] == {
        "type": "noul",
        "instructions": "Is document `documents.D2` relevant to the `query`? "
        "Relevant means it answers the query or contains information that directly addresses it.",
    }
    assert truncate == [False]
    assert out.probs == [0.01, 0.02, 0.03]
    assert r.run_stats() == {"requests": 1, "input_tokens": 30, "window_splits": 0, "truncated_passages": 0}
    assert r.joint and r.describe()["window"] == 0


def test_isolated_one_request_per_passage(fake):
    r = fake(1)
    out = r.score("q", ["a", "b", "c"], get_phrasing("p1-relevant"))
    requests, truncate = r.backend.calls[0]
    assert [list(state["documents"]) for state, _ in requests] == [["D1"]] * 3
    assert truncate == [True, True, True]  # a single passage may be cut
    assert out.probs == [0.01, 0.01, 0.01]
    assert not r.joint


def test_ties_broken_by_first_stage_rank(fake):
    r = fake(1)
    out = r.score("q", ["a", "b", "c"], get_phrasing("s1rank"))
    assert out.scores[0] > out.scores[1] > out.scores[2]


def test_too_long_splits_in_half(fake):
    FakeBackend.max_docs = 2
    r = fake(0)
    out = r.score("q", ["a", "b", "c", "d", "e"], get_phrasing("s1rank"))
    # 5 -> 2 + 3 -> 2 + (1 + 2); each part is labelled from D1 again
    assert out.probs == [0.01, 0.02, 0.01, 0.01, 0.02]
    assert r.run_stats()["window_splits"] == 2
    assert [(rec["offset"], rec["n"]) for rec in out.raw] == [(0, 2), (2, 1), (3, 2)]


def test_long_passage_is_isolated_then_cut_and_counted(fake):
    r = fake(0)
    out = r.score("q", ["a", "long", "c"], get_phrasing("s1rank"))
    # 3 -> 1 + 2 -> 1 + (1 + 1): "long" ends up alone and is cut
    assert out.probs == [0.01, 0.01, 0.01]
    assert r.count_truncated("q", []) == 1 and r.run_stats()["truncated_passages"] == 1


# --- TypeSafe backend with a fake typesafe_sdk


class FakeAPIError(Exception):
    def __init__(self, body):
        super().__init__(body)
        self.body = body


class FakeClient:
    answer_model = None
    max_docs = 1000

    def __init__(self, api_key, model, timeout):
        self.model = model

    def system_one(self, state, questions):
        if len(questions) > self.max_docs:
            raise FakeAPIError({"error": "max_tokens_exceeded"})
        body = {
            "model": self.answer_model or self.model,
            "answers": {k: {"type": "noul", "noul": 0.5} for k in questions},
            "usage": {"input_tokens": 100 * len(questions), "output_tokens": len(questions)},
        }
        http = types.SimpleNamespace(json=lambda: body, headers={"x-typesafe-request-id": "req-1"})
        return types.SimpleNamespace(raw_http_response=http)


@pytest.fixture
def jev(monkeypatch):
    sdk = types.ModuleType("typesafe_sdk")
    sdk.TypeSafeClient = FakeClient
    sdk.TypeSafeAPIError = FakeAPIError
    monkeypatch.setitem(sys.modules, "typesafe_sdk", sdk)
    monkeypatch.setenv("TYPESAFE_AI_API_KEY", "test-key")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    FakeClient.answer_model, FakeClient.max_docs = None, 1000
    from cdm.rerankers import load_reranker

    return load_reranker


def test_jev_cost_and_no_warmup(jev):
    r = jev("jev")
    out = r.score("q", ["a", "b"], get_phrasing("s1rank"))
    assert out.probs == [0.5, 0.5] and out.raw[0]["request_id"] == "req-1"
    assert r.run_stats()["api_cost_usd"] == pytest.approx(200 / 1e6 * 0.042)
    assert not r.warmup


def test_jev_context_error_splits(jev):
    FakeClient.max_docs = 1
    r = jev("jev")
    r.score("q", ["a", "b"], get_phrasing("s1rank"))
    assert r.run_stats()["window_splits"] == 1 and r.run_stats()["requests"] == 2


def test_jev_model_drift_is_an_error(jev):
    FakeClient.answer_model = "jev-1.14.0"
    with pytest.raises(RuntimeError, match="jev-1.14.0"):
        jev("jev").score("q", ["a"], get_phrasing("s1rank"))


def test_jev_missing_key(jev, monkeypatch):
    monkeypatch.delenv("TYPESAFE_AI_API_KEY")
    with pytest.raises(RuntimeError, match="API key"):
        jev("jev")

"""JevReranker against a fake typesafe_sdk: state and question format, joint vs isolated, splitting, model pinning."""

import sys
import types

import pytest

from cdm.phrasings import get_phrasing


class FakeAPIError(Exception):
    def __init__(self, body):
        super().__init__(body)
        self.body = body


class FakeHTTPResponse:
    def __init__(self, body):
        self._body = body
        self.headers = {"x-typesafe-request-id": "req-1"}

    def json(self):
        return self._body


class FakeClient:
    """Answers each Noul with 0.01 * (number in its label); too long above max_docs documents."""

    max_docs = 1000
    answer_model = None

    def __init__(self, api_key, model, timeout):
        self.model = model
        self.requests = []

    def system_one(self, state, questions):
        self.requests.append((state, questions))
        if len(state["documents"]) > self.max_docs:
            raise FakeAPIError({"error": "max_tokens_exceeded"})
        answers = {
            k: {"type": "noul", "noul": 0.5 if "tie" in state["documents"][k] else 0.01 * int(k[1:])} for k in questions
        }
        body = {
            "model": self.answer_model or self.model,
            "answers": answers,
            "usage": {"input_tokens": 100 * len(questions), "output_tokens": len(questions)},
        }
        return types.SimpleNamespace(raw_http_response=FakeHTTPResponse(body))


@pytest.fixture
def jev(monkeypatch):
    sdk = types.ModuleType("typesafe_sdk")
    sdk.TypeSafeClient = FakeClient
    sdk.TypeSafeAPIError = FakeAPIError
    monkeypatch.setitem(sys.modules, "typesafe_sdk", sdk)
    monkeypatch.setenv("TYPESAFE_AI_API_KEY", "test-key")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    FakeClient.max_docs, FakeClient.answer_model = 1000, None
    from cdm.rerankers import load_reranker

    return load_reranker


def test_joint_one_request_s1rank_format(jev):
    r = jev("jev")
    out = r.score("what is x", ["a", "b", "c"], get_phrasing("s1rank"))
    ((state, questions),) = r.client.requests
    assert state == {"query": "what is x", "documents": {"D1": "a", "D2": "b", "D3": "c"}}
    assert questions["D2"] == {
        "type": "noul",
        "instructions": "Is document `documents.D2` relevant to the `query`? "
        "Relevant means it answers the query or contains information that directly addresses it.",
    }
    assert out.probs == [0.01, 0.02, 0.03]
    assert [rec["nouls"] for rec in out.raw] == [[0.01, 0.02, 0.03]]
    stats = r.run_stats()
    assert stats["api_requests"] == 1 and stats["input_tokens"] == 300
    assert stats["api_cost_usd"] == pytest.approx(300 / 1e6 * 0.042)
    assert r.joint and r.describe()["window"] == 0


def test_isolated_one_request_per_passage(jev):
    r = jev("jev-isolated")
    out = r.score("q", ["a", "b", "c"], get_phrasing("p1-relevant"))
    assert len(r.client.requests) == 3
    assert all(list(state["documents"]) == ["D1"] for state, _ in r.client.requests)
    assert out.probs == [0.01, 0.01, 0.01]
    assert sorted(rec["offset"] for rec in out.raw) == [0, 1, 2]
    assert not r.joint


def test_ties_broken_by_first_stage_rank(jev):
    r = jev("jev")
    out = r.score("q", ["tie", "tie", "tie"], get_phrasing("s1rank"))
    assert out.probs == [0.5, 0.5, 0.5]
    assert out.scores[0] > out.scores[1] > out.scores[2]


def test_too_long_splits_in_half(jev):
    FakeClient.max_docs = 2
    r = jev("jev")
    out = r.score("q", ["a", "b", "c", "d", "e"], get_phrasing("s1rank"))
    # 5 -> 2 + 3 -> 2 + (1 + 2); each half is labelled from D1 again
    assert out.probs == [0.01, 0.02, 0.01, 0.01, 0.02]
    assert r.run_stats()["window_splits"] == 2
    assert [(rec["offset"], rec["n"]) for rec in out.raw] == [(0, 2), (2, 1), (3, 2)]


def test_model_drift_is_an_error(jev):
    FakeClient.answer_model = "jev-1.14.0"
    r = jev("jev")
    with pytest.raises(RuntimeError, match="jev-1.14.0"):
        r.score("q", ["a"], get_phrasing("s1rank"))


def test_missing_key(jev, monkeypatch):
    monkeypatch.delenv("TYPESAFE_AI_API_KEY")
    with pytest.raises(RuntimeError, match="API key"):
        jev("jev")

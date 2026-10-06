"""
System One decision models: Laya (open weights, the testbed) and Jev (closed API, the reference point).

Laya is still a STUB: the research plan's WP0 has to confirm its API, checkpoints, licence and fine-tuning notebook
before any code depends on them (the owner is getting Laya API access, docs/intent/models.md). Jev is implemented
against the TypeSafe API (typesafe-sdk). What the implementations must provide, so that RQ2-RQ4 can be answered from
stored outputs:

  - score() returns the model's own relevance probability per passage in Scores.probs (prob_kind = "model"),
    and Scores.scores as the ranking score (normally the same probability).
  - The primitive (Noul, Score or Choice) and the phrasing are parameters, recorded by describe().
  - joint=True variants score the whole candidate set in one state; Laya's 512/1024-token context limits the set
    size, so they take a set_size and report truncation via count_truncated().
"""

import os
import time
from typing import Any, Dict, List

from cdm.phrasings import Phrasing
from cdm.rerankers.base import Reranker, Scores


class LayaReranker(Reranker):
    prob_kind = "model"
    uses_phrasing = True

    def __init__(self, name: str, model_id: str, primitive: str = "noul", joint: bool = False, set_size: int = 0,
                 **_: Any):
        self.name = name
        self.model_id = model_id
        self.primitive = primitive
        self.joint = joint
        self.set_size = set_size
        raise NotImplementedError(
            "Laya is not wired up yet: confirm its API, checkpoints and licence first (research plan WP0), then "
            "implement LayaReranker in src/cdm/rerankers/system_one.py."
        )

    def score(self, query: str, passages: List[str], phrasing: Phrasing) -> Scores:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(primitive=self.primitive, set_size=self.set_size)
        return d




# Jev list price for jev-1.13.0, $ per million input tokens; output tokens are free (docs.typesafe.ai/models,
# read 2026-10-06). Recorded in describe() so a later price change can't silently rewrite old cost numbers.
JEV_PRICE_PER_M_INPUT = 0.042
# The SDK reads TYPESAFE_API_KEY; this project's .env uses TYPESAFE_AI_API_KEY. Either works.
JEV_API_KEY_VARS = ("TYPESAFE_API_KEY", "TYPESAFE_AI_API_KEY")


class JevContextTooLong(RuntimeError):
    """Jev rejected a request as longer than its context (64k tokens per request, 32k for state + longest question)."""


class JevReranker(Reranker):
    """
    Jev (TypeSafe's closed System One model) through its API, one Noul question per candidate.

    The state is {"query": ..., "documents": {"D1": ..., "Dn": ...}} with the documents in first-stage order, and the
    question for document Di is the phrasing with {doc} replaced by `documents.Di` -- the S1Rank setup
    (github.com/zaesho/S1Rank, s1rank/methods.py), so its numbers can be replicated. `window` sets how many
    candidates share one request: 0 = all of them (joint, one request per query), 1 = one per request (isolated).
    Both use the same state format, so joint vs isolated differs only in what else is in the state (RQ4).

    Jev returns nouls rounded to two decimals, so ties are common. Scores.probs holds the raw nouls; Scores.scores
    subtracts 1e-6 per first-stage position, which breaks ties by first-stage rank (as S1Rank does) without changing
    the order of distinct nouls. Every raw API response goes into Scores.raw for outputs/.../responses.jsonl.gz.
    """

    prob_kind = "model"
    uses_phrasing = True
    warmup = False  # nothing to warm up client-side; a warm-up request would only cost money

    def __init__(self, name: str, model_id: str, window: int = 0, primitive: str = "noul", max_workers: int = 16,
                 timeout: float = 120.0, **_: Any):
        if primitive != "noul":
            raise ValueError(f"JevReranker supports primitive 'noul' only so far, not '{primitive}' (WP3 adds more)")
        from typesafe_sdk import TypeSafeClient

        api_key = next((os.environ[v] for v in JEV_API_KEY_VARS if os.environ.get(v)), None)
        if not api_key:
            raise RuntimeError(f"No Jev API key: set one of {', '.join(JEV_API_KEY_VARS)} in .env")
        self.name = name
        self.model_id = model_id
        self.window = window
        self.joint = window != 1
        self.primitive = primitive
        self.max_workers = max_workers
        self.sdk_version = _version("typesafe-sdk")
        # Pinned model id, never an alias: an alias moves to a new model without notice (docs.typesafe.ai/models).
        self.client = TypeSafeClient(api_key=api_key, model=model_id, timeout=timeout)
        self.reset_stats()

    def reset_stats(self) -> None:
        self._stats = {"api_requests": 0, "input_tokens": 0, "output_tokens": 0, "window_splits": 0}

    def run_stats(self) -> Dict[str, float]:
        stats: Dict[str, float] = dict(self._stats)
        stats["api_cost_usd"] = self._stats["input_tokens"] / 1e6 * JEV_PRICE_PER_M_INPUT
        return stats

    def _ask(self, query: str, passages: List[str], offset: int, phrasing: Phrasing) -> Dict[str, Any]:
        import typesafe_sdk

        labels = [f"D{i + 1}" for i in range(len(passages))]
        state = {"query": query, "documents": dict(zip(labels, passages))}
        questions = {label: {"type": "noul", "instructions": phrasing.question.format(doc=f"documents.{label}")}
                     for label in labels}
        start = time.perf_counter()
        try:
            response = self.client.system_one(state=state, questions=questions)
        except typesafe_sdk.TypeSafeAPIError as e:
            if "max_tokens_exceeded" in str(e.body):
                raise JevContextTooLong(str(e.body)) from e
            raise
        latency = time.perf_counter() - start
        body = response.raw_http_response.json()
        if body.get("model") != self.model_id:
            raise RuntimeError(f"Jev answered with model '{body.get('model')}', expected '{self.model_id}'")
        usage = body.get("usage") or {}
        return {"offset": offset, "n": len(passages), "model": body["model"],
                "nouls": [body["answers"][label]["noul"] for label in labels],
                "usage": usage, "latency_s": round(latency, 4),
                "request_id": response.raw_http_response.headers.get("x-typesafe-request-id")}

    def _ask_split(self, query: str, passages: List[str], offset: int, phrasing: Phrasing) -> List[Dict[str, Any]]:
        """One request, or -- if Jev says it is too long -- two halves, recursively (recorded as window_splits)."""
        try:
            return [self._ask(query, passages, offset, phrasing)]
        except JevContextTooLong:
            if len(passages) == 1:
                raise
            self._stats["window_splits"] += 1
            half = len(passages) // 2
            return (self._ask_split(query, passages[:half], offset, phrasing)
                    + self._ask_split(query, passages[half:], offset + half, phrasing))

    def score(self, query: str, passages: List[str], phrasing: Phrasing) -> Scores:
        from concurrent.futures import ThreadPoolExecutor

        size = self.window or len(passages)
        chunks = [(i, passages[i:i + size]) for i in range(0, len(passages), size)]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(chunks))) as pool:
            records = [r for part in pool.map(lambda c: self._ask_split(query, c[1], c[0], phrasing), chunks)
                       for r in part]
        probs: List[float] = [float("nan")] * len(passages)
        for r in records:
            probs[r["offset"]:r["offset"] + r["n"]] = r["nouls"]
            self._stats["api_requests"] += 1
            self._stats["input_tokens"] += r["usage"].get("input_tokens") or 0
            self._stats["output_tokens"] += r["usage"].get("output_tokens") or 0
        scores = [p - 1e-6 * i for i, p in enumerate(probs)]
        return Scores(scores=scores, probs=probs, raw=records)

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(window=self.window, primitive=self.primitive, max_workers=self.max_workers,
                 sdk_version=self.sdk_version, price_per_m_input_usd=JEV_PRICE_PER_M_INPUT,
                 state_format='{"query": q, "documents": {"D1": d1, ...}}')
        return d


def _version(package: str) -> str:
    from importlib import metadata

    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"

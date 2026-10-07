"""
System One decision models as rerankers: Jev (TypeSafe's closed API), Laya (open weights, convaiinnovations/laya) and
Clef (open weights, Cloudflare/clef and Cloudflare/clef-flash).

All three take the same request -- a state and a map of typed questions -- so one reranker builds the request and a
backend only runs it. Every model therefore sees byte-identical states and questions, and the comparison between
models is not confounded by prompt formatting:

  state     {"query": q, "documents": {"D1": d1, ..., "Dn": dn}}   documents in first-stage order
  questions {"D1": <noul>, ..., "Dn": <noul>}                      instructions = phrasing with {doc} = documents.Di

This is S1Rank's format (github.com/zaesho/S1Rank, s1rank/methods.py), so its Jev numbers can be replicated.
`window` sets how many candidates share one request: 0 = all of them (joint), 1 = one per request (isolated), k = chunks
of k. Joint and isolated differ only in what else is in the state (RQ4).

A request that is too long for the model is split in half, recursively (counted as window_splits). A single passage
that is still too long is truncated by the model and counted (count_truncated). Laya builds one sequence per question
and cuts the state to its budget (about 320 tokens on the English checkpoint), so it runs isolated.

Scores.probs are the model's own probabilities (the noul, i.e. P(yes)) as returned -- Laya's include the temperatures
its checkpoint ships with. Scores.scores subtracts 1e-6 per first-stage position, which breaks ties (Jev rounds to two
decimals) by first-stage rank without changing the order of distinct probabilities. Every raw response goes into
Scores.raw for outputs/.../responses.jsonl.gz.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from cdm.phrasings import Phrasing
from cdm.rerankers.base import Reranker, Scores

# One request: (state, questions). A backend answers a list of them with response bodies in the Jev format
# ({"model", "answers", "usage"}), or TOO_LONG for a request that doesn't fit the model's context and may not be cut
# (truncate[i] False). A cut request says so in usage["truncated"].
Request = Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]
TOO_LONG = "too_long"

# Jev list price for jev-1.13.0, $ per million input tokens; output tokens are free (docs.typesafe.ai/models,
# read 2026-10-06). Recorded in describe() so a later price change can't silently rewrite old cost numbers.
JEV_PRICE_PER_M_INPUT = 0.042
# The SDK reads TYPESAFE_API_KEY; this project's .env uses TYPESAFE_AI_API_KEY. Either works.
JEV_API_KEY_VARS = ("TYPESAFE_API_KEY", "TYPESAFE_AI_API_KEY")


def _version(package: str) -> str:
    from importlib import metadata

    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"


class TypeSafeBackend:
    """Jev through the TypeSafe API (typesafe-sdk), requests in parallel threads."""

    price_per_m_input = JEV_PRICE_PER_M_INPUT

    def __init__(self, model_id: str, max_workers: int = 16, timeout: float = 120.0, **_: Any):
        from typesafe_sdk import TypeSafeClient

        api_key = next((os.environ[v] for v in JEV_API_KEY_VARS if os.environ.get(v)), None)
        if not api_key:
            raise RuntimeError(f"No Jev API key: set one of {', '.join(JEV_API_KEY_VARS)} in .env")
        self.model_id = model_id
        self.max_workers = max_workers
        # Pinned model id, never an alias: an alias moves to a new model without notice (docs.typesafe.ai/models).
        self.client = TypeSafeClient(api_key=api_key, model=model_id, timeout=timeout)

    def _one(self, request: Request) -> Any:
        import typesafe_sdk

        state, questions = request
        start = time.perf_counter()
        try:
            response = self.client.system_one(state=state, questions=questions)
        except typesafe_sdk.TypeSafeAPIError as e:
            if "max_tokens_exceeded" in str(e.body):
                return TOO_LONG
            raise
        body = response.raw_http_response.json()
        if body.get("model") != self.model_id:
            raise RuntimeError(f"Jev answered with model '{body.get('model')}', expected '{self.model_id}'")
        body["latency_s"] = round(time.perf_counter() - start, 4)
        body["request_id"] = response.raw_http_response.headers.get("x-typesafe-request-id")
        return body

    def answer(self, requests: List[Request], truncate: List[bool]) -> List[Any]:
        """Jev rejects a request that is too long instead of cutting it, so `truncate` changes nothing here."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(requests))) as pool:
            return list(pool.map(self._one, requests))

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": "typesafe-api",
            "sdk_version": _version("typesafe-sdk"),
            "max_workers": self.max_workers,
            "price_per_m_input_usd": self.price_per_m_input,
        }


class LayaBackend:
    """
    Laya on a local GPU (pip package `laya`). Requests that share their questions -- every isolated request does --
    go through Agent.predict_batch together. Laya never rejects a long state, it cuts it; with truncate=False a cut
    request is reported as TOO_LONG so it gets split.
    """

    price_per_m_input = 0.0

    def __init__(
        self,
        model_id: str,
        subfolder: Optional[str] = None,
        revision: Optional[str] = None,
        batch_size: int = 64,
        device: Optional[str] = None,
        **_: Any,
    ):
        import laya

        self.model_id = model_id
        self.subfolder = subfolder
        self.batch_size = batch_size
        self.agent = laya.load(model_id, subfolder=subfolder, revision=revision, device=device)
        self.revision = revision

    def answer(self, requests: List[Request], truncate: List[bool]) -> List[Any]:
        groups: Dict[str, List[int]] = {}
        for i, (_, questions) in enumerate(requests):
            groups.setdefault(json.dumps(questions, sort_keys=True), []).append(i)
        out: List[Any] = [None] * len(requests)
        for indices in groups.values():
            questions = requests[indices[0]][1]
            start = time.perf_counter()
            results = self.agent.predict_batch([requests[i][0] for i in indices], questions, batch_size=self.batch_size)
            latency = (time.perf_counter() - start) / len(indices)
            for i, result in zip(indices, results):
                if result["usage"].get("truncated") and not truncate[i]:
                    out[i] = TOO_LONG
                else:
                    out[i] = {
                        **result,
                        "model": self.model_id + (f"/{self.subfolder}" if self.subfolder else ""),
                        "latency_s": round(latency, 4),
                    }
        return out

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": "laya",
            "laya_version": _version("laya"),
            "subfolder": self.subfolder,
            "revision": self.revision,
            "batch_size": self.batch_size,
            "max_len": self.agent.cfg.get("max_len"),
            "head_max_len": self.agent.cfg.get("head_max_len"),
            "temperature": self.agent.cfg.get("temperature"),
            "device": str(getattr(self.agent, "device", "")),
        }


class ClefBackend:
    """
    Clef on a local GPU, through the joint_schema_model.py shipped in the model repo (read 2026-10-06: plain PyTorch,
    no network beyond the Hub download). Its encode_record silently cuts the state to max_length; this backend checks
    the full length first and reports TOO_LONG instead, unless truncate=True. Loaded with the tokenizer only (no
    image/video processor), since all inputs are text.
    """

    price_per_m_input = 0.0

    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        max_length: int = 16384,
        batch_size: int = 8,
        device: str = "cuda",
        **_: Any,
    ):
        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.path = snapshot_download(model_id, revision=revision)
        self.revision = os.path.basename(self.path)  # the snapshot directory is named by its commit hash
        sys.path.insert(0, self.path)
        import joint_schema_model as jsm

        self.jsm = jsm
        backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.path, dtype=torch.bfloat16, device_map={"": device}
        )
        backbone.config.use_cache = False
        with open(os.path.join(self.path, "joint_head_config.json")) as f:
            head = jsm.JointSchemaHead(**json.load(f))
        head.load_state_dict(load_file(os.path.join(self.path, "joint_head.safetensors")), strict=True)
        head = head.to(device=device, dtype=torch.bfloat16)
        self.model = jsm.ClefModel(backbone, head).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.device = torch.device(device)

    def answer(self, requests: List[Request], truncate: List[bool]) -> List[Any]:
        import torch

        out: List[Any] = [None] * len(requests)
        encoded = []
        for i, (state, questions) in enumerate(requests):
            record = {"state": state, "questions": questions}
            full = self.jsm.encode_record(self.tokenizer, record, max_length=10**9)
            if len(full.input_ids) > self.max_length and not truncate[i]:
                out[i] = TOO_LONG
                continue
            enc = (
                full
                if len(full.input_ids) <= self.max_length
                else self.jsm.encode_record(self.tokenizer, record, max_length=self.max_length)
            )
            encoded.append((i, enc, len(full.input_ids) - len(enc.input_ids)))
        # Longest first, so a batch holds sequences of similar length (less padding).
        encoded.sort(key=lambda x: -len(x[1].input_ids))
        for b in range(0, len(encoded), self.batch_size):
            batch = encoded[b : b + self.batch_size]
            start = time.perf_counter()
            with torch.inference_mode():
                logits = self.model(
                    self.jsm.collate_records([e for _, e, _ in batch], self.tokenizer.pad_token_id, self.device)
                )
            latency = (time.perf_counter() - start) / len(batch)
            for (i, enc, dropped), record_logits in zip(batch, logits):
                questions = requests[i][1]
                answers = {
                    q.question_id: self.jsm.systemone_answer(
                        questions[q.question_id], dict(zip(q.option_ids, ql.float().softmax(-1).tolist()))
                    )
                    for q, ql in zip(enc.questions, record_logits)
                }
                out[i] = {
                    "model": self.model_id,
                    "answers": answers,
                    "latency_s": round(latency, 4),
                    "usage": {
                        "input_tokens": len(enc.input_ids),
                        "output_tokens": 0,
                        "state_tokens_dropped": dropped,
                        "truncated": dropped > 0,
                    },
                }
        return out

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": "clef",
            "revision": self.revision,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "dtype": "bfloat16",
            "device": str(self.device),
        }


BACKENDS = {"typesafe": TypeSafeBackend, "laya": LayaBackend, "clef": ClefBackend}


class SystemOneReranker(Reranker):
    """One Noul question per candidate, asked of any System One model through its backend (see the module doc)."""

    prob_kind = "model"
    uses_phrasing = True

    def __init__(
        self,
        name: str,
        model_id: str,
        backend: str,
        window: int = 0,
        primitive: str = "noul",
        batch_size: Optional[int] = None,
        **backend_params: Any,
    ):
        if primitive != "noul":
            raise ValueError(f"SystemOneReranker supports primitive 'noul' only so far, not '{primitive}' (WP3)")
        self.name = name
        self.model_id = model_id
        self.window = window
        self.joint = window != 1
        self.primitive = primitive
        if batch_size is not None:
            backend_params["batch_size"] = batch_size
        self.backend = BACKENDS[backend](model_id, **backend_params)
        # API calls have nothing to warm up client-side; a warm-up request would only cost money.
        self.warmup = backend != "typesafe"
        self._last_truncated = 0
        self.reset_stats()

    def reset_stats(self) -> None:
        self._stats = {"requests": 0, "input_tokens": 0, "window_splits": 0, "truncated_passages": 0}

    def run_stats(self) -> Dict[str, float]:
        stats: Dict[str, float] = dict(self._stats)
        if self.backend.price_per_m_input:
            stats["api_cost_usd"] = self._stats["input_tokens"] / 1e6 * self.backend.price_per_m_input
        return stats

    @staticmethod
    def _request(query: str, passages: List[str], phrasing: Phrasing) -> Request:
        labels = [f"D{i + 1}" for i in range(len(passages))]
        state = {"query": query, "documents": dict(zip(labels, passages))}
        questions = {
            label: {"type": "noul", "instructions": phrasing.question.format(doc=f"documents.{label}")}
            for label in labels
        }
        return state, questions

    def score(self, query: str, passages: List[str], phrasing: Phrasing) -> Scores:
        size = self.window or len(passages)
        pending = [(i, passages[i : i + size]) for i in range(0, len(passages), size)]
        records: List[Dict[str, Any]] = []
        truncated = 0
        while pending:
            # Only a single passage may be cut; a longer request that doesn't fit is split in half instead.
            bodies = self.backend.answer(
                [self._request(query, chunk, phrasing) for _, chunk in pending],
                truncate=[len(chunk) == 1 for _, chunk in pending],
            )
            split = []
            for (offset, chunk), body in zip(pending, bodies):
                if body == TOO_LONG:
                    if len(chunk) == 1:
                        raise RuntimeError(f"{self.name}: a single passage is too long for the model")
                    self._stats["window_splits"] += 1
                    half = len(chunk) // 2
                    split += [(offset, chunk[:half]), (offset + half, chunk[half:])]
                else:
                    records.append(self._record(offset, chunk, body))
                    truncated += bool(body.get("usage", {}).get("truncated"))
            pending = split
        probs: List[float] = [float("nan")] * len(passages)
        for r in records:
            probs[r["offset"] : r["offset"] + r["n"]] = r["nouls"]
            self._stats["requests"] += 1
            self._stats["input_tokens"] += r["usage"].get("input_tokens") or 0
        self._stats["truncated_passages"] += truncated
        self._last_truncated = truncated
        records.sort(key=lambda r: r["offset"])
        scores = [p - 1e-6 * i for i, p in enumerate(probs)]
        return Scores(scores=scores, probs=probs, raw=records)

    @staticmethod
    def _record(offset: int, chunk: List[str], body: Dict[str, Any]) -> Dict[str, Any]:
        labels = [f"D{i + 1}" for i in range(len(chunk))]
        return {
            "offset": offset,
            "n": len(chunk),
            "model": body.get("model"),
            "nouls": [body["answers"][label]["noul"] for label in labels],
            "usage": body.get("usage") or {},
            "latency_s": body.get("latency_s"),
            "request_id": body.get("request_id"),
        }

    def count_truncated(self, query: str, passages: List[str]) -> int:
        """Passages the model had to cut in the last score() call (the rerank CLI calls this right after it)."""
        return self._last_truncated

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(
            window=self.window,
            primitive=self.primitive,
            **self.backend.describe(),
            state_format='{"query": q, "documents": {"D1": d1, ...}}',
        )
        return d

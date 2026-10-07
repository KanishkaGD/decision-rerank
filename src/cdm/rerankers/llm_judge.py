"""
A decoder-only LLM as a zero-shot pointwise relevance judge, through OpenRouter chat completions: Kanishka's
"decoder LLM at Jev's price" reference (master: src/decisionrerank/llm_judge.py), whose prompt and readout this follows.

One request per (query, passage), reasoning off, one generated token, temperature 0. P(yes) = mass(yes) / (mass(yes) +
mass(no)) over the first token's top-5 log-probabilities, summing case and leading-space variants. The prompt is fixed
(Kanishka's, chosen before any run), not one of src/cdm/phrasings.py: those address fields of a System One state.

Needs OPENROUTER_API_KEY in .env. Runs on the CPU partition, like Jev.
"""

import json
import math
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from cdm.phrasings import Phrasing
from cdm.rerankers.base import Reranker, Scores

URL = "https://openrouter.ai/api/v1/chat/completions"
NOUL_PROMPT = 'Query: {q}\nPassage: {p}\n\nDoes the passage answer the query? Answer with only "yes" or "no".'
RETRY_STATUS = {429, 500, 502, 503, 504}


def yes_probability(top: List[List[Any]]) -> Optional[float]:
    """P(yes) renormalised over yes/no from (token, logprob) pairs; None if neither is among them."""
    mass = {"yes": 0.0, "no": 0.0}
    for token, logprob in top:
        t = token.strip().lower()
        if t in mass:
            mass[t] += math.exp(logprob)
    total = mass["yes"] + mass["no"]
    return mass["yes"] / total if total > 0 else None


class OpenRouterJudge(Reranker):
    prob_kind = "llm-logprobs"
    warmup = False  # an API call has nothing to warm up; it would only cost money

    def __init__(
        self,
        name: str,
        model_id: str,
        max_workers: int = 24,
        timeout: float = 60.0,
        tries: int = 10,
        top_logprobs: int = 5,
        **_: Any,
    ):
        self.name = name
        self.model_id = model_id
        self.max_workers = max_workers
        self.timeout = timeout
        self.tries = tries
        self.top_logprobs = top_logprobs
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("No OpenRouter API key: set OPENROUTER_API_KEY in .env")
        self.reset_stats()

    def reset_stats(self) -> None:
        self._stats = {"requests": 0, "input_tokens": 0, "api_cost_usd": 0.0, "no_yes_no_answer": 0}

    def run_stats(self) -> Dict[str, float]:
        return dict(self._stats)

    def _call(self, prompt: str) -> Dict[str, Any]:
        body = json.dumps(
            {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": True,
                "top_logprobs": self.top_logprobs,
                "reasoning": {"enabled": False},
                "provider": {"require_parameters": True},
                "usage": {"include": True},
            }
        ).encode()
        last_error = ""
        for attempt in range(self.tries):
            request = urllib.request.Request(
                URL,
                data=body,
                method="POST",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            )
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as r:
                    d = json.load(r)
                content = ((d["choices"][0].get("logprobs") or {}).get("content")) or []
                if not content:  # e.g. a reasoning token came first: retry
                    raise ValueError("no logprobs")
                usage = d.get("usage") or {}
                return {
                    "model": d.get("model"),
                    "provider": d.get("provider"),
                    "token": content[0]["token"],
                    "top": [[x["token"], x["logprob"]] for x in content[0]["top_logprobs"]],
                    "cost": float(usage.get("cost") or 0.0),
                    "input_tokens": usage.get("prompt_tokens") or 0,
                    "latency_s": round(time.perf_counter() - start, 4),
                    "id": d.get("id"),
                }
            except urllib.error.HTTPError as e:
                last_error = f"HTTP {e.code}: {e.read().decode()[:200]}"
                if e.code not in RETRY_STATUS:
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError, KeyError) as e:
                last_error = repr(e)
            time.sleep(min(30.0, 1.5**attempt))
        raise RuntimeError(f"{self.name}: OpenRouter request failed ({last_error})")

    def score(self, query: str, passages: List[str], phrasing: Phrasing) -> Scores:
        from concurrent.futures import ThreadPoolExecutor

        prompts = [NOUL_PROMPT.format(q=query, p=p) for p in passages]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(prompts))) as pool:
            records = list(pool.map(self._call, prompts))
        probs = []
        for i, r in enumerate(records):
            p = yes_probability(r["top"])
            if p is None:  # neither answer among the top tokens: counted, and scored as undecided
                self._stats["no_yes_no_answer"] += 1
                p = 0.5
            probs.append(p)
            r.update(offset=i, noul=p)
            self._stats["requests"] += 1
            self._stats["input_tokens"] += r["input_tokens"]
            self._stats["api_cost_usd"] += r["cost"]
        # The provider rounds log-probs, so ties occur; break them by first-stage rank, as SystemOneReranker does.
        return Scores(scores=[p - 1e-6 * i for i, p in enumerate(probs)], probs=probs, raw=records)

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(
            backend="openrouter",
            prompt=NOUL_PROMPT,
            max_tokens=1,
            temperature=0,
            reasoning=False,
            top_logprobs=self.top_logprobs,
            max_workers=self.max_workers,
        )
        return d

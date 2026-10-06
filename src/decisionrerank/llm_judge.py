"""A decoder-only LLM as a zero-shot relevance judge, through OpenRouter chat completions.

The same two questions as Laya and Jev, as plain prompts (fixed before any run). Reasoning is switched off and
one token is generated; the answer's distribution comes from the first token's top log-probabilities:
- noul:  P(yes) = mass(yes) / (mass(yes) + mass(no)), summing case and leading-space variants;
- score: P(y) for y = 0..3 from the digit tokens, renormalised (digits missing from the top 5 get 0).
Every response is appended to runs/<slug>_responses.jsonl first, so a rerun never pays twice.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from . import data
from .questions import UMBRELA

URL = "https://openrouter.ai/api/v1/chat/completions"
NOUL = "Query: {q}\nPassage: {p}\n\nDoes the passage answer the query? Answer with only \"yes\" or \"no\"."
SCORE = ("Query: {q}\nPassage: {p}\n\nHow well does the passage answer the query? Choose one level:\n"
         + "\n".join(f"{i} = {t}" for i, t in enumerate(UMBRELA)) + "\nAnswer with only the number of the level.")
RETRY = {429, 500, 502, 503, 504}


def slug(model: str) -> str:
    return model.split("/")[-1].replace(".", "")


def cache_path(model: str):
    return data.RUNS / f"{slug(model)}_responses.jsonl"


def call(key: str, model: str, prompt: str, top: int = 5, timeout: float = 60.0, tries: int = 10) -> dict:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1,
                       "temperature": 0, "logprobs": True, "top_logprobs": top, "reasoning": {"enabled": False},
                       "provider": {"require_parameters": True}}).encode()
    for attempt in range(tries):
        req = urllib.request.Request(URL, data=body, method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            ch = d["choices"][0]
            lp = (ch.get("logprobs") or {}).get("content") or []
            if not lp:  # no log-probs (e.g. a reasoning token came first): retry
                raise ValueError("no logprobs")
            return {"token": lp[0]["token"], "top": [(x["token"], x["logprob"]) for x in lp[0]["top_logprobs"]],
                    "cost": float((d.get("usage") or {}).get("cost") or 0.0), "model": d.get("model")}
        except urllib.error.HTTPError as e:
            if e.code not in RETRY:
                return {"error": e.code, "body": e.read().decode()[:200]}
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError, KeyError):
            pass
        time.sleep(min(30, 1.5 ** attempt))
    return {"error": "retries exhausted"}


def noul_p(top) -> float:
    mass = {"yes": 0.0, "no": 0.0}
    for tok, lp in top:
        t = tok.strip().lower()
        if t in mass:
            mass[t] += math.exp(lp)
    s = mass["yes"] + mass["no"]
    return mass["yes"] / s if s > 0 else float("nan")


def score_p(top) -> list[float]:
    p = [0.0] * 4
    for tok, lp in top:
        t = tok.strip()
        if t in ("0", "1", "2", "3"):
            p[int(t)] += math.exp(lp)
    s = sum(p)
    return [x / s for x in p] if s > 0 else [float("nan")] * 4


def cached(model: str) -> dict:
    out, path = {}, cache_path(model)
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if "noul" in r and "score" in r and "error" not in r["noul"] and "error" not in r["score"]:
                out[(r["qset"], r["query_id"], r["doc_id"])] = r
    return out


def score(model: str, pairs: pd.DataFrame, workers: int = 24, budget_usd: float = 3.0, log_every: int = 1000) -> None:
    """Both questions for every pair not yet cached, in the given row order."""
    key = os.environ["OPENROUTER_API_KEY"]
    have = cached(model)
    todo = [r for r in pairs[data.KEY + ["query", "passage"]].itertuples(index=False)
            if (r.qset, r.query_id, r.doc_id) not in have]
    print(f"{model}: {len(pairs)} pairs, {len(pairs) - len(todo)} cached, {len(todo)} to request", flush=True)
    lock, spent, done, errors, t0 = threading.Lock(), [0.0], [0], [0], time.time()
    data.RUNS.mkdir(exist_ok=True)
    with open(cache_path(model), "a") as fh, ThreadPoolExecutor(workers) as pool:
        def one(r):
            if spent[0] >= budget_usd:
                return
            res = {"qset": r.qset, "query_id": r.query_id, "doc_id": r.doc_id,
                   "noul": call(key, model, NOUL.format(q=r.query, p=r.passage)),
                   "score": call(key, model, SCORE.format(q=r.query, p=r.passage))}
            with lock:
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                spent[0] += res["noul"].get("cost", 0.0) + res["score"].get("cost", 0.0)
                done[0] += 1
                errors[0] += ("error" in res["noul"]) + ("error" in res["score"])
                if done[0] % log_every == 0 or done[0] == len(todo):
                    el = time.time() - t0
                    print(f"  {done[0]}/{len(todo)} pairs  {done[0] / el:.1f} pairs/s  ETA {(len(todo) - done[0]) / (done[0] / el) / 60:.1f} min"
                          f"  spent ${spent[0]:.3f}  errors {errors[0]}", flush=True)
        for f in as_completed([pool.submit(one, r) for r in todo]):
            f.result()
    print(f"{model}: done, ${spent[0]:.3f} this run, {errors[0]} errors", flush=True)


def to_frame(model: str, prefix: str) -> pd.DataFrame:
    """KEY, {prefix}_noul_p, {prefix}_score_p0..3, {prefix}_cost, {prefix}_model."""
    rows = []
    for (qs, q, d), r in cached(model).items():
        sp = score_p(r["score"]["top"])
        rows.append({"qset": qs, "query_id": q, "doc_id": d, f"{prefix}_noul_p": noul_p(r["noul"]["top"]),
                     **{f"{prefix}_score_p{y}": sp[y] for y in range(4)},
                     f"{prefix}_cost": r["noul"]["cost"] + r["score"]["cost"], f"{prefix}_model": r["noul"].get("model")})
    return pd.DataFrame(rows)

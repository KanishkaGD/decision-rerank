"""TypeSafe Jev (closed System One model) through OpenRouter's decisions endpoint.

One request per (query, passage) pair, asking the same two questions as Laya (``questions.py``)
about the same state. Jev returns probabilities rounded to two decimals, so its rankings tie often;
``probs.systems`` also offers P(yes) with ties broken by Jev's own expected grade.

Every response is appended to runs/jev_responses.jsonl before anything else happens, so an
interrupted or repeated run never pays for a pair twice. Needs OPENROUTER_API_KEY (see .env).
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from . import data
from .questions import QUESTIONS, state

URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"  # the only Jev on OpenRouter; resolves to jev-1.13-20260917
CACHE = data.RUNS / "jev_responses.jsonl"
RETRY = {429, 500, 502, 503, 504}


def ask(key: str, st: dict, timeout: float = 60.0, tries: int = 8) -> dict:
    body = json.dumps({"model": MODEL, "state": st, "questions": QUESTIONS}).encode()
    for attempt in range(tries):
        req = urllib.request.Request(URL, data=body, method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.load(r)
            out["latency_s"] = time.time() - t
            return out
        except urllib.error.HTTPError as e:
            if e.code not in RETRY:
                return {"error": e.code, "body": e.read().decode()[:300]}
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(min(60, 2 ** attempt))
    return {"error": "retries exhausted"}


def cached() -> dict:
    out = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            r = json.loads(line)
            if "answers" in r:
                out[(r["qset"], r["query_id"], r["doc_id"])] = r
    return out


def score(pairs: pd.DataFrame, workers: int = 32, budget_usd: float = 5.0, log_every: int = 2000) -> None:
    """Fetch Jev answers for every pair not yet cached; stops submitting once `budget_usd` is spent."""
    key = os.environ["OPENROUTER_API_KEY"]
    have = cached()
    todo = [r for r in pairs[data.KEY + ["query", "passage"]].itertuples(index=False)
            if (r.qset, r.query_id, r.doc_id) not in have]
    print(f"jev: {len(pairs)} pairs, {len(pairs) - len(todo)} cached, {len(todo)} to request", flush=True)
    lock, spent, done, errors, t0 = threading.Lock(), [0.0], [0], [0], time.time()
    data.RUNS.mkdir(exist_ok=True)
    with open(CACHE, "a") as fh, ThreadPoolExecutor(workers) as pool:
        def one(r):
            if spent[0] >= budget_usd:
                return
            res = ask(key, state(r.query, r.passage))
            res.update(qset=r.qset, query_id=r.query_id, doc_id=r.doc_id)
            with lock:
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                spent[0] += float((res.get("usage") or {}).get("cost") or 0.0)
                done[0] += 1
                errors[0] += "error" in res
                if done[0] % log_every == 0 or done[0] == len(todo):
                    el = time.time() - t0
                    print(f"  {done[0]}/{len(todo)}  {done[0] / el:.1f} req/s  ETA {(len(todo) - done[0]) / (done[0] / el) / 60:.1f} min"
                          f"  spent ${spent[0]:.3f}  errors {errors[0]}", flush=True)
        for f in as_completed([pool.submit(one, r) for r in todo]):
            f.result()
    print(f"jev: done, ${spent[0]:.3f} spent this run, {errors[0]} errors", flush=True)


def to_frame() -> pd.DataFrame:
    """runs/jev.parquet rows: KEY, jev_noul_p, jev_score_p0..3, jev_model, jev_cost."""
    rows = []
    for (qs, q, d), r in cached().items():
        a = r["answers"]
        p = a["score"]["probabilities"]
        rows.append({"qset": qs, "query_id": q, "doc_id": d, "jev_noul_p": float(a["noul"]["noul"]),
                     **{f"jev_score_p{y}": float(p[str(y)]) for y in range(4)},
                     "jev_model": r.get("model"), "jev_cost": float((r.get("usage") or {}).get("cost") or 0.0)})
    out = pd.DataFrame(rows)
    P = out[[f"jev_score_p{y}" for y in range(4)]].to_numpy()
    out[[f"jev_score_p{y}" for y in range(4)]] = P / P.sum(axis=1, keepdims=True)  # rounded probs may not sum to 1
    return out

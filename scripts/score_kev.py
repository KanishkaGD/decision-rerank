"""Score every pair with a Kev decision model in-process; write runs/kev-0.8b.parquet (or --limit smoke file).

Kev (github.com/jaredpalmer/kev) needs transformers>=5.17, so this runs in Kev's own environment:
    PYTHONPATH=src /root/kev/.venv/bin/python scripts/score_kev.py      (flash-linear-attention 0.5.2 installed)
Same two questions and state as Laya and Jev. Stored: raw probabilities (T = 1) at full precision, so the
checkpoint's shipped temperature (recorded in the .json) is applied offline, as for Laya. Runs in bf16, Kev's serving
default: on 1,028 pool pairs of two queries it gave no within-query ties and a per-query Spearman of 0.998-0.9997
with the fp32 path (max |P(yes) diff| 0.056), at about 5x the speed.
"""

import argparse
import json
import os
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from kev.api import SystemOneRequest, to_record
from kev.checkpoint import Checkpoint, LoadOptions, fused_available
from kev.model import admit

from decisionrerank import data
from decisionrerank.questions import QUESTIONS, state

CHUNK = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jaredpalmer/kev-0.8b")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    a = ap.parse_args()
    name = a.model.split("/")[-1]
    stem = name if not a.limit else f"{name}_smoke"
    pairs = data.pairs()
    if a.limit:
        pairs = pairs.sample(a.limit, random_state=0)
    path = data.RUNS / f"{stem}.parquet"
    old = None
    if path.exists():  # incremental: score only pairs the stored run lacks
        old = pd.read_parquet(path).merge(pairs[data.KEY], on=data.KEY)
        m = pairs.merge(old[data.KEY], on=data.KEY, how="left", indicator=True)["_merge"].eq("left_only").to_numpy()
        pairs = pairs[m].reset_index(drop=True)
        print(f"{len(old)} cached, {len(pairs)} to score", flush=True)
        if not len(pairs):
            return
    ck = Checkpoint(a.model)
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    opts = replace(LoadOptions.from_env(), dtype=dtype, temperature=1.0, cuda_graphs=True, fused=fused_available())
    tok, model = ck.load("cuda", opts)
    order = np.argsort(pairs["passage"].str.len().to_numpy(), kind="stable")  # similar lengths per batch
    noul, score = np.empty(len(pairs)), np.empty((len(pairs), 4))
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(order), CHUNK):
            idx = order[s:s + CHUNK]
            encs, metas = [], []
            for i in idx:
                rec, meta = to_record(SystemOneRequest(model="kev", state=state(pairs["query"].iat[i], pairs["passage"].iat[i]),
                                                       questions=QUESTIONS))
                encs.append(admit(model, tok, rec)); metas.append(meta)
            probs, _ = model.probs_batch(encs, [None] * len(encs), [False] * len(encs))
            graphs = getattr(model, "graphs", None)   # capture shapes that ran eagerly, as kev.serve does when idle
            if graphs is not None and graphs.capture_due(idle=True):
                graphs.capture_pending()
            for i, p, meta in zip(idx, probs, metas):
                by_id = {m["id"]: np.asarray(q.tolist() if hasattr(q, "tolist") else q, float) for m, q in zip(meta, p)}
                noul[i] = by_id["noul"][1]                     # options are [no, yes]
                score[i] = by_id["score"]
            done = s + len(idx)
            if done % (CHUNK * 100) < CHUNK or done == len(order):
                el = time.time() - t0
                print(f"  {done}/{len(order)} pairs, {done / el:.0f}/s, ETA {(len(order) - done) / (done / el) / 60:.1f} min", flush=True)
    out = pairs[data.KEY].copy()
    out["kev_noul_p_raw"] = noul
    for y in range(4):
        out[f"kev_score_p{y}_raw"] = score[:, y]
    data.RUNS.mkdir(exist_ok=True)
    out = pd.concat([x for x in (old, out) if x is not None], ignore_index=True)
    out.to_parquet(path, index=False)
    meta = {"model": a.model, "shipped_temperature": float(ck.meta.temperature), "pairs": len(out),
            "seconds_last_run": round(time.time() - t0, 1), "dtype": a.dtype, "gpu": torch.cuda.get_device_name()}
    (data.RUNS / f"{stem}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    os.environ.setdefault("KEV_TEMPERATURE", "1.0")
    main()

"""Score every pair (official pool + judged) with one model; write raw logits to runs/MODEL.parquet.

Incremental: rows already in runs/MODEL.parquet for pairs still in the pair set are kept, and only
the missing pairs are scored (a model's output for a pair does not depend on the other pairs).

usage: score.py {laya,monot5,qwen3} [--limit N]
"""

import argparse
import json
import time

import pandas as pd
import torch

from decisionrerank import data
from decisionrerank.scorers import LAYA_ID, MONOT5_ID, QWEN3_ID, SCORERS

IDS = {"laya": LAYA_ID, "monot5": MONOT5_ID, "qwen3": QWEN3_ID}


def revision(model_id: str) -> str | None:
    from huggingface_hub import scan_cache_dir

    for repo in scan_cache_dir().repos:
        if repo.repo_id == model_id:
            return sorted(r.commit_hash for r in repo.revisions)[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(SCORERS))
    ap.add_argument("--limit", type=int, help="score only the first N pairs (smoke test)")
    a = ap.parse_args()
    pairs = data.pairs()
    stem = a.model if not a.limit else f"{a.model}_smoke"
    if a.limit:
        pairs = pairs.head(a.limit)
    path = data.RUNS / f"{stem}.parquet"
    old = pd.read_parquet(path) if path.exists() else None
    todo = pairs
    if old is not None:
        old = old.merge(pairs[data.KEY], on=data.KEY)
        todo = pairs.merge(old[data.KEY], on=data.KEY, how="left", indicator=True)
        todo = pairs[(todo["_merge"] == "left_only").to_numpy()]
    print(f"{a.model}: {len(pairs)} pairs, {len(pairs) - len(todo)} cached, {len(todo)} to score", flush=True)
    t = time.time()
    new = todo[data.KEY].join(SCORERS[a.model](todo)) if len(todo) else None
    secs = time.time() - t
    out = pd.concat([x for x in (old, new) if x is not None], ignore_index=True)
    out = pairs[data.KEY].merge(out, on=data.KEY, how="left", validate="1:1")
    assert out.drop(columns=data.KEY).notna().all().all(), "unscored pairs left"
    data.RUNS.mkdir(exist_ok=True)
    out.to_parquet(path, index=False)
    meta = {"model": IDS[a.model], "revision": revision(IDS[a.model]), "pairs": len(out),
            "scored_this_run": len(todo), "seconds_this_run": round(secs, 1),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    (data.RUNS / f"{stem}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()

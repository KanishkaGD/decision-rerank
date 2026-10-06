"""Ask Jev (via OpenRouter) both questions for every pair; write runs/jev.parquet.

usage: score_jev.py [--limit N] [--workers 32] [--budget 5.0]
Reads OPENROUTER_API_KEY from the environment or from .env. Responses are cached in
runs/jev_responses.jsonl, so reruns only request missing pairs.
"""

import argparse
import os

from decisionrerank import data, jev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--budget", type=float, default=5.0, help="stop submitting after this many USD")
    a = ap.parse_args()
    if "OPENROUTER_API_KEY" not in os.environ and (data.ROOT / ".env").exists():
        for line in (data.ROOT / ".env").read_text().splitlines():
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())
    pairs = data.pairs()
    if a.limit:
        pairs = pairs.sample(a.limit, random_state=0)
    # whole queries in random order: at any point the finished queries are a random query sample
    order = pairs[["qset", "query_id"]].drop_duplicates().sample(frac=1.0, random_state=0).reset_index(drop=True)
    order["q_order"] = order.index
    pairs = pairs.merge(order, on=["qset", "query_id"]).sort_values(["q_order", "doc_id"]).drop(columns="q_order")
    jev.score(pairs, workers=a.workers, budget_usd=a.budget)
    out = jev.to_frame()
    out.to_parquet(data.RUNS / "jev.parquet", index=False)
    full = out.merge(data.pairs()[data.KEY], on=data.KEY)
    print(f"runs/jev.parquet: {len(out)} pairs ({len(full)} in the current pair set), "
          f"models {sorted(out.jev_model.unique())}, total cost ${out.jev_cost.sum():.3f}")


if __name__ == "__main__":
    main()

"""Ask a decoder-only LLM (OpenRouter) both relevance questions for every pair; write runs/<slug>.parquet.

usage: score_llm.py [--model qwen/qwen3.7-flash] [--prefix qf] [--workers 24] [--budget 3.0]
Order: the standard BM25 run's top 100 first (for QPP), then the remaining pool and judged pairs, whole queries
in random order. Responses are cached in runs/<slug>_responses.jsonl, so reruns only request missing pairs.
"""

import argparse
import os

import pandas as pd

from decisionrerank import data, llm_judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen/qwen3.7-flash")
    ap.add_argument("--prefix", default="qf")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--budget", type=float, default=3.0)
    a = ap.parse_args()
    if "OPENROUTER_API_KEY" not in os.environ and (data.ROOT / ".env").exists():
        for line in (data.ROOT / ".env").read_text().splitlines():
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())
    q = data.queries().set_index(["qset", "query_id"]).text
    std = data.bm25_std()
    s100 = std[std.std_rank <= 100][data.KEY].assign(query=lambda f: q.loc[list(zip(f.qset, f.query_id))].to_numpy())
    s100["passage"] = data.passages().loc[s100.doc_id].to_numpy()
    rest = data.pairs()
    order = rest[["qset", "query_id"]].drop_duplicates().sample(frac=1.0, random_state=0).reset_index(drop=True)
    order["q_order"] = order.index
    rest = rest.merge(order, on=["qset", "query_id"]).sort_values(["q_order", "doc_id"])[data.KEY + ["query", "passage"]]
    allp = pd.concat([s100, rest], ignore_index=True).drop_duplicates(data.KEY)
    llm_judge.score(a.model, allp, workers=a.workers, budget_usd=a.budget)
    out = llm_judge.to_frame(a.model, a.prefix)
    out.to_parquet(data.RUNS / f"{llm_judge.slug(a.model)}.parquet", index=False)
    print(f"runs/{llm_judge.slug(a.model)}.parquet: {len(out)} pairs, total ${out[f'{a.prefix}_cost'].sum():.3f}, "
          f"NaN noul {out[f'{a.prefix}_noul_p'].isna().sum()}, NaN score {out[f'{a.prefix}_score_p0'].isna().sum()}")


if __name__ == "__main__":
    main()

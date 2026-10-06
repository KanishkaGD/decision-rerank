"""Expected nDCG@10 of the standard BM25 run from Jev's judgments of its top d passages.

nDCG@10 divides DCG@10 by the query's ideal DCG@10, which depends on relevant passages below rank 10. A
predictor that reads only the top 10 estimates DCG, not nDCG. Here Jev judges the run's top d (10, 20, 50,
100) and the ideal DCG is estimated from those judgments, as QPP-GenRE does with LLM judgments of the top 200.
Three estimators of nDCG@10, all with linear gains (trec_eval):
- plug-in: E[grade] as each passage's gain; ideal DCG = DCG of the 10 largest E[grade] in the top d;
- argmax (QPP-GenRE style): each passage gets its most likely grade; nDCG@10 computed from those labels;
- sampled: the expectation of nDCG@10 over grades drawn from Jev's distributions (500 draws per query).
Writes results/qpp_depth.{md,csv}. Needs OPENROUTER_API_KEY (.env) for top-100 pairs Jev has not answered.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from decisionrerank import data, jev
from decisionrerank.probs import jev_frame

sys.path.insert(0, os.path.dirname(__file__))
from qpp import std_actual  # noqa: E402

DEPTHS = (10, 20, 50, 100)
DRAWS = 500
DISC = 1 / np.log2(np.arange(2, 12))
QPP_GENRE = {"dl19": (0.474, 0.724), "dl20": (0.469, 0.638)}  # nDCG@10, 200 judged; tau, r (Meng et al., Table 3)


def ndcg10(gains: np.ndarray) -> np.ndarray:
    """nDCG@10 of rank-ordered gains (..., d), ideal from the same d gains; 0 when nothing is relevant."""
    dcg = gains[..., :10] @ DISC[: min(10, gains.shape[-1])]
    ideal = -np.sort(-gains, axis=-1)[..., :10] @ DISC[: min(10, gains.shape[-1])]
    return np.divide(dcg, ideal, out=np.zeros_like(dcg, dtype=float), where=ideal > 0)


def main():
    if "OPENROUTER_API_KEY" not in os.environ and (data.ROOT / ".env").exists():
        for line in (data.ROOT / ".env").read_text().splitlines():
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())
    std = data.bm25_std()
    t = std[std.std_rank <= max(DEPTHS)].copy()
    t["query"] = data.queries().set_index(["qset", "query_id"]).text.loc[list(zip(t.qset, t.query_id))].to_numpy()
    t["passage"] = data.passages().loc[t.doc_id].to_numpy()
    jev.score(t, workers=64, budget_usd=1.0)
    t = t.merge(jev_frame(jev.to_frame()), on=data.KEY, validate="1:1").sort_values(["query_id", "std_rank"])
    assert t.groupby("query_id").size().eq(max(DEPTHS)).all(), "missing Jev answers"

    rng, preds = np.random.default_rng(0), {}
    cols = [f"jev_score_p{y}" for y in range(4)]
    for qid, g in t.groupby("query_id", sort=False):
        P = g[cols].to_numpy(float)
        top10 = P[:10] @ np.arange(4.0)
        row = {"E[DCG@10] (top 10 only)": float(top10 @ DISC)}
        for d in DEPTHS:
            Pd = P[:d]
            row[f"plug-in, d={d}"] = float(ndcg10(Pd @ np.arange(4.0)))
            row[f"argmax, d={d}"] = float(ndcg10(Pd.argmax(1).astype(float)))
            draws = (rng.random((DRAWS, d))[..., None] > np.cumsum(Pd, 1)[None]).sum(-1).astype(float)
            row[f"sampled, d={d}"] = float(ndcg10(draws).mean())
        preds[qid] = row
    P = pd.DataFrame.from_dict(preds, orient="index")
    A = std_actual(data.qrels())["nDCG@10"]
    qset = std.drop_duplicates("query_id").set_index("query_id").qset

    rows = []
    for name in P:
        row = {"predictor": name}
        for q in data.QSETS:
            ids = qset[qset == q].index.intersection(P.index)
            row[f"tau {q}"] = stats.kendalltau(P.loc[ids, name], A.loc[ids]).statistic
            row[f"r {q}"] = stats.pearsonr(P.loc[ids, name], A.loc[ids]).statistic
        row["tau mean"] = (row["tau dl19"] + row["tau dl20"]) / 2
        row["r mean"] = (row["r dl19"] + row["r dl20"]) / 2
        if not name.startswith("E[DCG"):
            row["MAE nDCG@10"] = float((P[name] - A.loc[P.index]).abs().mean())
        rows.append(row)
    res = pd.DataFrame(rows)
    best = res[res.predictor != "E[DCG@10] (top 10 only)"].sort_values("tau mean").predictor.iloc[-1]
    cis = {}
    for q in data.QSETS:
        ids = qset[qset == q].index.intersection(P.index)
        x, y = P.loc[ids, best].to_numpy(), A.loc[ids].to_numpy()
        b = [stats.kendalltau(x[s], y[s]).statistic for s in (rng.integers(0, len(ids), len(ids)) for _ in range(2000))]
        cis[q] = np.percentile(b, [2.5, 97.5])
    res.to_csv(data.RESULTS / "qpp_depth.csv", index=False)
    spent = jev.to_frame().merge(t[data.KEY], on=data.KEY).jev_cost.sum()
    md = ["# Expected nDCG@10 of the standard BM25 run from Jev judgments at depth d", "",
          f"Jev cost of all top-100 answers: ${spent:.3f}. Kendall tau / Pearson r with the true nDCG@10.", "",
          "| Predictor | tau DL19 | tau DL20 | tau mean | r DL19 | r DL20 | r mean | MAE nDCG@10 |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, r in res.iterrows():
        mae = "" if pd.isna(r.get("MAE nDCG@10")) else f"{r['MAE nDCG@10']:.3f}"
        md.append(f"| {r.predictor} | {r['tau dl19']:.3f} | {r['tau dl20']:.3f} | {r['tau mean']:.3f} | {r['r dl19']:.3f} | "
                  f"{r['r dl20']:.3f} | {r['r mean']:.3f} | {mae} |")
    md.append(f"| QPP-GenRE (fine-tuned LLaMA-7B, 200 judged; published) | {QPP_GENRE['dl19'][0]:.3f} | {QPP_GENRE['dl20'][0]:.3f} | "
              f"{np.mean([v[0] for v in QPP_GENRE.values()]):.3f} | {QPP_GENRE['dl19'][1]:.3f} | {QPP_GENRE['dl20'][1]:.3f} | "
              f"{np.mean([v[1] for v in QPP_GENRE.values()]):.3f} | |")
    md += ["", f"Best Jev estimator by mean tau: {best}; 95% bootstrap CI of tau: DL19 [{cis['dl19'][0]:.3f}, {cis['dl19'][1]:.3f}], "
               f"DL20 [{cis['dl20'][0]:.3f}, {cis['dl20'][1]:.3f}].", ""]
    (data.RESULTS / "qpp_depth.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()

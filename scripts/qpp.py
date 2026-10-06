"""Query performance prediction from relevance probabilities. Writes results/qpp.{md,csv}.

Targets: per-query nDCG@10 and P@10 (rel >= 2) of three top-10 lists: the standard BM25 run (Anserini default,
k1=0.9, b=0.4, the run QPP papers such as QPP-GenRE predict; 99.7% judged), our bm25s ordering of the official
pool, and the monoT5 reranking of that pool (98-99% judged).

Predictors read a model's answers on that top 10, never the labels:
- expected P@10 = sum of P(relevant) over the top 10 / 10;
- expected DCG@10 = sum of E[grade] / log2(rank + 1), from a 4-level grade distribution.
Probabilities come as shipped, or after cross-fitted Platt scaling (fit on the other collection's judged
pairs; needs labels). Baselines: BM25 score spread (sigma over the top 100), an NQC-style ratio (sigma /
mean pool score), the top BM25 score, and the mean monoT5 log-odds of the top 10.

Evaluation: Kendall tau and Pearson r with the true per-query metric, per collection (the QPP convention)
and their mean; mean absolute error of expected vs actual P@10 (does the predictor get the value right?).
"""

import ir_measures
import numpy as np
import pandas as pd
from ir_measures import P as Prec
from ir_measures import nDCG
from scipy import stats

from decisionrerank import data
from decisionrerank.effectiveness import per_query
from decisionrerank.probs import crossfit_platt, logit, systems

DISC = 1 / np.log2(np.arange(2, 12))
PROB = {  # label -> (column, needs labels?)
    "Jev P(yes)": ("jev_noul_p", False),
    "Laya P(yes) (shipped T)": ("noul_p_shipped", False),
    "Kev-0.8B P(yes) (shipped T)": ("kev_noul_p", False),
    "monoT5 P(true), raw": ("monot5_p", False),
    "Qwen3-0.6B P(yes), raw": ("qwen3_p", False),
    "monoT5 P(true), Platt": ("monot5_platt", True),
    "Qwen3-0.6B P(yes), Platt": ("qwen3_platt", True),
    "Jev P(yes), Platt": ("jev_platt", True),
}
GRADES = {"Jev E[grade]": "jev_lin", "Laya E[grade] (shipped T)": "score_lin_shipped", "Kev-0.8B E[grade]": "kev_lin"}


def top10(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    f = frame[frame.in_pool].sort_values(["query_id", col, "bm25_rank"], ascending=[True, False, True])
    f = f.assign(pos=f.groupby("query_id").cumcount())
    return f[f.pos < 10]


def std_top10(frame: pd.DataFrame) -> pd.DataFrame:
    std = data.bm25_std()
    t = std[std.std_rank <= 10].merge(frame.drop(columns=["in_pool"]), on=data.KEY, validate="1:1")
    return t.assign(pos=t.std_rank - 1, in_pool=True)


def std_actual(qrels: pd.DataFrame) -> pd.DataFrame:
    """nDCG@10 and P@10 (rel >= 2) of the full standard run, per query."""
    std, rows = data.bm25_std(), []
    for qs in data.QSETS:
        qr, run = {}, {}
        for q, d, r in qrels[qrels.qset == qs][["query_id", "doc_id", "rel"]].itertuples(index=False):
            qr.setdefault(q, {})[d] = int(r)
        for q, d, sc in std[std.qset == qs][["query_id", "doc_id", "std_score"]].itertuples(index=False):
            run.setdefault(q, {})[d] = float(sc)
        names = {repr(nDCG @ 10): "nDCG@10", repr(Prec(rel=data.REL_MIN) @ 10): "P@10"}
        rows += [(m.query_id, names[repr(m.measure)], m.value)
                 for m in ir_measures.iter_calc([nDCG @ 10, Prec(rel=data.REL_MIN) @ 10], qr, run)]
    return pd.DataFrame(rows, columns=["query_id", "m", "v"]).pivot(index="query_id", columns="m", values="v")


def predictors(t: pd.DataFrame, pool: pd.DataFrame, which: str) -> pd.DataFrame:
    g = t.groupby("query_id")
    out = pd.DataFrame(index=g.size().index)
    for name, (col, _) in PROB.items():
        out[name] = g[col].sum() / 10
    for name, col in GRADES.items():
        out[name] = (t[col] * DISC[t.pos]).groupby(t.query_id).sum()
    pg = pool[pool.in_pool].groupby("query_id")
    if which == "BM25 standard run":
        std = data.bm25_std()
        top100 = std[std.std_rank <= 100].groupby("query_id").std_score
        out["BM25 sigma@100"] = top100.std()
        out["BM25 NQC-style (sigma@100 / mean@1000)"] = top100.std() / std.groupby("query_id").std_score.mean()
        out["BM25 top score"] = top100.max()
    if which == "BM25 (our pool order)":
        top100 = pool[pool.in_pool & (pool.bm25_rank <= 100)].groupby("query_id").bm25
        out["BM25 sigma@100"] = top100.std()
        out["BM25 NQC-style (sigma@100 / pool mean)"] = top100.std() / pg.bm25.mean()
        out["BM25 top score"] = top100.max()
    out["monoT5 mean log-odds@10"] = g.monot5_z.mean()
    return out


def actual(t: pd.DataFrame, frame: pd.DataFrame, col: str, qrels: pd.DataFrame) -> pd.DataFrame:
    rel = t.rel.fillna(0)
    out = pd.DataFrame({"P@10": (rel >= data.REL_MIN).groupby(t.query_id).sum() / 10})
    nd = pd.concat([per_query(frame[frame.qset == q], col, qrels[qrels.qset == q]) for q in data.QSETS])["nDCG@10"]
    return out.join(nd)


def main():
    frame, _ = systems()
    for name, z in (("monot5", frame.monot5_z), ("qwen3", frame.qwen3_z), ("jev", logit(frame.jev_noul_p.to_numpy()))):
        frame[f"{name}_platt"], _ = crossfit_platt(frame, np.asarray(z, float))
    qrels, qset = data.qrels(), frame.drop_duplicates("query_id").set_index("query_id").qset
    rows, md = [], ["# Query performance prediction from relevance probabilities", "",
                    "Kendall tau / Pearson r with the true per-query metric; per collection and their mean. "
                    "MAE: |expected P@10 - actual P@10| (probability predictors only). "
                    "'Labels' marks predictors that need judged pairs (cross-fitted on the other collection).", ""]
    for which, col in (("BM25 standard run", None), ("BM25 (our pool order)", "bm25"), ("monoT5", "monot5_p")):
        if col is None:
            t, A = std_top10(frame), std_actual(qrels)
        else:
            t = top10(frame, col)
            A = actual(t, frame, col, qrels)
        P = predictors(t, frame, which)
        for name in P:
            row = {"list": which, "predictor": name,
                   "labels": PROB.get(name, ("", False))[1], "kind": "prob" if name in PROB else "grade" if name in GRADES else "score"}
            for target in ("nDCG@10", "P@10"):
                taus, rs = [], []
                for q in data.QSETS:
                    ids = qset[qset == q].index.intersection(P.index)
                    taus.append(stats.kendalltau(P.loc[ids, name], A.loc[ids, target]).statistic)
                    rs.append(stats.pearsonr(P.loc[ids, name], A.loc[ids, target]).statistic)
                    row[f"tau {target} {q}"], row[f"r {target} {q}"] = taus[-1], rs[-1]
                row[f"tau {target} mean"], row[f"r {target} mean"] = np.mean(taus), np.mean(rs)
            if name in PROB:
                row["MAE P@10"] = float((P[name] - A["P@10"]).abs().mean())
                row["mean predicted P@10"], row["mean actual P@10"] = float(P[name].mean()), float(A["P@10"].mean())
            rows.append(row)
        res = pd.DataFrame([r for r in rows if r["list"] == which]).sort_values("tau nDCG@10 mean", ascending=False)
        md += [f"## {which} top 10 (mean actual nDCG@10 {A['nDCG@10'].mean():.3f}, P@10 {A['P@10'].mean():.3f})", "",
               "| Predictor | Labels | tau nDCG@10 DL19 | DL20 | mean | r nDCG@10 mean | tau P@10 mean | MAE P@10 |",
               "| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for _, r in res.iterrows():
            mae = f"{r['MAE P@10']:.3f}" if pd.notna(r.get("MAE P@10")) else ""
            md.append(f"| {r.predictor} | {'yes' if r.labels else ''} | {r['tau nDCG@10 dl19']:.3f} | {r['tau nDCG@10 dl20']:.3f} | "
                      f"{r['tau nDCG@10 mean']:.3f} | {r['r nDCG@10 mean']:.3f} | {r['tau P@10 mean']:.3f} | {mae} |")
        md.append("")
    pd.DataFrame(rows).to_csv(data.RESULTS / "qpp.csv", index=False)
    (data.RESULTS / "qpp.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()

"""RQ1: effectiveness of every reranker on the official TREC DL rerank pool, per collection and pooled.

Writes results/effectiveness.{csv,md}, results/per_query.csv and results/score_rules.csv.
Significance: paired two-sided t-tests over queries; Holm correction within each
(collection, measure) family of comparisons.
"""

import numpy as np
import pandas as pd

from decisionrerank import data
from decisionrerank.effectiveness import MEASURES, holm, paired_t, per_query, tie_rate
from decisionrerank.probs import CONDITIONS, SCORE_RULES, systems

SYSTEMS = {  # label -> ranking column (Laya rows use the raw condition; see score_rules.csv)
    "BM25 (bm25s order of the pool)": "bm25",
    "monoT5-base": "monot5_p",
    "Qwen3-Reranker-0.6B": "qwen3_p",
    "Laya noul: P(yes)": "noul_p_raw",
    "Laya score: E[y]": "score_lin_raw",
    "Laya score: E[2^y-1]": "score_exp_raw",
    "Laya score: P(y>=2)": "score_ge2_raw",
    "Jev noul: P(yes)": "jev_noul_p",
    "Jev noul, ties by E[y]": "jev_noul_lin",
    "Jev score: E[y]": "jev_lin",
    "Kev-0.8B noul: P(yes)": "kev_noul_p",
    "Kev-0.8B score: E[y]": "kev_lin",
}
BASELINES = {"BM25 (bm25s order of the pool)": "a", "monoT5-base": "b", "Qwen3-Reranker-0.6B": "c",
             "Laya noul: P(yes)": "d"}  # superscript marks
QSETS = {"DL19": ["dl19"], "DL20": ["dl20"], "DL19+20": ["dl19", "dl20"]}


def main():
    frame, temps = systems()
    for name, col in list(SYSTEMS.items()):  # Jev rows only once every pool pair has an answer
        if col not in frame or frame.loc[frame.in_pool, col].isna().any():
            print(f"skipping {name}: no complete run")
            del SYSTEMS[name]
    qrels = data.qrels()
    pq = {}  # (system, coll) -> per-query frame
    for name, col in SYSTEMS.items():
        for qs in data.QSETS:
            f = frame[frame.qset == qs]
            pq[name, qs] = per_query(f, col, qrels[qrels.qset == qs]).assign(qset=qs)
    for name in SYSTEMS:
        pq[name, "both"] = pd.concat([pq[name, q] for q in data.QSETS])
    rows, prows = [], []
    for coll, members in QSETS.items():
        key = members[0] if len(members) == 1 else "both"
        for measure in MEASURES:
            tests = []
            for name in SYSTEMS:
                rows.append({"collection": coll, "system": name, "measure": measure,
                             "value": pq[name, key][measure].mean()})
                for base in BASELINES:
                    if base == name or (name in BASELINES and BASELINES[base] != "a"):
                        continue
                    tests.append((name, base, paired_t(pq[name, key][measure], pq[base, key][measure])))
            adj = holm([t[2] for t in tests])
            prows += [{"collection": coll, "measure": measure, "system": s, "vs": b, "p": p, "p_holm": a}
                      for (s, b, p), a in zip(tests, adj)]
    res, pv = pd.DataFrame(rows), pd.DataFrame(prows)
    data.RESULTS.mkdir(exist_ok=True)
    res.to_csv(data.RESULTS / "effectiveness.csv", index=False)
    pv.to_csv(data.RESULTS / "effectiveness_pvalues.csv", index=False)
    pd.concat([pq[n, "both"].assign(system=n) for n in SYSTEMS]).to_csv(data.RESULTS / "per_query.csv")

    # Score-rule x temperature sensitivity (the one place calibration can move the ranking)
    srows = []
    for rule in SCORE_RULES:
        for c in CONDITIONS:
            for qs in data.QSETS:
                f = frame[frame.qset == qs]
                v = per_query(f, f"score_{rule}_{c}", qrels[qrels.qset == qs])["nDCG@10"].mean()
                srows.append({"rule": rule, "condition": c, "collection": qs, "nDCG@10": v})
    srows = pd.DataFrame(srows)
    srows.to_csv(data.RESULTS / "score_rules.csv", index=False)

    ties = {name: tie_rate(frame, col) for name, col in SYSTEMS.items()}
    # tie robustness: the same rankings with ties broken at random instead of by BM25 (mean of 20 draws)
    rng, trows = np.random.default_rng(0), []
    for name, col in SYSTEMS.items():
        if ties[name] < 0.05 or name.startswith("BM25"):
            continue
        vals = []
        for _ in range(20):
            f = frame.assign(rand_tie=rng.random(len(frame)))
            vals.append(pd.concat([per_query(f[f.qset == qs], col, qrels[qrels.qset == qs], tie="rand_tie")
                                   for qs in data.QSETS])[["nDCG@10", "nDCG'@10"]].mean())
        v = pd.concat(vals, axis=1).mean(axis=1)
        trows.append({"system": name, "share tied": round(ties[name], 3),
                      "nDCG@10 (BM25 tie-break)": res[(res.collection == "DL19+20") & (res.system == name) & (res.measure == "nDCG@10")].value.item(),
                      "nDCG@10 (random tie-break)": v["nDCG@10"], "nDCG'@10 (random tie-break)": v["nDCG'@10"]})
    trows = pd.DataFrame(trows)
    trows.to_csv(data.RESULTS / "tie_robustness.csv", index=False)
    md = ["# Effectiveness (official TREC DL rerank pool, up to 1000 per query)", "",
          "Superscripts: significantly different (paired t-test, Holm-corrected p < 0.05) from "
          "a = BM25, b = monoT5-base, c = Qwen3-Reranker-0.6B, d = Laya noul.", ""]
    for coll in QSETS:
        md += [f"## {coll}", "", "| System | " + " | ".join(MEASURES) + " |",
               "| --- | " + " | ".join("---:" for _ in MEASURES) + " |"]
        for name in SYSTEMS:
            cells = []
            for measure in MEASURES:
                v = res[(res.collection == coll) & (res.system == name) & (res.measure == measure)].value.item()
                sig = pv[(pv.collection == coll) & (pv.measure == measure) & (pv.system == name) & (pv.p_holm < 0.05)]
                marks = "".join(sorted(BASELINES[b] for b in sig.vs)) if measure != "Judged@10" else ""
                cells.append(f"{v:.3f}" + (f"<sup>{marks}</sup>" if marks else ""))
            md.append(f"| {name} | " + " | ".join(cells) + " |")
        md.append("")
    md += ["## Laya score: ranking rule x temperature (nDCG@10)", "",
           srows.pivot_table(index=["rule", "condition"], columns="collection", values="nDCG@10")
           .round(4).to_markdown(), "",
           "## Ties among a query's pool scores (broken by BM25 rank)", "",
           pd.Series(ties, name="share of pairs tied").round(4).to_markdown(), "",
           "## Tie robustness (DL19+20): random tie-break, mean of 20 draws", "",
           trows.round(4).to_markdown(index=False) if len(trows) else "(no system with > 5% ties)", "",
           "## Temperatures (keyed by the collection they are applied to)", "", "```", str(temps), "```", ""]
    (data.RESULTS / "effectiveness.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()

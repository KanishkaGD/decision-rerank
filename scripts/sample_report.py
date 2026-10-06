"""Quick comparison on the queries Jev has fully answered so far (score_jev.py runs whole queries in
random order, so these are a random query sample). Writes results/sample_report.md.

Same measures as evaluate.py / calibrate.py, restricted to the sampled queries; Laya temperatures
are still cross-fitted on the full other collection.
"""

import time

import numpy as np
import pandas as pd

from decisionrerank import calibration as cal
from decisionrerank import data, jev
from decisionrerank.effectiveness import holm, paired_t, per_query, tie_rate
from decisionrerank.probs import jev_frame, systems

RANK = {"BM25": "bm25", "monoT5-base": "monot5_p", "Qwen3-Reranker-0.6B": "qwen3_p",
        "Laya noul": "noul_p_raw", "Laya score E[y]": "score_lin_raw",
        "Jev noul": "jev_noul_p", "Jev noul, ties by E[y]": "jev_noul_lin", "Jev score E[y]": "jev_lin"}
PROB = {"monoT5-base": "monot5_p", "Qwen3-Reranker-0.6B": "qwen3_p", "Laya noul (shipped T)": "noul_p_shipped",
        "Laya score P(y>=2) (raw)": "score_ge2_raw", "Jev noul": "jev_noul_p", "Jev score P(y>=2)": "jev_ge2"}
REFS = ["BM25", "monoT5-base", "Qwen3-Reranker-0.6B", "Laya noul"]
MEAS = ["nDCG@10", "nDCG'@10", "RR@10", "Judged@10"]


def main():
    frame, _ = systems()
    frame = frame.drop(columns=[c for c in frame if c.startswith("jev_")])
    frame = frame.merge(jev_frame(jev.to_frame()), on=data.KEY, how="left")
    done = frame.groupby(["qset", "query_id"]).jev_noul_p.apply(lambda s: s.notna().all())
    sample = done[done].index
    f = frame.set_index(["qset", "query_id"]).loc[sample].reset_index()
    n = f.groupby("qset").query_id.nunique().to_dict()
    qrels = data.qrels()
    pq = {}
    for name, col in RANK.items():
        pq[name] = pd.concat([per_query(f[f.qset == qs], col, qrels[qrels.qset == qs]) for qs in n])
    md = [f"# Sample comparison ({time.strftime('%Y-%m-%d %H:%M')})", "",
          f"{len(sample)} queries Jev has fully answered (DL19 {n.get('dl19', 0)}, DL20 {n.get('dl20', 0)}), "
          "a random query sample; official rerank pool. Superscripts: Holm-corrected paired t-test p < 0.05 vs "
          "a = BM25, b = monoT5-base, c = Qwen3-Reranker-0.6B, d = Laya noul.", "",
          "## Effectiveness", "", "| System | " + " | ".join(MEAS) + " | ties |", "| --- |" + " ---: |" * (len(MEAS) + 1)]
    for m in MEAS:
        tests = [(s, b, paired_t(pq[s][m], pq[b][m])) for s in RANK for b in REFS
                 if s != b and not (s in REFS[:3] and b != "BM25") and not (s == "Laya noul" and b == "Laya noul")]
        for (s, b, _), adj in zip(tests, holm([t[2] for t in tests])):
            pq[s].attrs.setdefault(m, set())
            if adj < 0.05 and m != "Judged@10":
                pq[s].attrs[m].add("abcd"[REFS.index(b)])
    for name, col in RANK.items():
        cells = [f"{pq[name][m].mean():.3f}" + (f"<sup>{''.join(sorted(pq[name].attrs.get(m, [])))}</sup>"
                                               if pq[name].attrs.get(m) else "") for m in MEAS]
        md.append(f"| {name} | " + " | ".join(cells) + f" | {tie_rate(f, col):.2f} |")
    j = f[f.rel.notna()].reset_index(drop=True)
    y = (j.rel >= data.REL_MIN).to_numpy(float)
    groups = cal.query_groups(j)
    md += ["", f"## Calibration (judged pairs of the sample, n = {len(j):,}, rel >= 2 rate {y.mean():.3f})", "",
           "| System | ECE [95% CI] | Brier | NLL | AUROC | mean P |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, col in PROB.items():
        p = j[col].to_numpy(float)
        lo, hi = cal.bootstrap(lambda i: cal.ece(p[i], y[i]), groups, n=500)
        md.append(f"| {name} | {cal.ece(p, y):.3f} [{lo:.3f}, {hi:.3f}] | {cal.brier(p, y):.3f} | {cal.nll(p, y):.3f} "
                  f"| {cal.auroc(p, y):.3f} | {p.mean():.3f} |")
    P = j[[f"jev_score_p{k}" for k in range(4)]].to_numpy(float)
    L = j[[f"score_p{k}_raw" for k in range(4)]].to_numpy(float)
    g = j.rel.to_numpy(int)
    md += ["", "## Score question vs human grade", "", "| Model | RPS | Acc | Spearman E[y] |", "| --- | ---: | ---: | ---: |",
           *[f"| {nm} | {d['RPS']:.3f} | {d['Acc']:.3f} | {d['Spearman']:.3f} |"
             for nm, d in (("Laya (raw)", cal.graded(L, g)), ("Jev 1.13", cal.graded(P, g)))], ""]
    data.RESULTS.mkdir(exist_ok=True)
    (data.RESULTS / "sample_report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()

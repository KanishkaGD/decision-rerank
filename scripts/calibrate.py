"""RQ2: calibration of each model's P(relevant) against the human DL labels.

Population: every judged pair (independent of the first stage); a second table restricts to judged
pairs inside the official rerank pool. Targets: rel >= 2 (primary) and rel >= 1. A base-rate row predicts
the other collection's positive rate for every pair (the no-skill reference for Brier and NLL).
Writes results/calibration*.{csv,md}, results/reliability.csv and figures/{reliability,grades}.{png,pdf}.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from decisionrerank import calibration as cal
from decisionrerank import data
from decisionrerank.probs import CONDITIONS, systems

PROB = {  # label -> P(rel >= 2) column
    "monoT5-base": "monot5_p",
    "Qwen3-Reranker-0.6B": "qwen3_p",
    **{f"Laya noul ({c})": f"noul_p_{c}" for c in CONDITIONS},
    **{f"Laya score P(y>=2) ({c})": f"score_ge2_{c}" for c in CONDITIONS},
    "Jev noul": "jev_noul_p",
    "Jev score P(y>=2)": "jev_ge2",
    # the same unclamped cross-fitted temperature for every model's yes/no logit (probs.crossfit_binary)
    "monoT5-base (cross-fit T)": "monot5_xfit",
    "Qwen3-Reranker-0.6B (cross-fit T)": "qwen3_xfit",
    "Laya noul (cross-fit T, unclamped)": "laya_noul_xfit",
    "Jev noul (cross-fit T)": "jev_noul_xfit",
    "Kev-0.8B noul": "kev_noul_p",
    "Kev-0.8B noul raw (T = 1)": "kev_noul_p_raw",
    "Kev-0.8B score P(y>=2)": "kev_ge2",
    "Kev-0.8B noul (cross-fit T)": "kev_noul_xfit",
}
KEV = "#008300"
PAIRS = [  # paired query-bootstrap differences (first - second); negative = first is better
    ("Jev noul", "monoT5-base (cross-fit T)"),
    ("Jev noul", "Laya noul (shipped)"),
    ("Laya noul (shipped)", "monoT5-base (cross-fit T)"),
    ("Laya noul (shipped)", "monoT5-base"),
    ("Laya noul (fitted)", "Laya noul (shipped)"),
    ("Jev noul (cross-fit T)", "monoT5-base (cross-fit T)"),
    ("Kev-0.8B noul", "Jev noul"),
    ("Kev-0.8B noul", "Qwen3-Reranker-0.6B"),
    ("Kev-0.8B noul (cross-fit T)", "Qwen3-Reranker-0.6B (cross-fit T)"),
]
JEV = "#4a3aa7"
COLL = {"DL19": ["dl19"], "DL20": ["dl20"], "DL19+20": ["dl19", "dl20"]}
CI_METRICS = ("ECE", "Brier")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COND_COLOR = {"raw": "#2a78d6", "shipped": "#eb6834", "fitted": "#1baf7a"}


def base_rate(frame: pd.DataFrame, judged: pd.DataFrame, t: int) -> np.ndarray:
    """Each pair of `frame` gets the positive rate of the other collection's judged pairs."""
    rate = {q: (judged[judged.qset == q].rel >= t).mean() for q in data.QSETS}
    other = {"dl19": "dl20", "dl20": "dl19"}
    return frame.qset.map(lambda q: rate[other[q]]).to_numpy(float)


def binary_table(judged: pd.DataFrame, ci: bool) -> pd.DataFrame:
    rows = []
    for t in (2, 1):
        for coll, members in COLL.items():
            f = judged[judged.qset.isin(members)].reset_index(drop=True)
            y = (f.rel >= t).to_numpy(float)
            groups = cal.query_groups(f) if ci else None
            preds = {name: f[col].to_numpy(float) for name, col in PROB.items()}
            preds["Base rate (other collection)"] = base_rate(f, judged, t)
            for name, p in preds.items():
                row = {"target": f"rel>={t}", "collection": coll, "system": name, "n": len(f),
                       "pos_rate": y.mean(), "mean_p": p.mean(), "AUROC_query": macro_auroc(f, p, y)}
                if len(members) > 1:  # pooled ECE can hide opposite per-collection errors
                    row["ECE_wavg"] = sum(cal.ece(p[(f.qset == q).to_numpy()], y[(f.qset == q).to_numpy()])
                                          * (f.qset == q).mean() for q in members)
                for m, fn in cal.BINARY.items():
                    row[m] = np.nan if (m == "AUROC" and name.startswith("Base rate")) else fn(p, y)
                if ci and t == 2:
                    for m in CI_METRICS:
                        row[f"{m}_lo"], row[f"{m}_hi"] = cal.bootstrap(
                            lambda i, p=p, fn=cal.BINARY[m]: fn(p[i], y[i]), groups)
                rows.append(row)
    return pd.DataFrame(rows)


def macro_auroc(f: pd.DataFrame, p: np.ndarray, y: np.ndarray) -> float:
    """Mean per-query AUROC over queries with both classes (pooled AUROC mixes query base rates)."""
    vals = []
    for idx in cal.query_groups(f):
        if 0 < y[idx].sum() < len(idx):
            vals.append(cal.auroc(p[idx], y[idx]))
    return float(np.mean(vals))


def paired_table(judged: pd.DataFrame) -> pd.DataFrame:
    y = (judged.rel >= data.REL_MIN).to_numpy(float)
    groups, rows = cal.query_groups(judged), []
    halves = [(judged.qset == q).to_numpy() for q in data.QSETS]

    def ece_wavg(p, idx):
        out = 0.0
        for h in halves:
            m = idx[h[idx]]
            out += len(m) / len(idx) * cal.ece(p[m], y[m])
        return out

    metrics = {"Brier": lambda p, i: cal.brier(p[i], y[i]), "NLL": lambda p, i: cal.nll(p[i], y[i]),
               "ECE (per-collection mean)": ece_wavg}
    for a, b in PAIRS:
        if a not in PROB or b not in PROB:
            continue
        pa, pb = judged[PROB[a]].to_numpy(float), judged[PROB[b]].to_numpy(float)
        for m, fn in metrics.items():
            d = fn(pa, np.arange(len(y))) - fn(pb, np.arange(len(y)))
            lo, hi = cal.bootstrap(lambda i: fn(pa, i) - fn(pb, i), groups)
            rows.append({"first": a, "second": b, "metric": m, "diff": d, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def graded_table(judged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for coll, members in COLL.items():
        f = judged[judged.qset.isin(members)]
        g = f.rel.to_numpy(int)
        for c in CONDITIONS:
            P = f[[f"score_p{y}_{c}" for y in range(4)]].to_numpy(float)
            rows.append({"collection": coll, "condition": c, **cal.graded(P, g),
                         **{f"mean_P{y}": P[:, y].mean() for y in range(4)}})
        for pre, label in (("jev", "Jev 1.13"), ("kev", "Kev-0.8B (shipped T)")):
            if f"{pre}_score_p0" in f:
                P = f[[f"{pre}_score_p{y}" for y in range(4)]].to_numpy(float)
                rows.append({"collection": coll, "condition": label, **cal.graded(P, g),
                             **{f"mean_P{y}": P[:, y].mean() for y in range(4)}})
        prior = np.stack([np.bincount(judged[judged.qset != q].rel.astype(int), minlength=4)
                          / (judged.qset != q).sum() for q in f.qset])
        rows.append({"collection": coll, "condition": "base rate (other collection)", **cal.graded(prior, g),
                     **{f"mean_P{y}": prior[:, y].mean() for y in range(4)}})
        rows[-1]["Spearman"] = np.nan
    return pd.DataFrame(rows)


def fmt_ci(r, m):
    return f"{r[m]:.3f} [{r[f'{m}_lo']:.3f}, {r[f'{m}_hi']:.3f}]" if f"{m}_lo" in r and pd.notna(r.get(f"{m}_lo")) else f"{r[m]:.3f}"


def reliability_figure(judged: pd.DataFrame):
    y = (judged.rel >= data.REL_MIN).to_numpy(float)
    panels = [("Trained rerankers (raw)", [("monoT5-base", "monot5_p", INK2, "-"),
                                     ("Qwen3-Reranker-0.6B", "qwen3_p", INK2, "--")]),
              ("Laya noul: P(yes)", [(c, f"noul_p_{c}", COND_COLOR[c], "-") for c in CONDITIONS]),
              ("Laya score: P(y ≥ 2)", [(c, f"score_ge2_{c}", COND_COLOR[c], "-") for c in CONDITIONS]),
              ("Jev 1.13 (no temperature access)", [("noul P(yes)", "jev_noul_p", JEV, "-"),
                                                     ("score P(y ≥ 2)", "jev_ge2", JEV, "--")]),
              ("Kev-0.8B (shipped T)", [("noul P(yes)", "kev_noul_p", KEV, "-"),
                                        ("score P(y ≥ 2)", "kev_ge2", KEV, "--")])]
    panels = [pn for pn in panels if all(col in judged for _, col, _, _ in pn[1])]
    fig, axes = plt.subplots(2, len(panels), figsize=(3.4 * len(panels), 4.9), height_ratios=[3, 1], sharex=True,
                             gridspec_kw={"hspace": 0.08, "wspace": 0.12})
    rel_rows = []
    for j, (title, series) in enumerate(panels):
        ax, hx = axes[0, j], axes[1, j]
        ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, zorder=0)
        for label, col, color, ls in series:
            p = judged[col].to_numpy(float)
            r = cal.reliability(p, y)
            rel_rows.append(r.assign(system=title, series=label))
            ax.plot(r.mean_p, r.frac_pos, ls, color=color, lw=1.6, marker="o", ms=4,
                    mec="white", mew=0.8, label=f"{label}  (ECE {cal.ece(p, y):.3f})")
            hx.hist(p, bins=30, range=(0, 1), histtype="step", color=color, ls=ls, lw=1.3)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left", labelcolor=INK2)
        hx.set_yscale("log")
        hx.set_xlabel("predicted probability", fontsize=8.5, color=INK2)
        if j == 0:
            ax.set_ylabel("observed share with rel ≥ 2", fontsize=8.5, color=INK2)
            hx.set_ylabel("pairs", fontsize=8.5, color=INK2)
        else:
            ax.set_yticklabels([])
        for a in (ax, hx):
            a.tick_params(labelsize=7.5, colors=INK2, length=2)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                a.spines[s].set_color(GRID)
            a.grid(color=GRID, lw=0.5, alpha=0.6)
    fig.suptitle("Reliability on all judged DL19+20 pairs (n = 20,646), target rel ≥ 2",
                 fontsize=10.5, color=INK, x=0.125, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(data.FIGURES / f"reliability.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return pd.concat(rel_rows)


def grades_figure(judged: pd.DataFrame, temps: dict):
    conds = ("raw", "fitted") + tuple(c for c in ("jev", "kev") if f"{c}_score_p0" in judged)
    fig, axes = plt.subplots(1, len(conds), figsize=(3.55 * len(conds), 3.3), sharey=True,
                             gridspec_kw={"wspace": 0.06})
    for ax, c in zip(axes, conds):
        cols = [f"{c}_score_p{y}" for y in range(4)] if c in ("jev", "kev") else [f"score_p{y}_{c}" for y in range(4)]
        M = np.stack([judged[judged.rel == g][cols].mean().to_numpy() for g in range(4)])
        ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        for g in range(4):
            for k in range(4):
                ax.text(k, g, f"{M[g, k]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if M[g, k] > 0.55 else INK)
        n = judged.rel.value_counts().sort_index()
        ax.set_yticks(range(4), [f"{g} (n={n[g]:,})" for g in range(4)], fontsize=7.5, color=INK2)
        ax.tick_params(axis="y", labelleft=ax is axes[0])
        ax.set_xticks(range(4), range(4), fontsize=7.5, color=INK2)
        ax.set_xlabel({"jev": "Jev", "kev": "Kev"}.get(c, "Laya") + " grade y", fontsize=8.5, color=INK2)
        fitted = " / ".join(f"{t:.2f}" for t in temps["fitted"]["score"].values())  # DL19 / DL20
        ax.set_title({"raw": "Laya raw (T = 1)", "fitted": f"Laya fitted (T = {fitted})",
                      "jev": "Jev 1.13", "kev": "Kev-0.8B (shipped T)"}[c],
                     fontsize=9, color=INK2, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("human grade", fontsize=8.5, color=INK2)
    fig.suptitle("Mean P(y) of the score question, by human grade (DL19+20 judged pairs)",
                 fontsize=10, color=INK, x=0.02, y=1.04, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(data.FIGURES / f"grades.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    frame, temps = systems()
    judged = frame[frame.rel.notna()].reset_index(drop=True)
    for prefix, keys in (("jev_", ("Jev noul", "Jev score P(y>=2)", "Jev noul (cross-fit T)")),
                         ("kev_", ("Kev-0.8B noul", "Kev-0.8B noul raw (T = 1)", "Kev-0.8B score P(y>=2)",
                                   "Kev-0.8B noul (cross-fit T)"))):
        col = prefix + "noul_p"
        if col not in judged or judged[col].isna().any():  # a decision model only with a complete run
            print(f"skipping {prefix}: no complete run")
            for k in keys:
                PROB.pop(k, None)
            judged = judged.drop(columns=[c for c in judged if c.startswith(prefix)])
    data.RESULTS.mkdir(exist_ok=True), data.FIGURES.mkdir(exist_ok=True)

    main_tab = binary_table(judged, ci=True)
    top_tab = binary_table(judged[judged.in_pool].reset_index(drop=True), ci=False)
    grad_tab = graded_table(judged)
    pair_tab = paired_table(judged)
    pair_tab.to_csv(data.RESULTS / "calibration_paired.csv", index=False)
    main_tab.to_csv(data.RESULTS / "calibration.csv", index=False)
    top_tab.to_csv(data.RESULTS / "calibration_pool.csv", index=False)
    grad_tab.to_csv(data.RESULTS / "calibration_graded.csv", index=False)
    reliability_figure(judged).to_csv(data.RESULTS / "reliability.csv", index=False)
    grades_figure(judged, temps)

    cols = ["ECE", "ECE_wavg", "aECE", "Brier", "NLL", "AUROC", "AUROC_query", "mean_p"]
    md = ["# Calibration of P(relevant)", "",
          f"Laya temperatures: `{json.dumps(temps)}`", ""]
    for name, tab, ci in (("All judged pairs", main_tab, True), ("Judged pairs in the official pool", top_tab, False)):
        for t in ("rel>=2", "rel>=1"):
            for coll in COLL:
                s = tab[(tab.target == t) & (tab.collection == coll)]
                md += [f"## {name}, target {t}, {coll} (n = {s.n.iloc[0]:,}, positive rate {s.pos_rate.iloc[0]:.3f})", "",
                       "| System | " + " | ".join(cols) + " |", "| --- | " + " | ".join("---:" for _ in cols) + " |"]
                for _, r in s.iterrows():
                    cells = [fmt_ci(r, m) if (ci and m in CI_METRICS) else
                             ("" if m not in r or pd.isna(r[m]) else f"{r[m]:.3f}") for m in cols]
                    md.append(f"| {r.system} | " + " | ".join(cells) + " |")
                md.append("")
    md += ["## Paired differences (all judged pairs, rel >= 2, DL19+20; 95% query-bootstrap CI)", "",
           "Negative = the first system is better.", "",
           pair_tab.round(4).to_markdown(index=False), ""]
    md += ["## Score question: grade distribution vs human grade", "",
           grad_tab.round(3).to_markdown(index=False), ""]
    (data.RESULTS / "calibration.md").write_text("\n".join(md))
    print("\n".join(md[:60]))


if __name__ == "__main__":
    main()

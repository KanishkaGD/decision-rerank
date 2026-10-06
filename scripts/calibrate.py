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
    "Qwen3-Reranker-4B": "qwen3_p",
    **{f"Laya noul ({c})": f"noul_p_{c}" for c in CONDITIONS},
    **{f"Laya score P(y>=2) ({c})": f"score_ge2_{c}" for c in CONDITIONS},
}
COLL = {"DL19": ["dl19"], "DL20": ["dl20"], "DL19+20": ["dl19", "dl20"]}
CI_METRICS = ("ECE", "Brier")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COND_COLOR = {"raw": "#2a78d6", "shipped": "#eb6834", "fitted": "#1baf7a"}


def base_rate(frame: pd.DataFrame, t: int) -> np.ndarray:
    """Each pair gets the positive rate of the other collection's judged pairs."""
    rate = {q: (frame[frame.qset == q].rel >= t).mean() for q in data.QSETS}
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
            preds["Base rate (other collection)"] = base_rate(f, t)
            for name, p in preds.items():
                row = {"target": f"rel>={t}", "collection": coll, "system": name, "n": len(f),
                       "pos_rate": y.mean(), "mean_p": p.mean()}
                for m, fn in cal.BINARY.items():
                    row[m] = np.nan if (m == "AUROC" and np.ptp(p) == 0) else fn(p, y)
                if ci and t == 2:
                    for m in CI_METRICS:
                        row[f"{m}_lo"], row[f"{m}_hi"] = cal.bootstrap(
                            lambda i, p=p, fn=cal.BINARY[m]: fn(p[i], y[i]), groups)
                rows.append(row)
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
    panels = [("Trained rerankers", [("monoT5-base", "monot5_p", INK2, "-"),
                                     ("Qwen3-Reranker-4B", "qwen3_p", INK2, "--")]),
              ("Laya noul: P(yes)", [(c, f"noul_p_{c}", COND_COLOR[c], "-") for c in CONDITIONS]),
              ("Laya score: P(y ≥ 2)", [(c, f"score_ge2_{c}", COND_COLOR[c], "-") for c in CONDITIONS])]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 4.9), height_ratios=[3, 1], sharex=True,
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


def grades_figure(judged: pd.DataFrame):
    conds = ("raw", "fitted")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.25})
    for ax, c in zip(axes, conds):
        M = np.stack([judged[judged.rel == g][[f"score_p{y}_{c}" for y in range(4)]].mean().to_numpy()
                      for g in range(4)])
        ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        for g in range(4):
            for k in range(4):
                ax.text(k, g, f"{M[g, k]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if M[g, k] > 0.55 else INK)
        n = judged.rel.value_counts().sort_index()
        ax.set_yticks(range(4), [f"{g} (n={n[g]:,})" for g in range(4)], fontsize=7.5, color=INK2)
        ax.set_xticks(range(4), range(4), fontsize=7.5, color=INK2)
        ax.set_xlabel("Laya grade y", fontsize=8.5, color=INK2)
        ax.set_title(f"{c}", fontsize=9.5, color=INK, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("human grade", fontsize=8.5, color=INK2)
    fig.suptitle("Mean P(y) of the Laya score question, by human grade (DL19+20 judged pairs)",
                 fontsize=10, color=INK, x=0.1, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(data.FIGURES / f"grades.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    frame, temps = systems()
    judged = frame[frame.rel.notna()].reset_index(drop=True)
    data.RESULTS.mkdir(exist_ok=True), data.FIGURES.mkdir(exist_ok=True)

    main_tab = binary_table(judged, ci=True)
    top_tab = binary_table(judged[judged.in_pool].reset_index(drop=True), ci=False)
    grad_tab = graded_table(judged)
    main_tab.to_csv(data.RESULTS / "calibration.csv", index=False)
    top_tab.to_csv(data.RESULTS / "calibration_pool.csv", index=False)
    grad_tab.to_csv(data.RESULTS / "calibration_graded.csv", index=False)
    reliability_figure(judged).to_csv(data.RESULTS / "reliability.csv", index=False)
    grades_figure(judged)

    cols = ["ECE", "aECE", "Brier", "NLL", "AUROC", "mean_p"]
    md = ["# Calibration of P(relevant)", "",
          f"Laya temperatures: `{json.dumps(temps)}`", ""]
    for name, tab, ci in (("All judged pairs", main_tab, True), ("Judged pairs in the official pool", top_tab, False)):
        for t in ("rel>=2", "rel>=1"):
            for coll in COLL:
                s = tab[(tab.target == t) & (tab.collection == coll)]
                md += [f"## {name}, target {t}, {coll} (n = {s.n.iloc[0]:,}, positive rate {s.pos_rate.iloc[0]:.3f})", "",
                       "| System | " + " | ".join(cols) + " |", "| --- | " + " | ".join("---:" for _ in cols) + " |"]
                for _, r in s.iterrows():
                    cells = [fmt_ci(r, m) if (ci and m in CI_METRICS) else ("" if pd.isna(r[m]) else f"{r[m]:.3f}")
                             for m in cols]
                    md.append(f"| {r.system} | " + " | ".join(cells) + " |")
                md.append("")
    md += ["## Laya score question: grade distribution vs human grade", "",
           grad_tab.round(3).to_markdown(index=False), ""]
    (data.RESULTS / "calibration.md").write_text("\n".join(md))
    print("\n".join(md[:60]))


if __name__ == "__main__":
    main()

"""
Generate the paper's result tables (paper/tables/*.tex) from results/results.csv and results/literature.csv, so that
no number in the paper is typed by hand (CLAUDE.md, "Paper"). Standard library only, so it runs on the login node:

    python -m cdm.paper_tables            (or: make tables, in paper/)

Six tables: the triplets (RQ1's main table: per backbone the base model, its decision fine-tunes and its reranker
fine-tune; triplets.tex), the context table (every other system on the same candidates; context.tex), the fixed
cutoff (RQ2, Part A: `cdm cutoff` rows of results.csv; cutoff.tex) and the rerankers calibrated after the fact (RQ2,
Part C: `cdm posthoc` rows; posthoc.tex); for the ECIR short paper, RQ2 in one table (calibration.tex) and
the context as means only (context-short.tex).
Every reranker is read on its dataset's default candidates (BM25 top-100, docs/roadmap.md "Setup"); cells taken from
the literature (role `baseline` in results/literature.csv) win over computed ones and are marked with a dagger.
"""

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

from cdm.config import RESULTS_FOLDER
from cdm.datasets import get_dataset

TABLES_FOLDER = os.path.join("paper", "tables")

# Column order and headers: TREC DL, then BEIR.
DATASETS: List[Tuple[str, str]] = [
    ("dl19", "DL19"),
    ("dl20", "DL20"),
    ("dl21", "DL21"),
    ("dl22", "DL22"),
    ("dl23", "DL23"),
    ("trec-covid", "COVID"),
    ("nfcorpus", "NFC"),
    ("scifact", "SciF"),
    ("touche", "Touch\\'e"),
    ("dbpedia", "DBP"),
]
DL_DATASETS = [d for d, _ in DATASETS if d.startswith("dl")]

# Candidates reranked per dataset: BM25 top-100, which for DL21-23 is the official top-100 file.
FIRST_STAGE = {d: get_dataset(d).default_first_stage for d, _ in DATASETS}

# Collections without a training split in any of the fine-tunes' data: the zero-shot subset (docs/roadmap.md).
ZERO_SHOT = ["trec-covid", "touche", "dbpedia"]

Row = Tuple[str, str, str]  # system, display name, mode

# The triplets (docs/roadmap.md, "Triplets"): (backbone, [(system, display name, objective, mode)]).
TRIPLETS: List[Tuple[str, List[Tuple[str, str, str, str]]]] = [
    (
        "Qwen3-0.6B",
        [
            ("qwen3-0.6b-prompted", "Qwen3-0.6B", "base", ""),
            ("kev-0.6b-joint", "Kev-0.6B", "decision", "joint"),
            ("kev-0.6b-isolated", "Kev-0.6B", "decision", "isolated"),
            ("qwen3-reranker-0.6b", "Qwen3-Reranker-0.6B", "reranker", ""),
        ],
    ),
    (
        "Qwen3-8B",
        [
            ("qwen3-8b-prompted", "Qwen3-8B", "base", ""),
            ("kev-8b-joint", "Kev-8B", "decision", "joint"),
            ("kev-8b-isolated", "Kev-8B", "decision", "isolated"),
            ("qwen3-reranker-8b", "Qwen3-Reranker-8B", "reranker", ""),
        ],
    ),
    (
        "Qwen3.5-0.8B",
        [
            ("qwen3.5-0.8b-prompted", "Qwen3.5-0.8B", "base", ""),
            ("kev-0.8b-joint", "Kev-0.8B", "decision", "joint"),
            ("kev-0.8b-isolated", "Kev-0.8B", "decision", "isolated"),
            ("prism-0.8b", "Prism-0.8B", "reranker", ""),
        ],
    ),
    (
        "Qwen3.5-4B",
        [
            ("qwen3.5-4b-prompted", "Qwen3.5-4B", "base", ""),
            ("kev-4b-joint", "Kev-4B", "decision", "joint"),
            ("kev-4b-isolated", "Kev-4B", "decision", "isolated"),
            ("prism-4b", "Prism-4B", "reranker", ""),
        ],
    ),
    (
        "Qwen3.5-9B",
        [
            ("qwen3.5-9b-prompted", "Qwen3.5-9B", "base", ""),
            ("kev-9b-joint", "Kev-9B", "decision", "joint"),
            ("kev-9b-isolated", "Kev-9B", "decision", "isolated"),
            ("clef-flash-joint", "Clef-Flash", "decision", "joint"),
            ("clef-flash-isolated", "Clef-Flash", "decision", "isolated"),
            ("prism-9b", "Prism-9B", "reranker", ""),
            # Our reranker fine-tune of the same checkpoint (docs/notes/reranker-27b-plan.md; paper-focus.md 15).
            ("qwen3.5-9b-reranker", "Ours-9B", "reranker", ""),
        ],
    ),
    (
        "Qwen3.8-27B",
        [
            ("qwen3.8-27b-prompted", "Qwen3.8-27B", "base", ""),
            ("kev-27b-joint", "Kev-27B", "decision", "joint"),
            ("kev-27b-isolated", "Kev-27B", "decision", "isolated"),
            ("clef-joint", "Clef", "decision", "joint"),
            ("clef-isolated", "Clef", "decision", "isolated"),
        ],
    ),
]

# Row groups of the context table: (heading, [(system, display name, mode)]). The plan of record is
# docs/roadmap.md, "Systems"; systems not listed here (side experiments) are left out.
CONTEXT_ROWS: List[Tuple[str, List[Row]]] = [
    ("Listwise LLM rerankers", [("rankgpt-4", "RankGPT-4", ""), ("rankzephyr-7b", "RankZephyr-7B", "")]),
    (
        "Cross-encoders and LLM rerankers",
        [
            ("bge-m3", "bge-reranker-v2-m3", ""),
            ("monot5-3b", "monoT5-3B", ""),
            ("qwen3-reranker-8b", "Qwen3-Reranker-8B", ""),
            ("set-encoder-large", "Set-Encoder-large", ""),
        ],
    ),
    (
        "System One models",
        [
            # Joint mode only (docs/intent/paper-focus.md); both modes of Kev and Clef are in the triplet table.
            (f"{model}-joint", name, "joint")
            for model, name in [
                ("jev", "\\jev{}"),
                ("clef-flash", "Clef-Flash"),
                ("clef", "Clef"),
                ("kev-4b", "Kev-4B"),
                ("kev-27b", "Kev-27B"),
            ]
        ]
        + [("laya", "\\laya{}", "isolated")],
    ),
]

METRIC = "nDCG@10"
COST = "api_cost_usd_per_1k_queries"

Cell = Tuple[float, bool]  # value, taken from the literature
Cells = Dict[Tuple[str, str], Cell]  # (system, dataset) -> cell
Costs = Dict[Tuple[str, str], float]


def _float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_cells(results_folder: str = RESULTS_FOLDER) -> Tuple[Cells, Costs]:
    """(system, dataset) -> (nDCG@10, from literature) on the default candidates, and -> API cost per 1k queries."""
    cells: Cells = {}
    cost: Costs = {}
    with open(os.path.join(results_folder, "results.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row["subset"] or row["variant"] or row["dataset"] not in FIRST_STAGE:
                continue
            first_stage = FIRST_STAGE[row["dataset"]]
            key = (row["system"], row["dataset"])
            if row["job"] == "rerank" and row["first_stage"] == first_stage:
                value = _float(row[METRIC])
                if value is not None:
                    cells[key] = (value, False)
                if _float(row.get(COST, "")) is not None:
                    cost[key] = float(row[COST])
    # BM25 itself: the first stage's own evaluation (system bm25 / official, variant d100).
    with open(os.path.join(results_folder, "results.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row["job"] != "first-stage" or row["subset"] or row["variant"] != "d100":
                continue
            if row["dataset"] in FIRST_STAGE and f"{row['system']}-d100" == FIRST_STAGE[row["dataset"]]:
                value = _float(row[METRIC])
                if value is not None:
                    cells[("bm25", row["dataset"])] = (value, False)
    # Published numbers on the same candidates replace computed ones (CLAUDE.md, "Experiment rules").
    with open(os.path.join(results_folder, "literature.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row["role"] == "baseline" and row["metric"] == METRIC and row["candidates"] == "bm25-d100":
                cells[(row["system"], row["dataset"])] = (float(row["value"]), True)
    return cells, cost


def _fmt(value: float) -> str:
    return f"{100 * value:.1f}"


def _mean(cells: Cells, system: str, datasets: List[str]) -> Optional[float]:
    """Mean over the given datasets, only where every one is available."""
    values = [cells[(system, d)][0] for d in datasets if (system, d) in cells]
    return sum(values) / len(values) if values and len(values) == len(datasets) else None


def _header(lines: List[str], first: str, extra: List[str]) -> None:
    """Column groups (TREC DL, BEIR) and headers after `first` leading columns."""
    n_dl, n = len(DL_DATASETS), len(DATASETS)
    lines += [
        "\\toprule",
        " &" * (first.count("&") + 1) + f" \\multicolumn{{{n_dl}}}{{c}}{{TREC DL}} & "
        f"\\multicolumn{{{n - n_dl}}}{{c}}{{BEIR}}" + " &" * len(extra) + " \\\\",
        f"\\cmidrule(lr){{{first.count('&') + 2}-{first.count('&') + 1 + n_dl}}}"
        f"\\cmidrule(lr){{{first.count('&') + 2 + n_dl}-{first.count('&') + 1 + n}}}",
        first + " & " + " & ".join(h for _, h in DATASETS) + " & " + " & ".join(extra) + " \\\\",
    ]


# RQ1 significance marks in the triplet table (`cdm significance` rows): comparison -> superscript on the first system.
SIGNIFICANCE_MARKS = {"reranker-vs-decision": "\\ast", "decision-vs-base": "\\dagger", "joint-vs-isolated": "\\ddagger"}
Significance = Dict[Tuple[str, str], str]  # (system, dataset) -> marks of its significant comparisons


def load_significance(results_folder: str = RESULTS_FOLDER) -> Significance:
    """(system, dataset) -> the superscripts of the comparisons in which it differs significantly (Holm)."""
    marks: Dict[Tuple[str, str], str] = {}
    with open(os.path.join(results_folder, "results.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row["job"] != "significance" or row.get("sig_holm") != "1":
                continue
            mark = SIGNIFICANCE_MARKS.get(row.get("sig_comparison", ""))
            if mark:
                key = (row["system"], row["dataset"])
                marks[key] = marks.get(key, "") + mark
    return marks


def triplet_table(cells: Cells, significance: Optional[Significance] = None) -> str:
    """RQ1: the triplets -- per backbone the base model, its decision fine-tunes (both modes) and its reranker
    fine-tune; nDCG@10 per dataset, the mean over all ten and over the zero-shot subset. Best per backbone in bold;
    significant differences (`cdm significance`) as superscripts."""
    significance = significance or {}
    datasets = [d for d, _ in DATASETS]
    columns = datasets + ["avg", "zs"]

    def value(system: str, column: str) -> Optional[float]:
        if column == "avg":
            return _mean(cells, system, datasets)
        if column == "zs":
            return _mean(cells, system, ZERO_SHOT)
        return cells[(system, column)][0] if (system, column) in cells else None

    ncols = 3 + len(columns)
    lines = [
        "% Generated by `python -m cdm.paper_tables` (make tables, in paper/) from results/results.csv -- do not edit",
        "% by hand.",
        "\\begin{table*}[t]",
        "\\caption{Base model, decision fine-tune and reranker fine-tune on the same backbone (nDCG@10, $\\times 100$,",
        "BM25 top-100). Base: the post-trained release asked S1Rank's question zero-shot. Decision fine-tunes in joint",
        "and isolated mode. Avg: mean over all ten collections; ZS: mean over the collections without a training split",
        "(TREC-COVID, Touch\\'e, DBPedia). Best per backbone and column in bold. Paired $t$-test, Holm-corrected over",
        "the ten collections, $p < 0.05$: $^\\ast$ reranker fine-tune vs.\\ the decision fine-tune with the best Avg,",
        "$^\\dagger$ that decision fine-tune vs.\\ the base model, $^\\ddagger$ joint vs.\\ isolated. No reranker",
        "fine-tune of the 27B backbone exists.}",
        "\\label{tab:triplets}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{adjustbox}{max width=\\textwidth}",
        "\\begin{tabular}{lll" + "r" * len(columns) + "}",
    ]
    _header(lines, "System & Training & Mode", ["Avg", "ZS"])
    for backbone, rows in TRIPLETS:
        best = {
            c: max((_fmt(v) for v in (value(s, c) for s, _, _, _ in rows) if v is not None), key=float, default=None)
            for c in columns
        }
        lines += ["\\midrule", f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{backbone}}}}} \\\\"]
        for system, name, objective, mode in rows:
            out = [name, objective, mode]
            for c in columns:
                v = value(system, c)
                text = "--" if v is None else _fmt(v)
                text = f"\\textbf{{{text}}}" if v is not None and text == best[c] else text
                mark = significance.get((system, c), "")
                out.append(text + (f"\\rlap{{$^{{{mark}}}$}}" if mark else ""))
            lines.append(" & ".join(out) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table*}", ""]
    return "\n".join(lines)


def context_table(cells: Cells, cost: Costs) -> str:
    """RQ1 context: nDCG@10 of every other system on every dataset, the mean over all ten, and the API cost."""
    datasets = [d for d, _ in DATASETS]
    systems = [s for _, rows in CONTEXT_ROWS for s, _, _ in rows]

    def mean(system: str) -> Optional[float]:
        return _mean(cells, system, datasets)

    # Best reranker per column, compared as printed (ties are all bold).
    best = {
        d: max((_fmt(cells[(s, d)][0]) for s in systems if (s, d) in cells), key=float, default=None)
        for d in datasets
    }
    means = {s: mean(s) for s in systems + ["bm25"]}
    best["avg"] = max((_fmt(m) for s, m in means.items() if m is not None and s != "bm25"), key=float, default=None)

    def row(system: str, name: str, mode: str) -> str:
        out = [name, mode]
        for d in datasets:
            if (system, d) not in cells:
                out.append("--")
                continue
            value, literature = cells[(system, d)]
            text = _fmt(value)
            if system != "bm25" and text == best[d]:
                text = f"\\textbf{{{text}}}"
            out.append(text + ("\\rlap{$^\\dagger$}" if literature else ""))
        m = means[system]
        out.append("--" if m is None else (f"\\textbf{{{_fmt(m)}}}" if _fmt(m) == best["avg"] else _fmt(m)))
        costs = [cost[(system, d)] for d in datasets if (system, d) in cost]
        out.append(f"{sum(costs) / len(costs):.2f}" if costs else "--")
        return " & ".join(out) + " \\\\"

    ncols = 2 + len(datasets) + 2
    lines = [
        "% Generated by `python -m cdm.paper_tables` (make tables, in paper/) from results/results.csv and",
        "% results/literature.csv -- do not edit by hand.",
        "\\begin{table*}[t]",
        "\\caption{Context: other rerankers (nDCG@10, $\\times 100$) on the BM25 top-100 of TREC DL 2019--2023 and",
        "five BEIR collections; System One models in joint mode (\\laya{} isolated). $^\\dagger$: published number on",
        "the same candidates~\\cite{Abdallah2025HowGood,SetEncoderCard}, not recomputed. Avg: mean over all ten",
        "collections (only where all are available). \\$/1k: mean API cost in US dollars per 1,000 queries over the",
        "collections run. Best reranker per column in bold.}",
        "\\label{tab:context}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        # Natural size where it fits (ACM sigconf), shrunk to the text width where not (LNCS).
        "\\begin{adjustbox}{max width=\\textwidth}",
        "\\begin{tabular}{ll" + "r" * len(DL_DATASETS) + "r" * (len(datasets) - len(DL_DATASETS)) + "rr}",
        "\\toprule",
        f" & & \\multicolumn{{{len(DL_DATASETS)}}}{{c}}{{TREC DL}} & "
        f"\\multicolumn{{{len(datasets) - len(DL_DATASETS)}}}{{c}}{{BEIR}} & & \\\\",
        f"\\cmidrule(lr){{3-{2 + len(DL_DATASETS)}}}\\cmidrule(lr){{{3 + len(DL_DATASETS)}-{2 + len(datasets)}}}",
        "System & Mode & " + " & ".join(h for _, h in DATASETS) + " & Avg & \\$/1k \\\\",
        "\\midrule",
        row("bm25", "BM25", ""),
    ]
    for heading, rows in CONTEXT_ROWS:
        lines += ["\\midrule", f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{heading}}}}} \\\\"]
        lines += [row(*r) for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table*}", ""]
    return "\n".join(lines)


# RQ2's cutoff table: the triplets' systems plus these, on the datasets with judged non-relevant passages.
CUTOFF_OTHER: List[Tuple[str, str, str, str]] = [
    ("jev-joint", "\\jev{}", "decision", "joint"),
    ("jev-isolated", "\\jev{}", "decision", "isolated"),
    ("laya", "\\laya{}", "decision", "isolated"),
    ("monot5-3b", "monoT5-3B", "reranker", ""),
    ("bge-m3", "bge-reranker-v2-m3", "reranker", ""),
    ("set-encoder-large", "Set-Encoder-large", "reranker", ""),
]
# Where a default cell comes from the literature, `cdm cutoff` reads our efficiency run on the same candidates.
CUTOFF_VARIANTS = ("", "a100")

CutoffRows = Dict[Tuple[str, str], Dict[str, str]]  # (system, dataset) -> stored cutoff row


def load_cutoff(results_folder: str = RESULTS_FOLDER) -> CutoffRows:
    """(system, dataset) -> the `cdm cutoff` row on the default candidates (default variant before a100)."""
    rows: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    with open(os.path.join(results_folder, "results.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row["job"] != "cutoff" or row["subset"] or row["variant"] not in CUTOFF_VARIANTS:
                continue
            if row["dataset"] in DL_DATASETS and row["first_stage"] == FIRST_STAGE[row["dataset"]]:
                rows[(row["system"], row["dataset"], row["variant"])] = row
    out: CutoffRows = {}
    for system, dataset, _ in rows:
        for variant in CUTOFF_VARIANTS:
            if (system, dataset, variant) in rows:
                out[(system, dataset)] = rows[(system, dataset, variant)]
                break
    return out


def cutoff_summary(rows: CutoffRows, system: str) -> Optional[Dict[str, float]]:
    """Means over DL 2019-2023 of one system's cutoff values and the range of its oracle cutoff; None if incomplete."""
    if not all((system, d) in rows for d in DL_DATASETS):
        return None

    def mean(column: str) -> float:
        return sum(float(rows[(system, d)][column]) for d in DL_DATASETS) / len(DL_DATASETS)

    taus = [float(rows[(system, d)]["cut_oracle_tau"]) for d in DL_DATASETS]
    summary = {
        "F1@0.5": mean("cut_F1@0.5"),
        "F1@oracle": mean("cut_F1@oracle"),
        "regret": mean("cut_regret"),
        "tau_min": min(taus),
        "tau_max": max(taus),
        "size@0.5": mean("cut_size@0.5"),
        "topk_F1@10": mean("topk_F1@10"),
    }
    # Count test (Kanishka, 2026-10-12; src/cdm/cutoff.py count_test), where the rows have it.
    for key, column in (("count_MAE", "count_MAE"), ("count_ref", "count_median_ref")):
        if all(_float(rows[(system, d)].get(column, "")) is not None for d in DL_DATASETS):
            summary[key] = mean(column)
    return summary


def cutoff_table(rows: CutoffRows, ece: Optional[Dict[Tuple[str, str], float]] = None) -> str:
    """RQ2: calibration (mean ECE over DL 2019-2023) and one fixed cutoff (p >= 0.5, Part A) against the
    per-collection oracle cutoff and a fixed top-10."""
    ece = ece or {}
    groups = TRIPLETS + [("Other systems", CUTOFF_OTHER)]
    ncols = 3 + 6
    lines = [
        "% Generated by `python -m cdm.paper_tables` (make tables, in paper/) from the `cdm cutoff` rows of",
        "% results/results.csv -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\caption{Calibration and one cutoff for every collection, means over TREC DL 2019--2023. ECE: expected",
        "calibration error on the judged pairs (15 bins; lower is better). $F_1$@.5: set $F_1$ ($\\times 100$) when",
        "every candidate with $p \\ge 0.5$ is kept; Regret: $F_1$ at the best cutoff of each collection, chosen",
        "afterwards, minus $F_1$@.5; $\\tau^*$: range of these oracle cutoffs; $|S|$: mean number of passages kept at",
        "0.5; Top-10: $F_1$ of the ten highest-ranked passages.}",
        "\\label{tab:cutoff}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{adjustbox}{max width=\\linewidth}",
        "\\begin{tabular}{lllrrrcrr}",
        "\\toprule",
        "System & Training & Mode & ECE & $F_1$@.5 & Regret & $\\tau^*$ & $|S|$ & Top-10 \\\\",
    ]
    for heading, members in groups:
        lines += ["\\midrule", f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{heading}}}}} \\\\"]
        for system, name, objective, mode in members:
            s = cutoff_summary(rows, system)
            mean_ece = _mean_of(ece, system, DL_DATASETS)
            values = ["--" if mean_ece is None else f"{mean_ece:.3f}"]
            if s is None:
                values += ["--"] * 5
            else:
                values += [
                    _fmt(s["F1@0.5"]),
                    _fmt(s["regret"]),
                    f"{s['tau_min']:.2f}--{s['tau_max']:.2f}",
                    f"{s['size@0.5']:.1f}",
                    _fmt(s["topk_F1@10"]),
                ]
            lines.append(" & ".join([name, objective, mode] + values) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}", ""]
    return "\n".join(lines)


# RQ2, Part C: the rerankers scaled after the fact, and the decision models they are compared with (native only).
POSTHOC_RERANKERS: List[Tuple[str, str]] = [
    ("qwen3-reranker-0.6b", "Qwen3-Reranker-0.6B"),
    ("qwen3-reranker-8b", "Qwen3-Reranker-8B"),
    ("prism-0.8b", "Prism-0.8B"),
    ("prism-4b", "Prism-4B"),
    ("prism-9b", "Prism-9B"),
    ("monot5-3b", "monoT5-3B"),
    ("bge-m3", "bge-reranker-v2-m3"),
    ("set-encoder-large", "Set-Encoder-large"),
]
POSTHOC_DECISION: List[Tuple[str, str]] = [
    ("qwen3.5-9b-reranker", "Ours-9B (reranker)"),
    ("kev-27b-joint", "Kev-27B joint"),
    ("clef-joint", "Clef joint"),
    ("clef-flash-joint", "Clef-Flash joint"),
    ("jev-joint", "\\jev{} joint"),
]
BEIR_DATASETS = [d for d, _ in DATASETS if d not in DL_DATASETS]

PosthocRows = Dict[Tuple[str, str, str], Dict[str, str]]  # (system, dataset, method) -> stored row


def load_posthoc(results_folder: str = RESULTS_FOLDER) -> PosthocRows:
    """(system, dataset, method) -> the `cdm posthoc` row on the default candidates of a test collection."""
    out: PosthocRows = {}
    with open(os.path.join(results_folder, "results.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row["job"] == "posthoc" and not row["subset"] and row["dataset"] in FIRST_STAGE:
                out[(row["system"], row["dataset"], row["variant"])] = row
    return out


def load_rerank_column(column: str, results_folder: str = RESULTS_FOLDER) -> Dict[Tuple[str, str], float]:
    """(system, dataset) -> a metric of the default rerank row (variant before a100, as for the cutoff rows)."""
    found: Dict[Tuple[str, str, str], float] = {}
    with open(os.path.join(results_folder, "results.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row["job"] != "rerank" or row["subset"] or row["variant"] not in CUTOFF_VARIANTS:
                continue
            value = _float(row.get(column, ""))
            default = row["dataset"] in FIRST_STAGE and row["first_stage"] == FIRST_STAGE[row["dataset"]]
            if default and value is not None:
                found[(row["system"], row["dataset"], row["variant"])] = value
    out: Dict[Tuple[str, str], float] = {}
    for system, dataset, _ in found:
        for variant in CUTOFF_VARIANTS:
            if (system, dataset, variant) in found:
                out[(system, dataset)] = found[(system, dataset, variant)]
                break
    return out


def _mean_of(values: Dict[Tuple[str, str], float], system: str, datasets: List[str]) -> Optional[float]:
    found = [values[(system, d)] for d in datasets if (system, d) in values]
    return sum(found) / len(found) if found and len(found) == len(datasets) else None


def posthoc_table(rows: PosthocRows, cutoff: CutoffRows, ece: Dict[Tuple[str, str], float],
                  ece_unjudged0: Dict[Tuple[str, str], float]) -> str:
    """RQ2, Part C: native vs. temperature- vs. Platt-scaled probabilities (fitted on msmarco-dev)."""

    def column(system: str, method: str, name: str, datasets: List[str]) -> Dict[Tuple[str, str], float]:
        return {(system, d): float(rows[(system, d, method)][name]) for d in datasets
                if (system, d, method) in rows and _float(rows[(system, d, method)].get(name, "")) is not None}

    def cell(value: Optional[float], digits: int = 3, scale: float = 1.0) -> str:
        return "--" if value is None else f"{value * scale:.{digits}f}"

    lines = [
        "% Generated by `python -m cdm.paper_tables` (make tables, in paper/) from the `cdm posthoc`, `cdm cutoff` and",
        "% rerank rows of results/results.csv -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\caption{Calibrated after the fact: temperature (T) and Platt (P) scaling of each reranker's logit, fitted",
        "on MS~MARCO dev (1,000 queries, unjudged passages as non-relevant), against the native (N) probabilities, and",
        "our reranker fine-tune and decision models as they come. ECE: judged pairs of TREC DL 2019--2023; $F_1$@.5 and regret as in",
        "Table~\\ref{tab:cutoff} ($\\times 100$); BEIR: ECE with unjudged passages as non-relevant, mean over the five",
        "collections. Temperature scaling keeps the set at $p \\ge 0.5$ unchanged.}",
        "\\label{tab:posthoc}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{adjustbox}{max width=\\linewidth}",
        "\\begin{tabular}{lrrrrrrrrr}",
        "\\toprule",
        " & \\multicolumn{3}{c}{ECE (DL)} & \\multicolumn{2}{c}{$F_1$@.5} & \\multicolumn{2}{c}{Regret}"
        " & \\multicolumn{2}{c}{ECE (BEIR)} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}",
        "System & N & T & P & N & P & N & P & N & P \\\\",
        "\\midrule",
        "\\multicolumn{10}{l}{\\textit{Reranker fine-tunes and cross-encoders}} \\\\",
    ]
    for system, name in POSTHOC_RERANKERS:
        values = [
            cell(_mean_of(column(system, m, "ECE", DL_DATASETS), system, DL_DATASETS))
            for m in ("native", "temperature", "platt")
        ]
        values += [
            cell(_mean_of(column(system, m, metric, DL_DATASETS), system, DL_DATASETS), 1, 100)
            for metric in ("cut_F1@0.5", "cut_regret")
            for m in ("native", "platt")
        ]
        values += [
            cell(_mean_of(column(system, m, "ECE_unjudged0", BEIR_DATASETS), system, BEIR_DATASETS))
            for m in ("native", "platt")
        ]
        lines.append(" & ".join([name] + values) + " \\\\")
    lines += ["\\midrule", "\\multicolumn{10}{l}{\\textit{As they come: our reranker fine-tune and decision models}} \\\\"]
    for system, name in POSTHOC_DECISION:
        summary = cutoff_summary(cutoff, system)
        values = [cell(_mean_of(ece, system, DL_DATASETS)), "--", "--"]
        values += [cell(None if summary is None else summary["F1@0.5"], 1, 100), "--"]
        values += [cell(None if summary is None else summary["regret"], 1, 100), "--"]
        values += [cell(_mean_of(ece_unjudged0, system, BEIR_DATASETS)), "--"]
        lines.append(" & ".join([name] + values) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}", ""]
    return "\n".join(lines)


# The ECIR short paper's RQ2 table (docs/intent/paper-focus.md 16): per backbone the base model, each decision
# fine-tune in its better-calibrated mode and the reranker fine-tunes, then the other systems; calibration (ECE, also
# after temperature and Platt scaling for the rerankers) and the fixed cutoff in one table.
CALIBRATION_GROUPS: List[Tuple[str, List[Tuple[List[str], str, str]]]] = [
    (
        backbone,
        [
            ([s for s, n, o, _ in rows if n == name and o == objective], name, objective)
            for name, objective in dict.fromkeys((n, o) for _, n, o, _ in rows)
        ],
    )
    for backbone, rows in TRIPLETS
] + [
    (
        "Other systems",
        [
            (["jev-joint", "jev-isolated"], "\\jev{}", "decision"),
            (["monot5-3b"], "monoT5-3B", "reranker"),
            (["bge-m3"], "bge-reranker-v2-m3", "reranker"),
            (["set-encoder-large"], "Set-Encoder-large", "reranker"),
        ],
    )
]


# Groups in the left panel of the calibration table (Qwen3-0.6B to Qwen3.5-4B); the rest goes right.
CALIBRATION_LEFT_PANEL = 4


def calibration_table(cutoff: CutoffRows, posthoc: PosthocRows, ece: Dict[Tuple[str, str], float]) -> str:
    """RQ2 in one table: mean ECE over DL 2019-2023 (native, temperature, Platt), set F1 at p >= 0.5 (native, Platt),
    the regret of 0.5 against the per-collection oracle cutoff and the mean set size at 0.5."""

    def posthoc_mean(system: str, method: str, column: str) -> Optional[float]:
        values = {
            (system, d): float(posthoc[(system, d, method)][column])
            for d in DL_DATASETS
            if (system, d, method) in posthoc and _float(posthoc[(system, d, method)].get(column, "")) is not None
        }
        return _mean_of(values, system, DL_DATASETS)

    def cell(value: Optional[float], fmt: str, scale: float = 1.0) -> str:
        return "--" if value is None else fmt.format(scale * value)

    # The count test's reference (always guessing the median count) depends only on the labels: one value for all.
    summaries = (cutoff_summary(cutoff, system) for system in sorted({key[0] for key in cutoff}))
    references = [s["count_ref"] for s in summaries if s and "count_ref" in s]
    reference = f" (always guessing the median count: {references[0]:.1f})" if references else ""
    lines = [
        "% Generated by `python -m cdm.paper_tables` (make tables, in paper/) from the rerank, `cdm cutoff` and",
        "% `cdm posthoc` rows of results/results.csv -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\caption{Calibration and one cutoff, means over TREC DL 2019--2023. Per backbone: base model, decision",
        "fine-tunes in their better-calibrated mode (j/i: joint/isolated), reranker fine-tunes. N: native; T, P:",
        "temperature/Platt scaling fitted on MS~MARCO dev. $F_1$@.5 ($\\times 100$): set kept at $p \\ge 0.5$;",
        "Regret: $F_1$ at each collection's best cutoff minus $F_1$@.5; $|S|$: passages kept at 0.5 (of 100). Count:",
        f"mean absolute error of the summed probabilities against the number of relevant passages{reference}.}}",
        "\\label{tab:calibration}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{adjustbox}{max width=\\textwidth}",
    ]
    header = [
        "\\begin{tabular}[t]{lrrrrrrrr}",
        "\\toprule",
        " & \\multicolumn{3}{c}{ECE} & \\multicolumn{2}{c}{$F_1$@.5} & & & \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}",
        "System & N & T & P & N & P & Regret & $|S|$ & Count \\\\",
    ]
    # Two panels side by side (the small backbones, then the rest), to fit the short paper's page limit.
    panels: List[List[str]] = [list(header), list(header)]
    for index, (heading, members) in enumerate(CALIBRATION_GROUPS):
        panel = panels[0 if index < CALIBRATION_LEFT_PANEL else 1]
        panel += ["\\midrule", f"\\multicolumn{{9}}{{l}}{{\\textit{{{heading}}}}} \\\\"]
        for systems, name, _ in members:
            scored = [(_mean_of(ece, s, DL_DATASETS), s) for s in systems]
            scored = [(e, s) for e, s in scored if e is not None]
            if not scored:
                panel.append(" & ".join([name] + ["--"] * 8) + " \\\\")
                continue
            mean_ece, system = min(scored)
            if len(systems) > 1:
                name += " (j)" if system.endswith("-joint") else " (i)"
            summary = cutoff_summary(cutoff, system)
            values = [
                f"{mean_ece:.3f}",
                cell(posthoc_mean(system, "temperature", "ECE"), "{:.3f}"),
                cell(posthoc_mean(system, "platt", "ECE"), "{:.3f}"),
                cell(None if summary is None else summary["F1@0.5"], "{:.1f}", 100),
                cell(posthoc_mean(system, "platt", "cut_F1@0.5"), "{:.1f}", 100),
                cell(None if summary is None else summary["regret"], "{:.1f}", 100),
                cell(None if summary is None else summary["size@0.5"], "{:.1f}"),
                cell(None if summary is None else summary.get("count_MAE"), "{:.1f}"),
            ]
            panel.append(" & ".join([name] + values) + " \\\\")
    for panel in panels:
        panel += ["\\bottomrule", "\\end{tabular}"]
    lines += panels[0] + ["\\quad"] + panels[1]
    lines += ["\\end{adjustbox}", "\\end{table}", ""]
    return "\n".join(lines)


# The ECIR short paper's context table (docs/intent/paper-focus.md 16): systems outside the triplets, means only.
CONTEXT_SHORT: List[Row] = [
    ("bm25", "BM25", ""),
    ("rankgpt-4", "RankGPT-4", "listwise"),
    ("rankzephyr-7b", "RankZephyr-7B", "listwise"),
    ("monot5-3b", "monoT5-3B", "cross-encoder"),
    ("bge-m3", "bge-reranker-v2-m3", "cross-encoder"),
    ("set-encoder-large", "Set-Encoder-large", "cross-encoder"),
    ("jev-joint", "\\jev{}", "System One, joint"),
    ("jev-isolated", "\\jev{}", "System One, isolated"),
    ("laya", "\\laya{}", "System One, isolated"),
]


def context_short_table(cells: Cells) -> str:
    """RQ1 context in little space: means over TREC DL, BEIR, all ten and the zero-shot subset per system."""
    beir = [d for d, _ in DATASETS if d not in DL_DATASETS]
    groups = [("DL", DL_DATASETS), ("BEIR", beir), ("Avg", [d for d, _ in DATASETS]), ("ZS", ZERO_SHOT)]
    lines = [
        "% Generated by `python -m cdm.paper_tables` (make tables, in paper/) from results/results.csv and",
        "% results/literature.csv -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\caption{Other systems, same candidates (nDCG@10 $\\times 100$, means). $^\\dagger$: includes published",
        "numbers~\\cite{Abdallah2025HowGood,SetEncoderCard}; --: not all collections available.}",
        "\\label{tab:context-short}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "System & Type & " + " & ".join(g for g, _ in groups) + " \\\\",
        "\\midrule",
    ]
    for system, name, mode in CONTEXT_SHORT:
        out = [name, mode]
        for _, datasets in groups:
            mean = _mean(cells, system, datasets)
            literature = any(cells.get((system, d), (0.0, False))[1] for d in datasets)
            out.append("--" if mean is None else _fmt(mean) + ("\\rlap{$^\\dagger$}" if literature else ""))
        lines.append(" & ".join(out) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--results", default=RESULTS_FOLDER, help="Folder with results.csv and literature.csv.")
    parser.add_argument("--out", default=TABLES_FOLDER, help="Folder for the .tex tables.")
    args = parser.parse_args(argv)
    cells, cost = load_cells(args.results)
    os.makedirs(args.out, exist_ok=True)
    tables = {
        "triplets": triplet_table(cells, load_significance(args.results)),
        "context": context_table(cells, cost),
        "cutoff": cutoff_table(load_cutoff(args.results), load_rerank_column("ECE", args.results)),
        "posthoc": posthoc_table(
            load_posthoc(args.results),
            load_cutoff(args.results),
            load_rerank_column("ECE", args.results),
            load_rerank_column("ECE_unjudged0", args.results),
        ),
        "calibration": calibration_table(
            load_cutoff(args.results), load_posthoc(args.results), load_rerank_column("ECE", args.results)
        ),
        "context-short": context_short_table(cells),
    }
    for name, tex in tables.items():
        path = os.path.join(args.out, f"{name}.tex")
        with open(path, "w") as f:
            f.write(tex)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

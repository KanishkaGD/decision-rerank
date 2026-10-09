"""
Generate the paper's result tables (paper/tables/*.tex) from results/results.csv and results/literature.csv, so that
no number in the paper is typed by hand (CLAUDE.md, "Paper"). Standard library only, so it runs on the login node:

    python -m cdm.paper_tables            (or: make tables, in paper/)

Three tables: the triplets (RQ1's main table: per backbone the base model, its decision fine-tunes and its reranker
fine-tune; triplets.tex), the context table (every other system on the same candidates; context.tex) and the fixed
cutoff (RQ2, Part A: `cdm cutoff` rows of results.csv; cutoff.tex).
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


def triplet_table(cells: Cells) -> str:
    """RQ1: the triplets -- per backbone the base model, its decision fine-tunes (both modes) and its reranker
    fine-tune; nDCG@10 per dataset, the mean over all ten and over the zero-shot subset. Best per backbone in bold."""
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
        "(TREC-COVID, Touch\\'e, DBPedia). Best per backbone and column in bold. No reranker fine-tune of the 27B",
        "backbone exists.}",
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
                out.append(f"\\textbf{{{text}}}" if v is not None and text == best[c] else text)
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
        "\\caption{Context: other rerankers (nDCG@10, $\\times 100$) on the BM25 top-100 of TREC DL 2019--2023 and five",
        "BEIR collections; System One models in joint mode (\\laya{} isolated). $^\\dagger$: published number on the",
        "same candidates~\\cite{Abdallah2025HowGood,SetEncoderCard}, not recomputed. Avg: mean over all ten",
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
    return {
        "F1@0.5": mean("cut_F1@0.5"),
        "F1@oracle": mean("cut_F1@oracle"),
        "regret": mean("cut_regret"),
        "tau_min": min(taus),
        "tau_max": max(taus),
        "size@0.5": mean("cut_size@0.5"),
        "topk_F1@10": mean("topk_F1@10"),
    }


def cutoff_table(rows: CutoffRows) -> str:
    """RQ2, Part A: one fixed cutoff (p >= 0.5) against the per-collection oracle cutoff and a fixed top-10."""
    groups = TRIPLETS + [("Other systems", CUTOFF_OTHER)]
    ncols = 3 + 6
    lines = [
        "% Generated by `python -m cdm.paper_tables` (make tables, in paper/) from the `cdm cutoff` rows of",
        "% results/results.csv -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\caption{One cutoff for every collection: keep each candidate with $p \\ge 0.5$ (means over TREC DL",
        "2019--2023, $\\times 100$). $F_1$@.5: set $F_1$ of the kept passages; Oracle: $F_1$ at the best cutoff of each",
        "collection, chosen afterwards; Regret: Oracle $-$ $F_1$@.5; $\\tau^*$: range of the oracle cutoffs across the",
        "collections; $|S|$: mean number of passages kept at 0.5; Top-10: $F_1$ of the ten highest-ranked passages.}",
        "\\label{tab:cutoff}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{adjustbox}{max width=\\linewidth}",
        "\\begin{tabular}{lllrrrcrr}",
        "\\toprule",
        "System & Training & Mode & $F_1$@.5 & Oracle & Regret & $\\tau^*$ & $|S|$ & Top-10 \\\\",
    ]
    for heading, members in groups:
        lines += ["\\midrule", f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{heading}}}}} \\\\"]
        for system, name, objective, mode in members:
            s = cutoff_summary(rows, system)
            if s is None:
                values = ["--"] * 6
            else:
                values = [
                    _fmt(s["F1@0.5"]),
                    _fmt(s["F1@oracle"]),
                    _fmt(s["regret"]),
                    f"{s['tau_min']:.2f}--{s['tau_max']:.2f}",
                    f"{s['size@0.5']:.1f}",
                    _fmt(s["topk_F1@10"]),
                ]
            lines.append(" & ".join([name, objective, mode] + values) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}", ""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--results", default=RESULTS_FOLDER, help="Folder with results.csv and literature.csv.")
    parser.add_argument("--out", default=TABLES_FOLDER, help="Folder for the .tex tables.")
    args = parser.parse_args(argv)
    cells, cost = load_cells(args.results)
    os.makedirs(args.out, exist_ok=True)
    tables = {
        "triplets": triplet_table(cells),
        "context": context_table(cells, cost),
        "cutoff": cutoff_table(load_cutoff(args.results)),
    }
    for name, tex in tables.items():
        path = os.path.join(args.out, f"{name}.tex")
        with open(path, "w") as f:
            f.write(tex)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

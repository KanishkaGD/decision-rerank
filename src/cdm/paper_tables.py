"""
Generate the paper's result tables (paper/tables/*.tex) from results/results.csv and results/literature.csv, so that
no number in the paper is typed by hand (CLAUDE.md, "Paper"). Standard library only, so it runs on the login node:

    python -m cdm.paper_tables            (or: make tables, in paper/)

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

# Row groups of the main table: (heading, [(system, display name, mode)]). The plan of record is docs/roadmap.md,
# "Systems"; systems not listed here (side experiments) are left out.
MAIN_ROWS: List[Tuple[str, List[Tuple[str, str, str]]]] = [
    ("Listwise LLM rerankers", [("rankgpt-4", "RankGPT-4", ""), ("rankzephyr-7b", "RankZephyr-7B", "")]),
    (
        "Cross-encoders and LLM rerankers",
        [
            ("bge-m3", "bge-reranker-v2-m3", ""),
            ("monot5-3b", "monoT5-3B", ""),
            ("qwen3-reranker-8b", "Qwen3-Reranker-8B", ""),
            ("set-encoder-large", "Set-Encoder-large", ""),
            ("qwen3.7-flash", "Qwen3.7-Flash", ""),
        ],
    ),
    (
        "System One models",
        [
            (f"{model}-{mode}", name, mode)
            for model, name in [
                ("jev", "\\jev{}"),
                ("clef-flash", "Clef-Flash"),
                ("clef", "Clef"),
                ("kev-4b", "Kev-4B"),
                ("kev-27b", "Kev-27B"),
            ]
            for mode in ("joint", "isolated")
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


def main_table(cells: Cells, cost: Costs) -> str:
    """RQ1: nDCG@10 of every system on every dataset, the mean over all ten, and the API cost."""
    datasets = [d for d, _ in DATASETS]
    systems = [s for _, rows in MAIN_ROWS for s, _, _ in rows]

    def mean(system: str) -> Optional[float]:
        values = [cells[(system, d)][0] for d in datasets if (system, d) in cells]
        return sum(values) / len(values) if len(values) == len(datasets) else None

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
        "\\caption{Reranking effectiveness (nDCG@10, $\\times 100$) on the BM25 top-100 of TREC DL 2019--2023 and five",
        "BEIR collections. $^\\dagger$: published number on the same candidates~\\cite{Abdallah2025HowGood}, not",
        "recomputed. Avg: mean over all ten collections (only where all are available). \\$/1k: mean API cost in US",
        "dollars per 1,000 queries over the collections run. Best reranker per column in bold.}",
        "\\label{tab:main}",
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
    for heading, rows in MAIN_ROWS:
        lines += ["\\midrule", f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{heading}}}}} \\\\"]
        lines += [row(*r) for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table*}", ""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--results", default=RESULTS_FOLDER, help="Folder with results.csv and literature.csv.")
    parser.add_argument("--out", default=TABLES_FOLDER, help="Folder for the .tex tables.")
    args = parser.parse_args(argv)
    cells, cost = load_cells(args.results)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "main.tex")
    with open(path, "w") as f:
        f.write(main_table(cells, cost))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

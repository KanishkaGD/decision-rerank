"""
Generate the paper's result tables (tables/*.tex) from results/results.csv and results/literature.csv, so that no
number in the paper is typed by hand. Standard library only:

    python -m cdm.paper_tables

The paper's question is `answer` ("Does document ... answer the query?", QUESTION) for every decision and base model;
S1Rank's question appears only in the question ablation (question-ablation.tex). Base models asked the `answer`
question are the systems `<name>-prompted-answer`; the loaders file them under `<name>-prompted`.

Tables of the arXiv version: the triplets (RQ1's main table: per backbone the base model, its decision fine-tunes and
its reranker fine-tunes; triplets.tex), every other system on the same candidates (context.tex), RQ2 in one table
(calibration, post-hoc scaling, the fixed cutoff and the count test; calibration.tex) and S1Rank's question against
the paper's (question-ablation.tex). Every reranker is read on its dataset's default candidates (BM25 top-100); cells
taken from the literature (role `baseline` in results/literature.csv) win over computed ones and are marked with a
dagger.
"""

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

from cdm.config import RESULTS_FOLDER
from cdm.datasets import get_dataset

# The header comment of each table names `make tables` in the paper's folder, as in our working repository, so that
# the tables here are byte-identical to the paper's.
TABLES_FOLDER = "tables"

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

# Collections without a training split in any of the fine-tunes' data: the zero-shot subset.
ZERO_SHOT = ["trec-covid", "touche", "dbpedia"]

Row = Tuple[str, str, str]  # system, display name, mode

# The triplets: (backbone, [(system, display name, objective, mode)]).
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
            # Our reranker fine-tune of the same checkpoint.
            ("qwen3.5-9b-reranker", "\\textit{Ours-9B}", "reranker", ""),
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
            # Our reranker fine-tune (checkpoint final, chosen on msmarco-dev).
            ("qwen3.8-27b-reranker", "\\textit{Ours-27B}", "reranker", ""),
        ],
    ),
]

# Row groups of the context table: (heading, [(system, display name, mode)]).
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
            # Joint mode only; both modes of Kev and Clef are in the triplet table.
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
# The question asked of every decision and base model in the paper, and the one it is
# compared with in the ablation (S1Rank's). Systems that ask no question have phrasing "none" and appear under both.
QUESTION = "answer"
ABLATION_QUESTION = "s1rank"
NO_QUESTION = ("none", "")
BASE_SYSTEMS = {s for _, rows in TRIPLETS for s, _, objective, _ in rows if objective == "base"}


def question_system(system: str, objective: str, question: str) -> str:
    """The system id of a table row for a question: base models asked another question than S1Rank's are separate
    systems, `<name>-prompted-<question>`."""
    return f"{system}-{question}" if objective == "base" and question != ABLATION_QUESTION else system


def _row_system(row: Dict[str, str], question: str) -> Optional[str]:
    """The table system a results.csv row stands for under `question`, or None if it belongs to another question."""
    phrasing, system = row.get("phrasing", ""), row["system"]
    if phrasing not in (question,) + NO_QUESTION:
        return None
    if system in BASE_SYSTEMS:
        return system if question == ABLATION_QUESTION else None
    head, _, tail = system.rpartition("-")
    if head in BASE_SYSTEMS:
        return head if tail == question else None
    return system

COST = "api_cost_usd_per_1k_queries"

Cell = Tuple[float, bool]  # value, taken from the literature
Cells = Dict[Tuple[str, str], Cell]  # (system, dataset) -> cell
Costs = Dict[Tuple[str, str], float]


def _float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_cells(results_folder: str = RESULTS_FOLDER, question: str = QUESTION) -> Tuple[Cells, Costs]:
    """(system, dataset) -> (nDCG@10, from literature) on the default candidates, and -> API cost per 1k queries."""
    cells: Cells = {}
    cost: Costs = {}
    with open(os.path.join(results_folder, "results.csv"), newline="") as f:
        for row in csv.DictReader(f):
            system = _row_system(row, question)
            if system is None or row["subset"] or row["variant"] or row["dataset"] not in FIRST_STAGE:
                continue
            first_stage = FIRST_STAGE[row["dataset"]]
            key = (system, row["dataset"])
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
    # Published numbers on the same candidates replace computed ones.
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


def _header(lines: List[str], first: str, extra: List[str], shown: Optional[List[Tuple[str, str]]] = None) -> None:
    """Column groups (TREC DL, BEIR) and headers after `first` leading columns, for the `shown` datasets."""
    shown = shown or DATASETS
    n_dl, n = len([d for d, _ in shown if d in DL_DATASETS]), len(shown)
    lines += [
        "\\toprule",
        " &" * (first.count("&") + 1) + f" \\multicolumn{{{n_dl}}}{{c}}{{TREC DL}} & "
        f"\\multicolumn{{{n - n_dl}}}{{c}}{{BEIR}} & \\multicolumn{{{len(extra)}}}{{c}}{{Mean}} \\\\",
        f"\\cmidrule(lr){{{first.count('&') + 2}-{first.count('&') + 1 + n_dl}}}"
        f"\\cmidrule(lr){{{first.count('&') + 2 + n_dl}-{first.count('&') + 1 + n}}}"
        f"\\cmidrule(lr){{{first.count('&') + 2 + n}-{first.count('&') + 1 + n + len(extra)}}}",
        first + " & " + " & ".join(h for _, h in shown) + " & " + " & ".join(extra) + " \\\\",
    ]


# Datasets left out of the triplet table (they still count in Avg). The ECIR version hid DL20 and DL22 for space; the
# arXiv version shows all ten.
TRIPLET_HIDDEN: Tuple[str, ...] = ()


def _rank_marks(values: List[Optional[float]]) -> Tuple[Optional[str], Optional[str]]:
    """The best and second-best value as printed (ties share a rank); None where there is none."""
    printed = sorted({_fmt(v) for v in values if v is not None}, key=float, reverse=True)
    return (printed[0] if printed else None), (printed[1] if len(printed) > 1 else None)


# Row shading by training objective; the colours (rowbase, rowdecision, rowreranker) are defined in the paper.
ROW_COLORS = {"base": "rowbase", "decision": "rowdecision", "reranker": "rowreranker"}


def _shade(objective: str) -> str:
    return f"\\rowcolor{{{ROW_COLORS[objective]}}}" if objective in ROW_COLORS else ""


# A symbol per training objective and mode, in the first column, so the tables also read without colour (colour-blind
# readers, black-and-white print). The legend sentence goes into the captions.
SYMBOLS = {
    ("base", ""): "$\\circ$",
    ("decision", "joint"): "$\\blacktriangle$",
    ("decision", "isolated"): "$\\triangle$",
    ("reranker", ""): "$\\blacksquare$",
}
SYMBOL_LEGEND = (
    "$\\circ$~untuned model, $\\blacktriangle$/$\\triangle$~decision fine-tune (joint/isolated),"
    " $\\blacksquare$~reranker fine-tune"
)


def _symbol(objective: str, mode: str = "") -> str:
    return SYMBOLS.get((objective, mode), "")


def _emphasize(text: str, best: Optional[str], second: Optional[str]) -> str:
    """Bold for the best value, underline for the second best."""
    if text == best:
        return f"\\textbf{{{text}}}"
    if text == second:
        return f"\\underline{{{text}}}"
    return text


# RQ1 significance marks in the triplet table (`cdm significance` rows): comparison -> superscript on the first system.
SIGNIFICANCE_MARKS = {"reranker-vs-decision": "\\ast", "decision-vs-base": "\\dagger", "joint-vs-isolated": "\\ddagger"}
Significance = Dict[Tuple[str, str], str]  # (system, dataset) -> marks of its significant comparisons


def load_significance(results_folder: str = RESULTS_FOLDER, question: str = QUESTION) -> Significance:
    """(system, dataset) -> the superscripts of the comparisons in which it differs significantly (Holm), from the
    `cdm significance --phrasing <question>` rows (variant `vs-<other>@<question>`; no suffix for S1Rank's)."""
    marks: Dict[Tuple[str, str], str] = {}
    with open(os.path.join(results_folder, "results.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row["job"] != "significance" or row.get("sig_holm") != "1":
                continue
            suffix = row["variant"].rpartition("@")[2] if "@" in row["variant"] else ABLATION_QUESTION
            system = _row_system(row, question)
            if suffix != question or system is None:
                continue
            mark = SIGNIFICANCE_MARKS.get(row.get("sig_comparison", ""))
            if mark:
                key = (system, row["dataset"])
                marks[key] = marks.get(key, "") + mark
    return marks


def triplet_table(cells: Cells, significance: Optional[Significance] = None) -> str:
    """RQ1: the triplets -- per backbone the base model, its decision fine-tunes (both modes) and its reranker
    fine-tune; nDCG@10 per dataset, the mean over all ten and over the zero-shot subset. Best per backbone in bold;
    significant differences (`cdm significance`) as superscripts."""
    significance = significance or {}
    datasets = [d for d, _ in DATASETS]
    shown = [(d, h) for d, h in DATASETS if d not in TRIPLET_HIDDEN]
    columns = [d for d, _ in shown] + ["avg", "zs"]

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
        "\\begin{table*}[!t]",
        "\\caption{Untuned model, decision fine-tunes and reranker fine-tunes of the same backbone (nDCG@10,",
        "$\\times 100$, BM25 top-100). Untuned and decision models are asked whether the passage answers the query.",
        "Avg: mean over all ten collections; ZS: mean over the collections without a training split in any",
        "fine-tune's data (TREC-COVID, Touch\\'e, DBPedia). Per backbone and column: best in bold, second best",
        f"underlined. Training: {SYMBOL_LEGEND} (rows shaded grey, red, blue);",
        "\\textit{Ours}: our reranker fine-tunes.",
        "Paired $t$-test, Holm-corrected over",
        "the ten collections, $p < 0.05$: $^\\ast$ reranker fine-tune vs.\\ the decision fine-tune with the best Avg,",
        "$^\\dagger$ that decision fine-tune vs.\\ the untuned model, $^\\ddagger$ joint vs.\\ isolated.}",
        "\\label{tab:triplets}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{adjustbox}{max width=\\textwidth}",
        "\\begin{tabular}{cll" + "r" * len(columns) + "}",
    ]
    _header(lines, " & System & Mode", ["Avg", "ZS"], shown)
    for backbone, rows in TRIPLETS:
        ranks = {c: _rank_marks([value(s, c) for s, _, _, _ in rows]) for c in columns}
        lines += ["\\midrule", f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{backbone}}}}} \\\\"]
        for system, name, objective, mode in rows:
            out = [_symbol(objective, mode), name, mode]
            for c in columns:
                v = value(system, c)
                text = "--" if v is None else _fmt(v)
                text = text if v is None else _emphasize(text, *ranks[c])
                mark = significance.get((system, c), "")
                out.append(text + (f"\\rlap{{$^{{{mark}}}$}}" if mark else ""))
            lines.append(_shade(objective) + " & ".join(out) + " \\\\")
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
        "\\begin{table*}[!t]",
        "\\caption{Context: other rerankers (nDCG@10, $\\times 100$) on the BM25 top-100 of TREC DL 2019--2023 and",
        "five BEIR collections; System One models in joint mode (\\laya{} isolated), asked whether the passage answers",
        "the query (Sect.~\\ref{sec:setup}). $^\\dagger$: published number on",
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


# Where a default cell comes from the literature, `cdm cutoff` reads our efficiency run on the same candidates.
CUTOFF_VARIANTS = ("", "a100")

CutoffRows = Dict[Tuple[str, str], Dict[str, str]]  # (system, dataset) -> stored cutoff row


def load_cutoff(results_folder: str = RESULTS_FOLDER, question: str = QUESTION) -> CutoffRows:
    """(system, dataset) -> the `cdm cutoff` row on the default candidates (default variant before a100)."""
    rows: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    with open(os.path.join(results_folder, "results.csv"), newline="") as f:
        for row in csv.DictReader(f):
            system = _row_system(row, question)
            if system is None or row["job"] != "cutoff" or row["subset"] or row["variant"] not in CUTOFF_VARIANTS:
                continue
            if row["dataset"] in DL_DATASETS and row["first_stage"] == FIRST_STAGE[row["dataset"]]:
                rows[(system, row["dataset"], row["variant"])] = row
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
    # Count test (src/cdm/cutoff.py count_test), where the rows have it.
    for key, column in (("count_MAE", "count_MAE"), ("count_ref", "count_median_ref")):
        if all(_float(rows[(system, d)].get(column, "")) is not None for d in DL_DATASETS):
            summary[key] = mean(column)
    return summary


PosthocRows = Dict[Tuple[str, str, str], Dict[str, str]]  # (system, dataset, method) -> stored row


def load_posthoc(results_folder: str = RESULTS_FOLDER) -> PosthocRows:
    """(system, dataset, method) -> the `cdm posthoc` row on the default candidates of a test collection."""
    out: PosthocRows = {}
    with open(os.path.join(results_folder, "results.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row.get("phrasing", "") not in NO_QUESTION:  # rerankers only: they ask no question
                continue
            if row["job"] == "posthoc" and not row["subset"] and row["dataset"] in FIRST_STAGE:
                out[(row["system"], row["dataset"], row["variant"])] = row
    return out


def load_rerank_column(
    column: str, results_folder: str = RESULTS_FOLDER, question: str = QUESTION, job: str = "rerank"
) -> Dict[Tuple[str, str], float]:
    """(system, dataset) -> a metric of the default `job` row (variant before a100, as for the cutoff rows) for the
    given question."""
    found: Dict[Tuple[str, str, str], float] = {}
    with open(os.path.join(results_folder, "results.csv"), newline="") as f:
        for row in csv.DictReader(f):
            system = _row_system(row, question)
            if system is None or row["job"] != job or row["subset"] or row["variant"] not in CUTOFF_VARIANTS:
                continue
            value = _float(row.get(column, ""))
            default = row["dataset"] in FIRST_STAGE and row["first_stage"] == FIRST_STAGE[row["dataset"]]
            if default and value is not None:
                found[(system, row["dataset"], row["variant"])] = value
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


# The compact RQ2 table: per backbone the base model, each decision
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
            (["laya"], "\\laya{}", "decision"),
            (["monot5-3b"], "monoT5-3B", "reranker"),
            (["bge-m3"], "bge-reranker-v2-m3", "reranker"),
            (["set-encoder-large"], "Set-Encoder-large", "reranker"),
        ],
    )
]


# Groups in the left panel of the calibration table; the rest goes right, so both panels have about the same height.
CALIBRATION_LEFT_PANEL = {"Qwen3-0.6B", "Qwen3-8B", "Qwen3.5-0.8B", "Other systems"}


def count_median_reference(values: Dict[Tuple[str, str], float]) -> Optional[float]:
    """Mean over DL 2019-2023 of the count test's reference (always guessing the median count). It depends only on
    the labels, so any system's value for a collection will do."""
    per_dataset = {d: v for (_, d), v in values.items()}
    if not all(d in per_dataset for d in DL_DATASETS):
        return None
    return sum(per_dataset[d] for d in DL_DATASETS) / len(DL_DATASETS)


def calibration_table(
    cutoff: CutoffRows,
    posthoc: PosthocRows,
    ece: Dict[Tuple[str, str], float],
    brier: Optional[Dict[Tuple[str, str], float]] = None,
) -> str:
    """RQ2 in one table, for the paper's question: mean ECE over DL 2019-2023 (native, temperature, Platt), Brier score,
    set F1 at p >= 0.5 (native, Platt), the regret of 0.5 against the per-collection oracle cutoff, the mean set size
    at 0.5 and the count error. Decision models in their better-calibrated mode; per group and column the best value
    in bold, the second underlined (lower is better, except F1)."""
    brier = brier or {}

    def posthoc_mean(system: str, method: str, column: str) -> Optional[float]:
        values = {
            (system, d): float(posthoc[(system, d, method)][column])
            for d in DL_DATASETS
            if (system, d, method) in posthoc and _float(posthoc[(system, d, method)].get(column, "")) is not None
        }
        return _mean_of(values, system, DL_DATASETS)

    # The count test's reference (always guessing the median count) depends only on the labels: one value for all.
    summaries = (cutoff_summary(cutoff, system) for system in sorted({key[0] for key in cutoff}))
    references = [s["count_ref"] for s in summaries if s and "count_ref" in s]
    reference = f" (always guessing the median: {references[0]:.1f})" if references else ""
    lines = [
        "% Generated by `python -m cdm.paper_tables` (make tables, in paper/) from the rerank, `cdm cutoff` and",
        "% `cdm posthoc` rows of results/results.csv -- do not edit by hand.",
        "\\begin{table*}[!t]",
        "\\caption{Calibration and its uses on the judged pairs of TREC DL 2019--2023 (means). Per backbone:",
        "untuned model, decision fine-tunes in their better-calibrated mode (j/i: joint/isolated), reranker fine-tunes.",
        "ECE (lower is better): N native; T, P after temperature/Platt scaling fitted on MS~MARCO dev (rerankers",
        "only). Brier: mean squared error of the probability. $F_1$@.5 ($\\times 100$): set kept at $p \\ge 0.5$, native",
        "(N) and Platt-scaled (P); Regret: $F_1$ at each collection's best cutoff minus $F_1$@.5; $|S|$: passages kept",
        "at 0.5 (of 100). Count: mean absolute error of the summed probabilities against the number of relevant",
        f"passages{reference}. Per backbone and column: best in bold, second best underlined.}}",
        "\\label{tab:calibration}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.2pt}",
        "\\begin{adjustbox}{max width=\\textwidth}",
    ]
    header = [
        "\\begin{tabular}[t]{clrrrrrrrrr}",
        "\\toprule",
        " & & \\multicolumn{3}{c}{ECE} & & \\multicolumn{2}{c}{$F_1$@.5} & & & \\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){7-8}",
        " & System & N & T & P & Brier & N & P & Regret & $|S|$ & Count \\\\",
    ]
    # Lower is better, except for F1 (values 4 and 5); |S| (value 7) is not ranked.
    higher_better = {4, 5}
    unranked = {7}
    panels: List[List[str]] = [list(header), list(header)]
    for heading, members in CALIBRATION_GROUPS:
        panel = panels[0 if heading in CALIBRATION_LEFT_PANEL else 1]
        rows = []
        for systems, name, objective in members:
            scored = [(e, s) for e, s in ((_mean_of(ece, s, DL_DATASETS), s) for s in systems) if e is not None]
            if not scored:
                continue
            mean_ece, system = min(scored)
            mode = ""
            if objective == "decision" and system != "laya":
                mode = "joint" if system.endswith("-joint") else "isolated"
                if len(systems) > 1:
                    name += " (j)" if mode == "joint" else " (i)"
            elif system == "laya":
                mode = "isolated"
            summary = cutoff_summary(cutoff, system) or {}
            values = [
                mean_ece,
                posthoc_mean(system, "temperature", "ECE"),
                posthoc_mean(system, "platt", "ECE"),
                _mean_of(brier, system, DL_DATASETS),
                summary.get("F1@0.5"),
                posthoc_mean(system, "platt", "cut_F1@0.5"),
                summary.get("regret"),
                summary.get("size@0.5"),
                summary.get("count_MAE"),
            ]
            rows.append((_symbol(objective, mode), name, objective, values))
        formats = ["{:.3f}"] * 4 + ["{:.1f}"] * 5
        scales = [1.0] * 4 + [100.0] * 3 + [1.0] * 2
        printed = [
            [None if v is None else f.format(v * k) for v, f, k in zip(values, formats, scales)]
            for _, _, _, values in rows
        ]

        def marks(column: int) -> Tuple[Optional[str], Optional[str]]:
            if column in unranked:
                return None, None
            found = sorted({p[column] for p in printed if p[column] is not None}, key=float,
                           reverse=column in higher_better)
            return (found[0] if found else None), (found[1] if len(found) > 1 else None)

        column_marks = [marks(i) for i in range(len(formats))]
        panel += ["\\midrule", f"\\multicolumn{{11}}{{l}}{{\\textit{{{heading}}}}} \\\\"]
        for (symbol, name, objective, _), cells in zip(rows, printed):
            out = ["--" if t is None else _emphasize(t, *column_marks[i]) for i, t in enumerate(cells)]
            panel.append(_shade(objective) + " & ".join([symbol, name] + out) + " \\\\")
    for panel in panels:
        panel += ["\\bottomrule", "\\end{tabular}"]
    lines += panels[0] + ["\\quad"] + panels[1]
    lines += ["\\end{adjustbox}", "\\end{table*}", ""]
    return "\n".join(lines)


# The question ablation: every system that asks a question, with S1Rank's question
# and with the paper's. Rerankers ask none and are left out.
ABLATION_GROUPS: List[Tuple[str, List[Tuple[str, str, str, str]]]] = [
    (backbone, [r for r in rows if r[2] != "reranker"]) for backbone, rows in TRIPLETS
] + [
    (
        "Other systems",
        [
            ("jev-joint", "\\jev{}", "decision", "joint"),
            ("jev-isolated", "\\jev{}", "decision", "isolated"),
            ("laya", "\\laya{}", "decision", "isolated"),
        ],
    )
]


def question_ablation_table(
    cells: Dict[str, Cells], ece: Dict[str, Dict[Tuple[str, str], float]], count: Dict[str, Dict[Tuple[str, str], float]]
) -> str:
    """S1Rank's question against the paper's, per question-asking system: nDCG@10 over all ten collections and over
    TREC DL, ECE and count error over DL 2019-2023 (each argument maps a question to its values)."""
    old, new = ABLATION_QUESTION, QUESTION
    datasets = [d for d, _ in DATASETS]

    def pair(values: Tuple[Optional[float], Optional[float]], fmt: str, scale: float) -> List[str]:
        a, b = values
        out = ["--" if v is None else fmt.format(v * scale) for v in (a, b)]
        if a is None or b is None:
            return out + ["--"]
        delta = float(fmt.format(b * scale)) - float(fmt.format(a * scale))
        return out + [("$+$" if delta > 0 else "$-$" if delta < 0 else "\\phantom{$+$}") + fmt.format(abs(delta))]

    lines = [
        "% Generated by `python -m cdm.paper_tables` (make tables, in paper/) from the rerank and `cdm cutoff` rows of",
        "% results/results.csv -- do not edit by hand.",
        "\\begin{table*}[!t]",
        "\\caption{The question ablation: S1Rank's question (\\emph{Rel.}: ``Is document \\{doc\\} relevant to the",
        "query? Relevant means it answers the query or contains information that directly addresses it.'') against",
        "the question used in the rest of the paper (\\emph{Ans.}: ``Does document \\{doc\\} answer the query?''), for",
        "every system that is asked a question. nDCG@10 ($\\times 100$): mean over all ten collections (Avg) and over",
        "TREC DL 2019--2023 (DL); ECE and Count as in Table~\\ref{tab:calibration} (DL 2019--2023). $\\Delta$: Ans.\\ minus",
        f"Rel. Training: {SYMBOL_LEGEND}.}}",
        "\\label{tab:question-ablation}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{adjustbox}{max width=\\textwidth}",
        "\\begin{tabular}{cll" + "rrr" * 4 + "}",
        "\\toprule",
        " & & & \\multicolumn{3}{c}{nDCG@10 Avg} & \\multicolumn{3}{c}{nDCG@10 DL} & \\multicolumn{3}{c}{ECE}"
        " & \\multicolumn{3}{c}{Count} \\\\",
        "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-12}\\cmidrule(lr){13-15}",
        " & System & Mode" + " & Rel. & Ans. & $\\Delta$" * 4 + " \\\\",
    ]
    for heading, rows in ABLATION_GROUPS:
        lines += ["\\midrule", f"\\multicolumn{{15}}{{l}}{{\\textit{{{heading}}}}} \\\\"]
        for system, name, objective, mode in rows:
            out = [_symbol(objective, mode), name, mode]
            out += pair(tuple(_mean(cells[q], system, datasets) for q in (old, new)), "{:.1f}", 100)
            out += pair(tuple(_mean(cells[q], system, DL_DATASETS) for q in (old, new)), "{:.1f}", 100)
            out += pair(tuple(_mean_of(ece[q], system, DL_DATASETS) for q in (old, new)), "{:.3f}", 1)
            out += pair(tuple(_mean_of(count[q], system, DL_DATASETS) for q in (old, new)), "{:.1f}", 1)
            lines.append(_shade(objective) + " & ".join(out) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table*}", ""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--results", default=RESULTS_FOLDER, help="Folder with results.csv and literature.csv.")
    parser.add_argument("--out", default=TABLES_FOLDER, help="Folder for the .tex tables.")
    args = parser.parse_args(argv)
    cells, cost = load_cells(args.results)
    os.makedirs(args.out, exist_ok=True)
    questions = (ABLATION_QUESTION, QUESTION)
    tables = {
        "triplets": triplet_table(cells, load_significance(args.results)),
        "context": context_table(cells, cost),
        "calibration": calibration_table(
            load_cutoff(args.results),
            load_posthoc(args.results),
            load_rerank_column("ECE", args.results),
            load_rerank_column("Brier", args.results),
        ),
        "question-ablation": question_ablation_table(
            {q: load_cells(args.results, q)[0] for q in questions},
            {q: load_rerank_column("ECE", args.results, q) for q in questions},
            {q: load_rerank_column("count_MAE", args.results, q, job="cutoff") for q in questions},
        ),
    }
    for name, tex in tables.items():
        path = os.path.join(args.out, f"{name}.tex")
        with open(path, "w") as f:
            f.write(tex)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

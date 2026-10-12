import csv
import re

from cdm.paper_tables import context_table, main, triplet_table

RESULT_COLUMNS = ["job", "system", "model", "dataset", "first_stage", "phrasing", "variant", "subset", "nDCG@10",
                  "api_cost_usd_per_1k_queries"]


def _write(path, columns, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def test_context_table(tmp_path):
    _write(
        tmp_path / "results.csv",
        RESULT_COLUMNS,
        [
            {"job": "first-stage", "system": "bm25", "dataset": "dl19", "variant": "d100", "nDCG@10": "0.5058"},
            {"job": "rerank", "system": "jev-joint", "dataset": "dl19", "first_stage": "bm25-d100",
             "nDCG@10": "0.7329", "api_cost_usd_per_1k_queries": "0.5"},
            # DL21 is reranked on the official top-100, so a bm25-d100 row there is not the paper's setup.
            {"job": "rerank", "system": "jev-joint", "dataset": "dl21", "first_stage": "bm25-d100", "nDCG@10": "0.9"},
            {"job": "rerank", "system": "jev-joint", "dataset": "dl22", "first_stage": "official-d100",
             "nDCG@10": "0.4999", "subset": "q20"},
            {"job": "rerank", "system": "bge-m3", "dataset": "dl19", "first_stage": "bm25-d100", "nDCG@10": "0.1"},
        ],
    )
    _write(
        tmp_path / "literature.csv",
        ["system", "dataset", "metric", "value", "candidates", "role"],
        [
            {"system": "bge-m3", "dataset": "dl19", "metric": "nDCG@10", "value": "0.7219",
             "candidates": "bm25-d100", "role": "baseline"},
            {"system": "jev-joint", "dataset": "dl19", "metric": "nDCG@10", "value": "0.737",
             "candidates": "bm25-d100", "role": "reference"},
        ],
    )
    main(["--results", str(tmp_path), "--out", str(tmp_path)])
    tex = (tmp_path / "context.tex").read_text()
    rows = {line.split(" & ")[0] + "/" + line.split(" & ")[1]: line.split(" & ") for line in tex.splitlines()
            if " & " in line and not line.startswith(("System", " &"))}
    assert rows["BM25/"][2] == "50.6"
    jev = rows["\\jev{}/joint"]
    assert jev[2] == "\\textbf{73.3}"  # best in the column; the literature reference row is ignored
    assert jev[4] == "--" and jev[5] == "--"  # wrong candidates, quick subset
    assert jev[-1] == "0.50 \\\\"
    assert rows["bge-reranker-v2-m3/"][2] == "72.2\\rlap{$^\\dagger$}"  # literature wins over the computed cell
    assert "\\label{tab:context}" in tex and context_table({}, {}).count("--") > 0


def test_triplet_table(tmp_path):
    zero_shot = ["trec-covid", "touche", "dbpedia"]
    rows = [{"job": "rerank", "system": s, "dataset": d, "first_stage": "bm25-d100", "nDCG@10": v}
            for s, v in [("qwen3-reranker-0.6b", "0.6"), ("kev-0.6b-isolated", "0.3")] for d in zero_shot]
    rows.append({"job": "rerank", "system": "kev-0.6b-joint", "dataset": "trec-covid", "first_stage": "bm25-d100",
                 "nDCG@10": "0.7"})
    _write(tmp_path / "results.csv", RESULT_COLUMNS, rows)
    _write(tmp_path / "literature.csv", ["system", "dataset", "metric", "value", "candidates", "role"], [])
    main(["--results", str(tmp_path), "--out", str(tmp_path)])
    tex = (tmp_path / "triplets.tex").read_text()
    # Rows start with their shading (\rowcolor{...}); the Training column is gone, the shading encodes it.
    rows = [re.sub(r"^\\rowcolor\{\w+\}", "", line).split(" & ") for line in tex.splitlines() if line.count(" & ") > 5]
    assert "\\rowcolor{rowreranker}Qwen3-Reranker-0.6B & " in tex
    lines = {tuple(cells[:2]): cells for cells in rows}
    reranker = lines[("Qwen3-Reranker-0.6B", "")]
    assert reranker[-1] == "\\textbf{60.0} \\\\"  # ZS mean, best of the backbone
    assert reranker[-2] == "--"  # no mean over all ten without every dataset
    joint = lines[("Kev-0.6B", "joint")]
    assert joint[2 + 3] == "\\textbf{70.0}"  # TREC-COVID (DL20, DL22 hidden): best of the backbone in that column
    assert joint[-1] == "-- \\\\"  # zero-shot mean needs all three collections
    assert "\\label{tab:triplets}" in tex and triplet_table({}).count("--") > 0


def test_posthoc_table():
    from cdm.paper_tables import BEIR_DATASETS, DL_DATASETS, posthoc_table

    rows = {}
    for d in DL_DATASETS:
        for m, ece, f1 in (("native", "0.30", "0.40"), ("temperature", "0.20", "0.40"), ("platt", "0.10", "0.45")):
            rows[("qwen3-reranker-8b", d, m)] = {"ECE": ece, "cut_F1@0.5": f1, "cut_regret": "0.05",
                                                 "ECE_unjudged0": "0.2"}
    for d in BEIR_DATASETS[:-1]:  # one BEIR collection missing: no BEIR mean
        rows[("qwen3-reranker-8b", d, "native")] = {"ECE_unjudged0": "0.2"}
    ece = {("kev-27b-joint", d): 0.15 for d in DL_DATASETS}
    tex = posthoc_table(rows, {}, ece, {})
    line = next(x for x in tex.splitlines() if x.startswith("Qwen3-Reranker-8B"))
    assert line == "Qwen3-Reranker-8B & 0.300 & 0.200 & 0.100 & 40.0 & 45.0 & 5.0 & 5.0 & -- & -- \\\\"
    kev = next(x for x in tex.splitlines() if x.startswith("Kev-27B joint"))
    assert kev.startswith("Kev-27B joint & 0.150 & -- & --") and "\\label{tab:posthoc}" in tex


def test_calibration_table():
    from cdm.paper_tables import DL_DATASETS, calibration_table

    ece = {("kev-27b-joint", d): 0.15 for d in DL_DATASETS}
    ece.update({("kev-27b-isolated", d): 0.25 for d in DL_DATASETS})
    ece.update({("prism-4b", d): 0.2 for d in DL_DATASETS})
    cutoff = {("kev-27b-joint", d): {"cut_F1@0.5": "0.4", "cut_F1@oracle": "0.5", "cut_regret": "0.1",
                                     "cut_oracle_tau": "0.5", "cut_size@0.5": "8", "topk_F1@10": "0.3",
                                     "count_MAE": "7", "count_median_ref": "8.6"}
              for d in DL_DATASETS}
    posthoc = {("prism-4b", d, "temperature"): {"ECE": "0.1"} for d in DL_DATASETS}
    tex = calibration_table(cutoff, posthoc, ece)
    lines = tex.splitlines()
    assert "Kev-27B (j) & 0.150 & -- & -- & 40.0 & -- & 10.0 & 8.0 & 7.0 \\\\" in lines  # better-calibrated mode
    assert "Prism-4B & 0.200 & 0.100 & -- & -- & -- & -- & -- & -- \\\\" in lines
    assert "\\label{tab:calibration-full}" in tex and "median count: 8.6" in tex


def test_calibration_ece_brier_table():
    from cdm.paper_tables import DL_DATASETS, calibration_ece_brier_table

    ece = {("kev-4b-joint", d): 0.25 for d in DL_DATASETS}
    ece.update({("kev-4b-isolated", d): 0.2 for d in DL_DATASETS})
    ece.update({("prism-4b", d): 0.3 for d in DL_DATASETS})
    brier = {("kev-4b-isolated", d): 0.21 for d in DL_DATASETS}
    brier.update({("prism-4b", d): 0.17 for d in DL_DATASETS})
    lines = calibration_ece_brier_table(ece, brier).splitlines()
    # Better-calibrated mode chosen by ECE; per group the lowest value bold, the second lowest underlined.
    assert "\\rowcolor{rowdecision}Kev-4B (i) & \\textbf{0.200} & \\underline{0.210} \\\\" in lines
    assert "\\rowcolor{rowreranker}Prism-4B & \\underline{0.300} & \\textbf{0.170} \\\\" in lines


def test_context_short_table():
    from cdm.paper_tables import DATASETS, context_short_table

    cells = {("jev-joint", d): (0.5, d == "dl19") for d, _ in DATASETS}
    tex = context_short_table(cells)
    row = next(x for x in tex.splitlines() if x.startswith("\\rowcolor{rowdecision}\\jev{} & System One, joint"))
    # The only reranker in the table: best in every column (bold), dagger where a published number is included.
    assert row.split(" & ")[2:] == ["\\textbf{50.0}\\rlap{$^\\dagger$}", "\\textbf{50.0}",
                                    "\\textbf{50.0}\\rlap{$^\\dagger$}", "\\textbf{50.0} \\\\"]


def test_triplet_table_significance_marks(tmp_path):
    from cdm.paper_tables import load_significance

    columns = ["job", "system", "dataset", "variant", "subset", "sig_comparison", "sig_holm"]
    _write(tmp_path / "results.csv", columns, [
        {"job": "significance", "system": "prism-4b", "dataset": "dl19", "variant": "vs-kev-4b-isolated",
         "sig_comparison": "reranker-vs-decision", "sig_holm": "1"},
        {"job": "significance", "system": "prism-4b", "dataset": "dl20", "variant": "vs-kev-4b-isolated",
         "sig_comparison": "reranker-vs-decision", "sig_holm": "0"},
    ])
    marks = load_significance(str(tmp_path))
    assert marks == {("prism-4b", "dl19"): "\\ast"}
    tex = triplet_table({("prism-4b", "dl19"): (0.744, False)}, marks)
    assert "\\textbf{74.4}\\rlap{$^{\\ast}$}" in tex

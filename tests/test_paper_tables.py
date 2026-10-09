import csv

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
    lines = {tuple(line.split(" & ")[:3]): line.split(" & ") for line in tex.splitlines() if line.count(" & ") > 5}
    reranker = lines[("Qwen3-Reranker-0.6B", "reranker", "")]
    assert reranker[-1] == "\\textbf{60.0} \\\\"  # ZS mean, best of the backbone
    assert reranker[-2] == "--"  # no mean over all ten without every dataset
    joint = lines[("Kev-0.6B", "decision", "joint")]
    assert joint[3 + 5] == "\\textbf{70.0}"  # TREC-COVID: best of the backbone in that column
    assert joint[-1] == "-- \\\\"  # zero-shot mean needs all three collections
    assert "\\label{tab:triplets}" in tex and triplet_table({}).count("--") > 0

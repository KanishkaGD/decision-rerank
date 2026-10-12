import csv
import gzip
import json

import numpy as np
import pytest

from cdm.cutoff import QueryScores, cutoff_analysis, read_scores, threshold_metrics, topk_metrics
from cdm.paper_tables import DL_DATASETS, cutoff_summary, cutoff_table, load_cutoff

# Ranking order; labels 1 relevant, 0 judged non-relevant, -1 unjudged.
A = QueryScores("a", np.array([0.9, 0.6, 0.4, 0.2]), np.array([1, 0, 1, -1]))
B = QueryScores("b", np.array([0.4, 0.2]), np.array([0, -1]))  # no relevant candidate


def test_threshold_metrics_by_hand():
    m = threshold_metrics([A], [0.3, 0.5, 0.7])
    assert m["size"].tolist() == [3, 2, 1]
    assert m["P"] == pytest.approx([2 / 3, 1 / 2, 1])
    assert m["R"] == pytest.approx([1, 1 / 2, 1 / 2])
    assert m["F1"] == pytest.approx([0.8, 0.5, 2 / 3])
    assert m["unjudged"].tolist() == [0, 0, 0]
    assert threshold_metrics([A], [0.1])["unjudged"] == pytest.approx([1 / 4])


def test_query_without_relevant_candidates_rewards_abstention():
    m = threshold_metrics([B], [0.3, 0.5])
    assert m["F1"].tolist() == [0.0, 1.0]  # keeps one passage at 0.3, nothing at 0.5
    assert m["empty"].tolist() == [0.0, 1.0]
    both = threshold_metrics([A, B], [0.5])
    assert both["F1"] == pytest.approx([(0.5 + 1.0) / 2])  # macro average


def test_topk_follows_the_ranking():
    m = topk_metrics([A], [1, 3])
    assert m["F1"] == pytest.approx([2 / 3, 0.8])


def test_oracle_and_regret():
    out = cutoff_analysis([A])
    assert out["cut_F1@0.5"] == pytest.approx(0.5)
    assert out["cut_F1@oracle"] == pytest.approx(0.8)
    assert out["cut_oracle_tau"] == pytest.approx(0.201)  # lowest cutoff that keeps exactly the top three
    assert out["cut_regret"] == pytest.approx(0.3)
    assert out["topk_F1@5"] == pytest.approx(2 / 3) and out["cut_n_queries_no_rel"] == 0  # all four kept
    assert cutoff_analysis([]) == {}


def test_read_scores(tmp_path):
    path = tmp_path / "scores.jsonl.gz"
    rows = [
        {"qid": "q1", "docid": "d1", "first_stage_rank": 1, "score": 0.2, "prob": 0.2},
        {"qid": "q1", "docid": "d2", "first_stage_rank": 2, "score": 0.9, "prob": 0.9},
        {"qid": "q1", "docid": "d3", "first_stage_rank": 3, "score": 0.9, "prob": 0.9},  # tie: first-stage rank
        {"qid": "q2", "docid": "d1", "first_stage_rank": 1, "score": 0.5, "prob": 0.5},  # unjudged query
        {"qid": "q1", "docid": "d4", "first_stage_rank": 4, "score": 3.0, "prob": None},  # no probability
    ]
    with gzip.open(path, "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    (q,) = read_scores(str(path), {"q1": {"d2": 3, "d3": 1}}, rel_threshold=2)
    assert q.qid == "q1" and q.probs.tolist() == [0.9, 0.9, 0.2]
    assert q.labels.tolist() == [1, 0, -1]


def test_cutoff_table(tmp_path):
    columns = ["job", "system", "model", "dataset", "first_stage", "phrasing", "variant", "subset", "cut_F1@0.5",
               "cut_F1@oracle", "cut_regret", "cut_oracle_tau", "cut_size@0.5", "topk_F1@10"]
    first = {"dl19": "bm25-d100", "dl20": "bm25-d100", "dl21": "official-d100", "dl22": "official-d100",
             "dl23": "official-d100"}
    rows = []
    for i, (dataset, stage) in enumerate(first.items()):
        values = {"cut_F1@0.5": "0.4", "cut_F1@oracle": "0.5", "cut_regret": "0.1", "cut_oracle_tau": str(0.3 + i / 10),
                  "cut_size@0.5": "8", "topk_F1@10": "0.35"}
        rows.append({"job": "cutoff", "system": "kev-27b-joint", "dataset": dataset, "first_stage": stage, **values})
        # bge-m3 on DL19/20 only from the efficiency run (its default cells are literature numbers)
        variant = "a100" if dataset in ("dl19", "dl20") else ""
        rows.append({"job": "cutoff", "system": "bge-m3", "dataset": dataset, "first_stage": stage, "variant": variant,
                     **values})
    rows.append({"job": "cutoff", "system": "prism-4b", "dataset": "dl19", "first_stage": "bm25-d100", **values})
    with open(tmp_path / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({c: r.get(c, "") for c in columns} for r in rows)
    loaded = load_cutoff(str(tmp_path))
    s = cutoff_summary(loaded, "kev-27b-joint")
    assert s["F1@0.5"] == pytest.approx(0.4) and (s["tau_min"], s["tau_max"]) == pytest.approx((0.3, 0.7))
    assert cutoff_summary(loaded, "bge-m3") is not None
    assert cutoff_summary(loaded, "prism-4b") is None  # incomplete
    tex = cutoff_table(loaded, {("kev-27b-joint", d): 0.15 for d in DL_DATASETS})
    line = next(x for x in tex.splitlines() if x.startswith("Kev-27B & decision & joint"))
    assert line == "Kev-27B & decision & joint & 0.150 & 40.0 & 10.0 & 0.30--0.70 & 8.0 & 35.0 \\\\"
    assert "Prism-4B & reranker &  & -- & -- & -- & -- & -- & -- \\\\" in tex


def test_count_test_by_hand():
    from cdm.cutoff import count_test

    out = count_test([A, B])
    # A: judged probabilities 0.9 + 0.6 + 0.4 = 1.9 against 2 relevant; all four sum to 2.1
    # B: judged 0.4 against 0 relevant; both sum to 0.6
    assert out["count_MAE"] == pytest.approx((0.1 + 0.4) / 2)
    assert out["count_MAE_full"] == pytest.approx((0.1 + 0.6) / 2)
    assert out["count_ratio"] == pytest.approx((1.9 + 0.4) / 2)  # mean truth is (2 + 0) / 2 = 1
    assert out["count_ratio_full"] == pytest.approx((2.1 + 0.6) / 2)
    assert out["count_median_ref"] == pytest.approx(1.0)  # median count 1, errors 1 and 1
    assert count_test([]) == {}
    assert "count_MAE" in cutoff_analysis([A, B])

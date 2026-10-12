"""Training groups for our reranker fine-tunes: leakage filter, query sampling, group building, the source reader."""

import gzip
import json
import random

import pytest

from cdm.finetune_data import (
    LeakageFilter,
    bag_of_words,
    build_group,
    build_groups,
    line_qid,
    normalize_query,
    parse_quota,
    read_groups,
    select_qids,
    source_qids,
    write_groups,
)


def test_normalize_and_bag_of_words():
    assert normalize_query("  What IS the cost-of  a 'Tesla'?") == "what is the cost of a tesla"
    assert bag_of_words("cost of tesla") == bag_of_words("Tesla, cost of!") == "cost of tesla"


def test_leakage_filter_reasons():
    f = LeakageFilter.from_queries([("dl19", "1037798", "Who is Robert Gray?"), ("scifact", "7", "x causes y")])
    assert f.reason("1037798", "unrelated") == "qid:dl19"
    assert f.reason("5", "who is robert gray") == "text:dl19"
    assert f.reason("5", "Y causes X") == "bow:scifact"
    assert f.reason("5", "who is robert grey") is None


def test_select_qids_deterministic_and_nested():
    ids = [str(i) for i in range(1000)]
    a = select_qids(ids, 50, seed=1)
    assert a == select_qids(list(reversed(ids)), 50, seed=1)  # independent of file order
    assert select_qids(ids, 100, seed=1)[:50] == a  # a larger sample contains the smaller one
    assert a != select_qids(ids, 50, seed=2)
    assert len(select_qids(ids + ids, None, seed=1)) == 1000


def p(docid, text=None):
    return {"docid": docid, "text": text or f"text {docid}"}


def test_build_group_quota_dedup_and_top_up():
    rng = random.Random(0)
    negs = {
        "a": [p("d1"), p("d2"), p("pos2"), p("d3")],  # pos2 is a positive: never a negative
        "b": [p("d1"), p("d9")],  # d1 duplicates source a
    }
    g = build_group("q", "query", [p("pos1"), p("pos2")], negs, {"a": 2, "b": 2}, rng)
    ids = [n["docid"] for n in g["negs"]]
    assert len(ids) == 4 and len(set(ids)) == 4
    assert not {"pos1", "pos2"} & set(ids)
    assert g["pos"]["docid"] in {"pos1", "pos2"}
    # b has only one new negative (d9) after deduplication, so a tops it up
    assert sum(n["source"] == "a" for n in g["negs"]) == 3 and "d9" in ids
    assert build_group("q", "query", [], negs, {"a": 1}, rng) is None
    assert build_group("q", "query", [p("pos1")], negs, {"a": 3, "b": 3}, rng) is None  # only 4 distinct negatives


def test_parse_quota():
    assert parse_quota("tevatron-aug:8,tevatron-bm25:7") == {"tevatron-aug": 8, "tevatron-bm25": 7}
    with pytest.raises(KeyError):
        parse_quota("nope:3")


def write_source(path, rows):
    with gzip.open(path, "wt") as f:
        for qid, query, pos, negs in rows:
            f.write(
                json.dumps(
                    {
                        "query_id": qid,
                        "query": query,
                        "positive_passages": [{"docid": d, "title": "T", "text": f"pos {d}"} for d in pos],
                        "negative_passages": [{"docid": d, "title": "T", "text": f"neg {d}"} for d in negs],
                    }
                )
                + "\n"
            )


def test_line_qid_and_source_qids(tmp_path):
    path = str(tmp_path / "s.jsonl.gz")
    write_source(path, [("11", 'a "quoted" query', ["p"], ["n"])])
    with gzip.open(path, "rt") as f:
        assert line_qid(f.readline()) == "11"
    assert list(source_qids(path)) == [("11", 'a "quoted" query')]


def test_build_groups_end_to_end(tmp_path):
    a, b = str(tmp_path / "a.jsonl.gz"), str(tmp_path / "b.jsonl.gz")
    rows_a = [(str(i), f"query number {i}", [f"p{i}"], [f"a{i}_{j}" for j in range(5)]) for i in range(30)]
    rows_a.append(("dl", "who is robert gray", ["px"], [f"ax{j}" for j in range(5)]))  # a test query's text
    write_source(a, rows_a)
    write_source(b, [(str(i), f"query number {i}", [f"p{i}"], [f"b{i}_{j}" for j in range(5)]) for i in range(20)])
    leakage = LeakageFilter.from_queries([("dl19", "999", "Who is Robert Gray?"), ("dl20", "3", "x")])
    groups, stats = build_groups(
        {"tevatron-aug": a, "tevatron-bm25": b}, {"tevatron-aug": 2, "tevatron-bm25": 2}, leakage, n_queries=10, seed=0
    )
    assert len(groups) == 10 and stats["group_size"] == 5
    assert stats["removed_by_leakage_filter"] == {"qid:dl20": 1, "text:dl19": 1}
    assert stats["eligible"] == 19  # in both sources (0-19) minus qid 3
    qids = {g["qid"] for g in groups}
    assert "3" not in qids and "dl" not in qids and all(int(q) < 20 for q in qids)
    for g in groups:
        assert g["pos"]["text"] == f"pos p{g['qid']}"  # text only, no title
        assert sorted(n["source"] for n in g["negs"]) == ["tevatron-aug"] * 2 + ["tevatron-bm25"] * 2
    again, _ = build_groups(
        {"tevatron-aug": a, "tevatron-bm25": b}, {"tevatron-aug": 2, "tevatron-bm25": 2}, leakage, n_queries=10, seed=0
    )
    assert again == groups  # deterministic
    out = str(tmp_path / "g" / "groups.jsonl.gz")
    assert write_groups(out, groups) == 10 and read_groups(out) == groups

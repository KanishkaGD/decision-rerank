"""RankZephyr's sliding windows, permutation parsing and score assembly (no model download)."""

from cdm.phrasings import get_phrasing
from cdm.rerankers.listwise import RankZephyrReranker, parse_permutation, replace_number, windows


def test_windows_like_rank_llm():
    assert windows(100, 20, 10) == [(80, 100), (70, 90), (60, 80), (50, 70), (40, 60), (30, 50), (20, 40), (10, 30),
                                    (0, 20)]
    assert windows(15, 20, 10) == [(0, 15)]
    assert windows(0, 20, 10) == []


def test_parse_permutation():
    assert parse_permutation("[3] > [1] > [2]", 3) == [2, 0, 1]
    assert parse_permutation("[2] > [2] > [9] > [1]", 3) == [1, 0, 2]  # duplicates and unknown ids dropped
    assert parse_permutation("I cannot rank these.", 3) == [0, 1, 2]


def test_replace_number():
    assert replace_number("see [12] and [a]") == "see (12) and [a]"


def test_score_reverses_by_windows(monkeypatch):
    r = RankZephyrReranker.__new__(RankZephyrReranker)
    r.window, r.stride = 4, 2
    r.reset_stats()

    def fake_window(query, passages):  # always reverses the window
        return {"answer": "", "order": list(range(len(passages)))[::-1], "input_tokens": 0, "output_tokens": 0}

    monkeypatch.setattr(r, "_rank_window", fake_window)
    out = r.score("q", ["a", "b", "c", "d", "e", "f"], get_phrasing("none"))
    # windows (2,6), (0,4): [a b c d e f] -> [a b f e d c] -> [e f b a d c]
    order = sorted(range(6), key=lambda i: -out.scores[i])
    assert order == [4, 5, 1, 0, 3, 2] and out.probs is None and len(out.raw) == 2

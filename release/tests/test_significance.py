import pytest

from cdm.significance import comparisons, holm_flags


def test_comparisons_pick_the_best_decision_mode():
    averages = {"kev-4b-joint": 0.498, "kev-4b-isolated": 0.513, "prism-4b": 0.587, "qwen3.5-4b-prompted": 0.527}
    pairs = comparisons(averages)
    assert ("prism-4b", "kev-4b-isolated", "reranker-vs-decision") in pairs
    assert ("kev-4b-isolated", "qwen3.5-4b-prompted", "decision-vs-base") in pairs
    assert ("kev-4b-joint", "kev-4b-isolated", "joint-vs-isolated") in pairs
    assert ("jev-joint", "jev-isolated", "joint-vs-isolated") in pairs
    # A backbone without complete averages gets no reranker-vs-decision pair.
    assert not any(a == "prism-9b" for a, _, c in pairs if c == "reranker-vs-decision")


def test_holm_flags():
    pytest.importorskip("pandas")  # cdm.evaluation imports it
    assert holm_flags([0.001, 0.04, 0.2]) == [True, False, False]


def test_paired_t():
    pytest.importorskip("scipy")
    from cdm.significance import paired_t

    a = {"q1": 0.9, "q2": 0.8, "q3": 0.7, "q4": 0.95}
    b = {"q1": 0.5, "q2": 0.4, "q3": 0.45, "q4": 0.5, "q5": 1.0}  # q5: only one system, left out
    result = paired_t(a, b)
    assert result["n"] == 4 and result["delta"] == pytest.approx(0.375) and result["p"] < 0.01
    assert paired_t(a, a)["p"] == 1.0


def test_comparisons_with_another_question_use_its_base_models():
    averages = {"kev-4b-joint": 0.49, "kev-4b-isolated": 0.51, "prism-4b": 0.58, "qwen3.5-4b-prompted-answer": 0.52}
    pairs = comparisons(averages, "answer")
    assert ("kev-4b-isolated", "qwen3.5-4b-prompted-answer", "decision-vs-base") in pairs
    assert ("prism-4b", "kev-4b-isolated", "reranker-vs-decision") in pairs

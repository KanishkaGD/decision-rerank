import math

from cdm.evaluation import calibration, holm, judged_pairs, paired_bootstrap


def test_calibration_perfect_and_worst():
    assert calibration([1.0, 0.0], [1, 0])["ECE"] == 0.0
    assert calibration([1.0, 0.0], [1, 0])["Brier"] == 0.0
    worst = calibration([1.0, 0.0], [0, 1])
    assert math.isclose(worst["ECE"], 1.0) and math.isclose(worst["Brier"], 1.0)


def test_calibration_binned():
    # One bin at 0.7 with 70% positives: calibrated.
    probs = [0.7] * 10
    labels = [1] * 7 + [0] * 3
    assert math.isclose(calibration(probs, labels)["ECE"], 0.0, abs_tol=1e-12)


def test_calibration_empty():
    assert calibration([], [])["n_judged_pairs"] == 0


def test_judged_pairs_threshold_and_unjudged():
    qrels = {"q1": {"a": 3, "b": 1}}
    probs, labels = judged_pairs([("q1", "a", 0.9), ("q1", "b", 0.4), ("q1", "c", 0.5)], qrels, rel_threshold=2)
    assert probs == [0.9, 0.4] and labels == [1, 0]


def test_paired_bootstrap():
    assert paired_bootstrap([0.5] * 30, [0.5] * 30) == 1.0
    a = [0.9] * 30
    b = [0.1] * 30
    assert paired_bootstrap(a, b) < 0.01


def test_holm():
    sig = holm({"x": 0.001, "y": 0.02, "z": 0.04})
    assert sig == {"x": True, "y": True, "z": True}
    sig = holm({"x": 0.001, "y": 0.04, "z": 0.03})
    assert sig == {"x": True, "z": False, "y": False}

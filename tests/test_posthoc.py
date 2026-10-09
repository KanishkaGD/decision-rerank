import numpy as np
import pytest

from cdm.cutoff import QueryScores
from cdm.evaluation import calibration
from cdm.posthoc import evaluate_scaled, fit, fit_logistic, logits, scaled


def _queries(z, labels, size=10):
    """Split arrays into queries of `size` candidates, ranked by z; stored prob = sigmoid(z), score = z."""
    out = []
    for i in range(0, len(z), size):
        zi, li = z[i : i + size], labels[i : i + size]
        order = np.argsort(-zi, kind="stable")
        out.append(QueryScores(f"q{i}", 1 / (1 + np.exp(-zi[order])), li[order], zi[order]))
    return out


def test_platt_recovers_a_known_mapping():
    rng = np.random.default_rng(0)
    z = rng.normal(size=20_000)
    y = (rng.random(z.size) < 1 / (1 + np.exp(-(2.0 * z - 1.0)))).astype(float)
    a, b = fit_logistic(z, y, bias=True)
    assert a == pytest.approx(2.0, abs=0.1) and b == pytest.approx(-1.0, abs=0.1)
    a_t, b_t = fit_logistic(z, y, bias=False)
    assert b_t == 0.0 and a_t > 0


def test_fit_and_scaling():
    rng = np.random.default_rng(1)
    z = rng.normal(2.0, 2.0, size=5_000)  # overconfident: true probabilities are lower
    labels = (rng.random(z.size) < 1 / (1 + np.exp(-(z - 3.0)))).astype(int)
    dev = _queries(z, labels)
    params = fit(dev)
    assert params["native"] == (1.0, 0.0)
    assert params["platt"][1] < 0  # shifts the probabilities down
    native = evaluate_scaled(scaled(dev, *params["native"]), with_cutoff=True)
    platt = evaluate_scaled(scaled(dev, *params["platt"]), with_cutoff=True)
    assert platt["ECE"] < native["ECE"] and platt["NLL_unjudged0"] < native["NLL_unjudged0"]
    # Temperature keeps p = 0.5 at z = 0, so the set kept at 0.5 is unchanged; Platt's bias moves it.
    temp = evaluate_scaled(scaled(dev, *params["temperature"]), with_cutoff=True)
    assert temp["cut_size@0.5"] == pytest.approx(native["cut_size@0.5"])
    assert platt["cut_size@0.5"] < native["cut_size@0.5"]
    # Monotone: ranking-based numbers are untouched.
    assert platt["topk_F1@10"] == pytest.approx(native["topk_F1@10"])


def test_native_matches_the_rerank_calibration():
    z = np.array([2.0, 0.5, -1.0, -3.0, 1.0, 0.0])
    labels = np.array([1, 0, -1, 0, 1, -1])
    (q,) = _queries(z, labels, size=6)
    out = evaluate_scaled(scaled([q], 1.0, 0.0), with_cutoff=False)
    p = 1 / (1 + np.exp(-z))
    judged = labels >= 0
    assert out["ECE"] == pytest.approx(calibration(p[judged], labels[judged])["ECE"])
    assert out["ECE_unjudged0"] == pytest.approx(calibration(p, (labels == 1).astype(int))["ECE"])
    assert "cut_F1@0.5" not in out


def test_logits_fall_back_to_the_probability():
    q = QueryScores("q", np.array([0.9, 0.1]), np.array([1, 0]), np.array([0.9, 0.1]))  # score is the probability
    assert logits([q]) == pytest.approx(np.log([9.0, 1 / 9]))
    q2 = QueryScores("q", 1 / (1 + np.exp(-np.array([3.0, -2.0]))), np.array([1, 0]), np.array([3.0, -2.0]))
    assert logits([q2]) == pytest.approx([3.0, -2.0])


def test_without_judged_nonrelevant_pairs_only_unjudged0():
    (q,) = _queries(np.array([1.0, -1.0, 0.5]), np.array([1, -1, -1]), size=3)
    out = evaluate_scaled([q], with_cutoff=False)
    assert "ECE" not in out and "ECE_unjudged0" in out


def test_temperature_stays_positive():
    rng = np.random.default_rng(2)
    z = rng.uniform(3.0, 6.0, size=2_000)  # uniformly positive logits, few positives: only a < 0 would lower p
    y = (rng.random(z.size) < 0.01).astype(float)
    a, b = fit_logistic(z, y, bias=False)
    assert a > 0 and b == 0.0

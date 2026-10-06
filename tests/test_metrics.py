import numpy as np
import pandas as pd
import pytest

from decisionrerank import calibration as cal
from decisionrerank.effectiveness import holm, per_query, ranked_run
from decisionrerank.probs import fit_temperatures, softmax


def test_ece_zero_for_calibrated_forecasts():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200_000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    assert cal.ece(p, y) < 0.01
    assert cal.adaptive_ece(p, y) < 0.01


def test_ece_of_constant_forecast_is_bias():
    y = np.array([1, 0, 0, 0], float)
    assert cal.ece(np.full(4, 0.9), y) == pytest.approx(0.65)
    assert cal.brier(np.full(4, 0.25), y) == pytest.approx(0.1875)


def test_rps():
    P = np.array([[1, 0, 0, 0], [0.25, 0.25, 0.25, 0.25]], float)
    # exact hit scores 0; uniform vs grade 3: CDF gaps .25,.5,.75 -> (0.0625+0.25+0.5625)/3
    assert cal.rps(P, np.array([0, 3])) == pytest.approx(0.875 / 3 / 2)


def test_bootstrap_brackets_point_estimate():
    rng = np.random.default_rng(1)
    f = pd.DataFrame({"qset": "dl19", "query_id": np.repeat(np.arange(50).astype(str), 40)})
    p = rng.uniform(size=len(f))
    y = (rng.uniform(size=len(f)) < p).astype(float)
    lo, hi = cal.bootstrap(lambda i: cal.brier(p[i], y[i]), cal.query_groups(f), n=200)
    assert lo <= cal.brier(p, y) <= hi


def test_temperature_keeps_binary_ranking():
    z = np.random.default_rng(2).normal(size=(500, 2))
    a, b = softmax(z, 1.0)[:, 1], softmax(z, 3.0)[:, 1]
    assert (np.argsort(a) == np.argsort(b)).all()


def test_laya_fitter_recovers_planted_temperature():
    rng = np.random.default_rng(3)
    n, t_true = 6000, 2.0
    zn, zs = rng.normal(0, 4, (n, 2)), rng.normal(0, 4, (n, 4))
    yn = (rng.uniform(size=n) < softmax(zn, t_true)[:, 1]).astype(int)
    ys = np.array([rng.choice(4, p=p) for p in softmax(zs, t_true)])
    # rel encodes both targets only when they agree, so fit noul and score on separate frames
    fn = pd.DataFrame({"noul_z0": zn[:, 0], "noul_z1": zn[:, 1], **{f"score_z{j}": zs[:, j] for j in range(4)},
                       "rel": np.where(yn == 1, 2, 0)})
    fs = fn.assign(rel=ys)
    assert fit_temperatures(fn)["noul"] == pytest.approx(t_true, rel=0.1)
    assert fit_temperatures(fs)["score"] == pytest.approx(t_true, rel=0.1)


def test_holm():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_ties_broken_by_bm25_rank_and_ndcg():
    f = pd.DataFrame({"query_id": "q", "doc_id": ["a", "b", "c"], "in_pool": True,
                      "s": [0.5, 0.5, 0.1], "bm25_rank": [2, 1, 3]})
    run = ranked_run(f, "s")
    assert sorted(run["q"], key=run["q"].get, reverse=True) == ["b", "a", "c"]
    qrels = pd.DataFrame({"query_id": "q", "doc_id": ["a"], "rel": [3]})
    v = per_query(f, "s", qrels).loc["q"]
    assert v["nDCG@10"] == pytest.approx(1 / np.log2(3))  # relevant doc at rank 2, linear gain
    assert v["RR@10"] == pytest.approx(0.5)

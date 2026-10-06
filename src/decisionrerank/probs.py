"""From raw logits to relevance probabilities and ranking scores.

Laya conditions (RQ2): ``raw`` divides the option logits by T=1; ``shipped`` by the checkpoint's
temperatures (noul 1.98, score 1.25); ``fitted`` by temperatures from Laya's own
``fit_temperature_map``, cross-fitted: a DL20 pair uses temperatures fitted on every judged DL19
pair, and vice versa. The noul target is rel >= 2 (the question asks whether the passage answers
the query); the score target is the grade itself.

Ranking rules for the score question: ``lin`` = E[y] (Bayes-optimal for linear-gain nDCG),
``exp`` = E[2^y - 1] (exponential gain), ``ge2`` = P(y >= 2). Temperature never changes the noul
ranking (P(yes) is monotone in z1 - z0); it can change all three score rankings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data

CONDITIONS = ("raw", "shipped", "fitted")
SCORE_RULES = ("lin", "exp", "ge2")
NOUL_COLS = ["noul_z0", "noul_z1"]
SCORE_COLS = [f"score_z{j}" for j in range(4)]


def softmax(z: np.ndarray, t: float | np.ndarray = 1.0) -> np.ndarray:
    z = np.asarray(z, np.float64) / np.asarray(t, np.float64).reshape(-1, 1)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def shipped_temperatures() -> dict:
    """{'noul': T, 'score': T} as the checkpoint's decoder looks them up (read from its config)."""
    import json

    from huggingface_hub import hf_hub_download
    from laya.common import QTYPES, clamp_temperature, temp_bucket

    from .scorers import LAYA_ID

    cfg = json.load(open(hf_hub_download(LAYA_ID, "rl_agent_config.json")))  # the file Agent reads
    by_opt = {k: clamp_temperature(v) for k, v in cfg.get("temperature_by_options", {}).items()}
    temp = [clamp_temperature(t) for t in cfg["temperature"]]
    return {q: by_opt.get(temp_bucket(QTYPES[q], k), temp[QTYPES[q]]) for q, k in (("noul", 2), ("score", 4))}


def fit_temperatures(judged: pd.DataFrame) -> dict:
    """Laya's own fitter on judged pairs (needs noul_z*, score_z*, rel): {'noul': T, 'score': T}."""
    from laya.calibrate import fit_temperature_map
    from laya.common import QTYPES, temp_bucket

    rel = judged["rel"].to_numpy().astype(int)
    eye2, eye4 = np.eye(2), np.eye(4)
    recs = [(QTYPES["noul"], z, eye2[int(r >= data.REL_MIN)], 2)
            for z, r in zip(judged[NOUL_COLS].to_numpy(), rel)]
    recs += [(QTYPES["score"], z, eye4[r], 4) for z, r in zip(judged[SCORE_COLS].to_numpy(), rel)]
    fit = fit_temperature_map(recs)
    by_opt, temp = fit["temperature_by_options"], fit["temperature"]
    return {q: by_opt.get(temp_bucket(QTYPES[q], k), temp[QTYPES[q]]) for q, k in (("noul", 2), ("score", 4))}


def laya_temperatures(run: pd.DataFrame) -> pd.DataFrame:
    """Per-pair temperatures for each condition: columns {cond}_noul, {cond}_score."""
    shipped = shipped_temperatures()
    out = pd.DataFrame(index=run.index)
    for q in ("noul", "score"):
        out[f"raw_{q}"] = 1.0
        out[f"shipped_{q}"] = shipped[q]
        out[f"fitted_{q}"] = np.nan
    judged = run[run["rel"].notna()]
    for test, train in (("dl19", "dl20"), ("dl20", "dl19")):
        fitted = fit_temperatures(judged[judged.qset == train])
        for q in ("noul", "score"):
            out.loc[run.qset == test, f"fitted_{q}"] = fitted[q]
    return out


def laya_frame(run: pd.DataFrame) -> pd.DataFrame:
    """Probabilities and ranking scores of every Laya condition, one row per pair.

    noul_p_{cond}: P(yes); score_p{y}_{cond}: P(y); score_{rule}_{cond}: ranking scores.
    """
    temps = laya_temperatures(run)
    out = run[data.KEY + ["rel", "in_pool"]].copy()
    gains_lin, gains_exp = np.arange(4.0), 2.0 ** np.arange(4) - 1
    for c in CONDITIONS:
        out[f"noul_p_{c}"] = softmax(run[NOUL_COLS].to_numpy(), temps[f"{c}_noul"].to_numpy())[:, 1]
        p = softmax(run[SCORE_COLS].to_numpy(), temps[f"{c}_score"].to_numpy())
        for y in range(4):
            out[f"score_p{y}_{c}"] = p[:, y]
        out[f"score_lin_{c}"] = p @ gains_lin
        out[f"score_exp_{c}"] = p @ gains_exp
        out[f"score_ge2_{c}"] = p[:, 2] + p[:, 3]
    out.attrs["temperatures"] = {c: {q: sorted(set(temps[f"{c}_{q}"].round(4)))
                                     for q in ("noul", "score")} for c in CONDITIONS}
    return out


def systems() -> tuple[pd.DataFrame, dict]:
    """One frame over all pairs with a probability or ranking score per system, plus temperatures.

    Columns: bm25 and bm25_rank (pool only), monot5_p, qwen3_p, and laya_frame's columns.
    """
    pairs = data.pairs()[data.KEY + ["rel", "in_pool"]]
    run = pairs.merge(data.load_run("laya"), on=data.KEY, validate="1:1")
    lay = laya_frame(run)
    t5 = data.load_run("monot5")
    t5["monot5_p"] = softmax(t5[["z_false", "z_true"]].to_numpy())[:, 1]
    qw = data.load_run("qwen3")
    qw["qwen3_p"] = softmax(qw[["z_no", "z_yes"]].to_numpy())[:, 1]
    bm = data.candidates()[data.KEY + ["bm25", "bm25_rank"]]
    out = (lay.merge(t5[data.KEY + ["monot5_p"]], on=data.KEY, validate="1:1")
              .merge(qw[data.KEY + ["qwen3_p"]], on=data.KEY, validate="1:1")
              .merge(bm, on=data.KEY, how="left", validate="1:1"))
    return out, lay.attrs["temperatures"]

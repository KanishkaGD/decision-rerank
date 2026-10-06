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
    for test, train in CROSS:
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
    out["laya_noul_z"] = (run["noul_z1"] - run["noul_z0"]).to_numpy()
    gains_lin, gains_exp = np.arange(4.0), 2.0 ** np.arange(4) - 1
    for c in CONDITIONS:
        out[f"noul_p_{c}"] = softmax(run[NOUL_COLS].to_numpy(), temps[f"{c}_noul"].to_numpy())[:, 1]
        p = softmax(run[SCORE_COLS].to_numpy(), temps[f"{c}_score"].to_numpy())
        for y in range(4):
            out[f"score_p{y}_{c}"] = p[:, y]
        out[f"score_lin_{c}"] = p @ gains_lin
        out[f"score_exp_{c}"] = p @ gains_exp
        out[f"score_ge2_{c}"] = p[:, 2] + p[:, 3]
    # keyed by the collection a temperature is applied to (fitted ones come from the other collection)
    out.attrs["temperatures"] = {c: {q: {qs: round(float(temps.loc[run.qset == qs, f"{c}_{q}"].iloc[0]), 4)
                                         for qs in data.QSETS} for q in ("noul", "score")} for c in CONDITIONS}
    return out


CROSS = (("dl19", "dl20"), ("dl20", "dl19"))  # (apply to, fit on)


def fit_binary_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """Unclamped NLL-optimal T for sigmoid(z / T) (Laya's own fitter clamps T to [0.5, 5])."""
    from scipy.optimize import minimize_scalar

    def nll(lt):
        p = np.clip(1 / (1 + np.exp(-z / np.exp(lt))), 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    return float(np.exp(minimize_scalar(nll, bounds=(np.log(0.05), np.log(100)), method="bounded").x))


def crossfit_binary(frame: pd.DataFrame, z: np.ndarray) -> tuple[np.ndarray, dict]:
    """sigmoid(z / T) with T fitted on the other collection's judged pairs (target rel >= 2)."""
    out, temps = np.empty(len(frame)), {}
    for test, train in CROSS:
        m = (frame.qset == train).to_numpy() & frame.rel.notna().to_numpy()
        t = fit_binary_temperature(z[m], (frame.rel.to_numpy()[m] >= data.REL_MIN).astype(float))
        sel = (frame.qset == test).to_numpy()
        out[sel] = 1 / (1 + np.exp(-z[sel] / t))
        temps[test] = round(t, 4)
    return out, temps


def logit(p: np.ndarray, eps: float = 0.005) -> np.ndarray:
    """Jev returns probabilities rounded to 0.01; clip 0 and 1 to half a step before the logit."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


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
    t5["monot5_z"] = t5.z_true - t5.z_false
    qw["qwen3_z"] = qw.z_yes - qw.z_no
    out = (lay.merge(t5[data.KEY + ["monot5_p", "monot5_z"]], on=data.KEY, validate="1:1")
              .merge(qw[data.KEY + ["qwen3_p", "qwen3_z"]], on=data.KEY, validate="1:1")
              .merge(bm, on=data.KEY, how="left", validate="1:1"))
    temps = dict(lay.attrs["temperatures"])
    zs = {"monot5": out.monot5_z.to_numpy(), "qwen3": out.qwen3_z.to_numpy(),
          "laya_noul": out.laya_noul_z.to_numpy()}
    if (data.RUNS / "jev.parquet").exists():
        out = out.merge(jev_frame(), on=data.KEY, how="left", validate="1:1")
        if out.jev_noul_p.notna().all():
            zs["jev_noul"] = logit(out.jev_noul_p.to_numpy())
    if (data.RUNS / f"{KEV}.parquet").exists():
        kv, temps["kev_shipped"] = kev_frame()
        out = out.merge(kv, on=data.KEY, how="left", validate="1:1")
        if out.kev_noul_p.notna().all():
            zs["kev_noul"] = out.kev_noul_z.to_numpy()
    for name, z in zs.items():  # one cross-fitted temperature per model, same fitter for all
        out[f"{name}_xfit"], temps[f"{name}_xfit"] = crossfit_binary(out, z)
    return out, temps


def jev_frame(jv: pd.DataFrame | None = None) -> pd.DataFrame:
    """Jev columns: jev_noul_p, jev_score_p{y}, ranking scores jev_{lin,exp,ge2} and jev_noul_lin.

    Jev rounds probabilities to two decimals. jev_noul_lin ranks by P(yes) and breaks its ties by
    E[y] from the same request (the added term is below the 0.01 rounding step).
    """
    jv = pd.read_parquet(data.RUNS / "jev.parquet") if jv is None else jv.copy()
    P = jv[[f"jev_score_p{y}" for y in range(4)]].to_numpy(float)
    jv["jev_lin"] = P @ np.arange(4.0)
    jv["jev_exp"] = P @ (2.0 ** np.arange(4) - 1)
    jv["jev_ge2"] = P[:, 2] + P[:, 3]
    jv["jev_noul_lin"] = jv["jev_noul_p"] + 1e-3 * jv["jev_lin"] / 3
    return jv.drop(columns=["jev_model", "jev_cost"])


KEV = "kev-0.8b"


def kev_frame() -> tuple[pd.DataFrame, float]:
    """Kev columns from its raw (T = 1) probabilities: kev_noul_p (as shipped: the checkpoint's temperature),
    kev_noul_p_raw, kev_noul_z (raw log-odds), kev_score_p{y} (shipped) and ranking scores kev_{lin,exp,ge2}."""
    import json

    kv = pd.read_parquet(data.RUNS / f"{KEV}.parquet")
    t = float(json.loads((data.RUNS / f"{KEV}.json").read_text())["shipped_temperature"])
    pr = np.clip(kv["kev_noul_p_raw"].to_numpy(float), 1e-12, 1 - 1e-12)
    kv["kev_noul_z"] = np.log(pr / (1 - pr))
    kv["kev_noul_p"] = 1 / (1 + np.exp(-kv["kev_noul_z"] / t))
    lp = np.log(np.clip(kv[[f"kev_score_p{y}_raw" for y in range(4)]].to_numpy(float), 1e-12, 1))
    P = softmax(lp, t)
    for y in range(4):
        kv[f"kev_score_p{y}"] = P[:, y]
    kv["kev_lin"] = P @ np.arange(4.0)
    kv["kev_exp"] = P @ (2.0 ** np.arange(4) - 1)
    kv["kev_ge2"] = P[:, 2] + P[:, 3]
    return kv, round(t, 4)

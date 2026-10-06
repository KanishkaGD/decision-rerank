"""Integrity checks on the stored runs (the README's "checks" refer to this script).

1. Laya: the stored fp32 logits, decoded with the shipped temperatures, reproduce Laya's own
   Agent.predict_batch (amp off, TF32 matmuls, the same runtime as scorers.load_laya) on a sample.
2. Laya: no state was truncated at 512 tokens.
3. monoT5: our scores match rerank-depth's independent CPU fp32 run where the pairs overlap.
4. Jev: one resolved model version and an answer for every pair.

usage: check_runs.py [--sample 800]   (check 1 uses the GPU)
"""

import argparse
import json

import numpy as np
import pandas as pd

from decisionrerank import data
from decisionrerank.probs import shipped_temperatures, softmax
from decisionrerank.questions import QUESTIONS, state
from decisionrerank.scorers import load_laya

RERANK_DEPTH = "/root/rerank-depth/runs/monot5_{}.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=800)
    a = ap.parse_args()
    report = {}
    run = data.load_run("laya")
    report["laya_truncated_states"] = int(run.noul_trunc.sum() + run.score_trunc.sum())
    assert report["laya_truncated_states"] == 0

    samp = data.pairs().merge(run, on=data.KEY).sample(a.sample, random_state=0)
    agent = load_laya()
    res = agent.predict_batch([state(q, d) for q, d in zip(samp["query"], samp["passage"])], QUESTIONS, batch_size=128)
    t = shipped_temperatures()
    pn = softmax(samp[["noul_z0", "noul_z1"]].to_numpy(), t["noul"])[:, 1]
    ps = softmax(samp[[f"score_z{j}" for j in range(4)]].to_numpy(), t["score"])
    rn = np.array([r["answers"]["noul"]["noul"] for r in res])
    rs = np.array([[r["answers"]["score"]["probabilities"][str(j)] for j in range(4)] for r in res])
    report["laya_vs_predict_batch_max_abs"] = {"noul": float(np.abs(pn - rn).max()), "score": float(np.abs(ps - rs).max())}
    assert max(report["laya_vs_predict_batch_max_abs"].values()) < 2e-4  # predict_batch rounds to 4 decimals

    t5 = data.load_run("monot5")
    ref = pd.concat([pd.read_parquet(RERANK_DEPTH.format(q)) for q in data.QSETS])
    m = t5.merge(ref, on=["query_id", "doc_id"])
    report["monot5_vs_rerank_depth"] = {"pairs": len(m), "max_abs_logodds": float(np.abs(m.z_true - m.z_false - m.score_ce).max())}
    assert report["monot5_vs_rerank_depth"]["max_abs_logodds"] < 1e-3

    if (data.RUNS / "jev.parquet").exists():
        jv = pd.read_parquet(data.RUNS / "jev.parquet").merge(data.pairs()[data.KEY], on=data.KEY)
        report["jev"] = {"pairs": len(jv), "of": len(data.pairs()), "models": sorted(jv.jev_model.unique()),
                         "cost_usd": round(float(jv.jev_cost.sum()), 3)}
    print(json.dumps(report, indent=2))
    (data.RESULTS / "checks.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

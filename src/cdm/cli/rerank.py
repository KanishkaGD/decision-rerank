import gzip
import json
import os
import random
import time

import click
import numpy as np

from cdm.config import FIRST_STAGE_DEPTH, OUTPUTS_FOLDER, SEED
from cdm.datasets import DATASETS, get_dataset
from cdm.evaluation import calibration, evaluate, judged_pairs
from cdm.phrasings import NONE, PHRASINGS, get_phrasing
from cdm.provenance import write_manifest
from cdm.rerankers import SYSTEMS, load_reranker
from cdm.results_csv import result_exists, save_result
from cdm.runs import first_stage_name, read_candidates, read_qrels, run_dir, write_run


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


@click.command("rerank")
@click.argument("system", type=click.Choice(list(SYSTEMS)))
@click.argument("dataset", type=click.Choice(list(DATASETS)))
@click.option("--first-stage", "first_stage", default=first_stage_name("bm25", FIRST_STAGE_DEPTH), show_default=True,
              help="Which cached candidate set to rerank (a directory under runs/<dataset>/).")
@click.option("--depth", type=int, default=None, help="Rerank only the top N candidates (default: all).")
@click.option("--phrasings", default=None,
              help="Comma-separated phrasing ids, or 'all' (src/cdm/phrasings.py). Default: all for systems that use "
                   "a phrasing, 'none' otherwise.")
@click.option("--variant", default="", help="Label for a non-default configuration, kept in results.csv and the "
                                            "output path (e.g. a fine-tuned checkpoint).")
@click.option("--batch-size", type=int, default=None, help="Override the system's batch size.")
@click.option("--max-queries", type=int, default=None, help="Only the first N queries (quick checks).")
@click.option("--no-save", is_flag=True, help="Print metrics only; write nothing.")
@click.option("--overwrite", is_flag=True, help="Recompute even if results.csv already has this configuration.")
@click.option("--seed", type=int, default=SEED, show_default=True)
def rerank(system, dataset, first_stage, depth, phrasings, variant, batch_size, max_queries, no_save, overwrite, seed):
    """
    Rerank the cached first-stage candidates of DATASET with SYSTEM. Stores the raw score (and probability, if the
    system has one) of every (query, passage, phrasing) under outputs/, and the metrics in results/results.csv:
    nDCG@10, R@10, R@100, RR@10, AP, ECE and Brier on the judged pairs, per-query latency (p50, p95) and the share of
    truncated inputs.
    """
    _seed_everything(seed)
    ds = get_dataset(dataset)
    cache = run_dir(dataset, first_stage)
    if not os.path.exists(os.path.join(cache, "candidates.jsonl.gz")):
        raise click.ClickException(f"No cached candidates in {cache}: run `cdm first-stage {dataset}` first.")
    queries = list(read_candidates(os.path.join(cache, "candidates.jsonl.gz")))
    if max_queries:
        queries = queries[:max_queries]
    qrels = read_qrels(os.path.join(cache, "qrels.tsv"))
    subset = f"max{max_queries}" if max_queries else ""

    reranker = load_reranker(system, batch_size=batch_size)
    if phrasings is None:
        phrasing_ids = [p for p in PHRASINGS if p != NONE.id] if reranker.uses_phrasing else [NONE.id]
    elif phrasings == "all":
        phrasing_ids = [p for p in PHRASINGS if p != NONE.id]
    else:
        phrasing_ids = phrasings.split(",")
    if not reranker.uses_phrasing and phrasing_ids != [NONE.id]:
        raise click.ClickException(f"{system} takes no phrasing; use --phrasings none.")

    for phrasing_id in phrasing_ids:
        phrasing = get_phrasing(phrasing_id)
        if reranker.uses_phrasing and not phrasing.frozen and ds.split == "test" and not max_queries:
            click.echo(f"WARNING: phrasing '{phrasing_id}' is not frozen yet, but {dataset} is a test set. The plan "
                       f"selects phrasings on dev only (WP3).", err=True)
        key = {"job": "rerank", "system": system, "model": reranker.model_id, "dataset": dataset,
               "first_stage": first_stage if depth is None else f"{first_stage}@{depth}", "phrasing": phrasing_id,
               "variant": variant, "subset": subset}
        if not no_save and not overwrite and result_exists(key):
            click.echo(f"Skipping {key}: already in results.csv (--overwrite to redo).")
            continue

        run, scored, latencies, truncated, n_pairs = {}, [], [], 0, 0
        out = os.path.join(OUTPUTS_FOLDER, dataset, key["first_stage"], f"{system}{'-' + variant if variant else ''}",
                           phrasing_id, *([subset] if subset else []))
        if not no_save:
            os.makedirs(out, exist_ok=True)
        scores_file = gzip.open(os.path.join(out, "scores.jsonl.gz"), "wt") if not no_save else None
        responses_file = None
        if queries and reranker.warmup:  # warm-up (CUDA kernels, caches), so the first query doesn't inflate latency
            reranker.score(queries[0].query, [c.text for c in queries[0].candidates[: depth or None]], phrasing)
        reranker.reset_stats()
        try:
            for i, q in enumerate(queries, start=1):
                cands = q.candidates[:depth] if depth else q.candidates
                start = time.perf_counter()
                result = reranker.score(q.query, [c.text for c in cands], phrasing)
                latencies.append((time.perf_counter() - start) * 1000)
                truncated += reranker.count_truncated(q.query, [c.text for c in cands])
                n_pairs += len(cands)
                run[q.qid] = {c.docid: s for c, s in zip(cands, result.scores)}
                for j, c in enumerate(cands):
                    prob = result.probs[j] if result.probs is not None else None
                    if prob is not None:
                        scored.append((q.qid, c.docid, prob))
                    if scores_file:
                        scores_file.write(json.dumps({"qid": q.qid, "docid": c.docid, "first_stage_rank": c.rank,
                                                      "score": result.scores[j], "prob": prob}) + "\n")
                if result.raw is not None and not no_save:
                    if responses_file is None:
                        responses_file = gzip.open(os.path.join(out, "responses.jsonl.gz"), "wt")
                    for record in result.raw:
                        responses_file.write(json.dumps({"qid": q.qid, **record}) + "\n")
                if i % 50 == 0 or i == len(queries):
                    click.echo(f"  {phrasing_id}: {i}/{len(queries)} queries")
        finally:
            for f in (scores_file, responses_file):
                if f:
                    f.close()

        means, per_query = evaluate(run, qrels, ds.rel_threshold)
        calib = calibration(*judged_pairs(scored, qrels, ds.rel_threshold)) if scored else {}
        lat = np.asarray(latencies)
        stats = {**means, **calib, "latency_p50_ms": float(np.percentile(lat, 50)),
                 "latency_p95_ms": float(np.percentile(lat, 95)), "truncated_frac": truncated / max(n_pairs, 1),
                 "n_pairs": n_pairs, **reranker.run_stats()}
        if "api_cost_usd" in stats:
            stats["api_cost_usd_per_1k_queries"] = stats["api_cost_usd"] / max(len(queries), 1) * 1000
        click.echo(f"{system} on {dataset} [{phrasing_id}]: " + ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()))
        if no_save:
            continue

        write_run(os.path.join(out, "run.trec.gz"), run, tag=f"{system}-{phrasing_id}")
        per_query.to_csv(os.path.join(out, "perquery.csv"))
        manifest = write_manifest(out, dataset=dataset, first_stage=first_stage, depth=depth, phrasing=phrasing_id,
                                  question=phrasing.question, variant=variant, seed=seed, max_queries=max_queries,
                                  reranker=reranker.describe(), metrics=stats)
        save_result(key, {**stats, "prob_kind": reranker.prob_kind, "rel_threshold": ds.rel_threshold,
                          "split": ds.split, "device": manifest["hardware"].get("gpu", "cpu"), "seed": seed,
                          "code": manifest["code"].get("describe", manifest["code"].get("commit", "")),
                          "output": out, "slurm_job_id": os.environ.get("SLURM_JOB_ID", "")})

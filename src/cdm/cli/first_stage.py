import os

import click

from cdm.config import FIRST_STAGE_DEPTH
from cdm.datasets import DATASETS, get_dataset
from cdm.evaluation import evaluate
from cdm.provenance import write_manifest
from cdm.results_csv import save_result
from cdm.runs import first_stage_name, run_dir, write_candidates, write_qrels, write_run


@click.command("first-stage")
@click.argument("dataset", type=click.Choice(list(DATASETS)))
@click.option(
    "--retriever",
    type=click.Choice(["official", "bm25"]),
    default=None,
    help="'official': the track's rerank pool (needs `cdm prefetch`); 'bm25': our own BM25 top-N. Default: "
    "official where the dataset has a pool, else bm25. A dense retriever is still to be added.",
)
@click.option(
    "--depth",
    type=int,
    default=FIRST_STAGE_DEPTH,
    show_default=True,
    help="Candidates per query (bm25 only; an official pool keeps all its candidates).",
)
@click.option("--threads", type=int, default=8, show_default=True, help="Search threads (Slurm: --cpus-per-task).")
@click.option(
    "--max-queries",
    type=int,
    default=None,
    help="Only the first N queries (quick checks; not saved to " "results.csv unless --save-subset).",
)
@click.option("--save-subset", is_flag=True, help="Record a --max-queries run in results.csv too.")
@click.option("--overwrite", is_flag=True, help="Replace an existing cache.")
def first_stage(dataset, retriever, depth, threads, max_queries, save_subset, overwrite):
    """
    Retrieve and cache the first-stage candidates of DATASET under runs/<dataset>/<retriever>-d<depth>/: the
    candidates with their texts, a TREC run file, the qrels and a manifest. Every reranker reads this cache.
    """
    from cdm.first_stage import official_pool, retrieve_bm25

    ds = get_dataset(dataset)
    retriever = retriever or ("official" if ds.pool else "bm25")
    if retriever == "official":
        if not ds.pool:
            raise click.ClickException(f"{dataset} has no official rerank pool; use --retriever bm25.")
        depth = ds.pool.depth
    name = first_stage_name(retriever, depth)
    out = run_dir(dataset, name) if not max_queries else run_dir(dataset, f"{name}-max{max_queries}")
    if os.path.exists(os.path.join(out, "candidates.jsonl.gz")) and not overwrite:
        click.echo(f"{out} already exists; pass --overwrite to redo it.")
        return
    os.makedirs(out, exist_ok=True)

    if retriever == "official":
        click.echo(f"Reading the official pool of {dataset} ({ds.pool.url}), ordered by BM25 on {ds.index}...")
        candidates, qrels, params = official_pool(ds, threads=threads, max_queries=max_queries)
    else:
        click.echo(f"Retrieving top-{depth} with {retriever} for {dataset} ({ds.index}, {ds.topics})...")
        candidates, qrels, params = retrieve_bm25(ds, depth=depth, threads=threads, max_queries=max_queries)
    write_candidates(os.path.join(out, "candidates.jsonl.gz"), candidates)
    run = {q.qid: {c.docid: c.score for c in q.candidates} for q in candidates}
    write_run(os.path.join(out, "run.trec.gz"), run, tag=name)
    write_qrels(os.path.join(out, "qrels.tsv"), qrels)

    means, per_query = evaluate(run, qrels, ds.rel_threshold)
    per_query.to_csv(os.path.join(out, "perquery.csv"))
    write_manifest(out, dataset=dataset, **params)
    click.echo(
        f"{dataset} {name}: "
        + ", ".join(f"{k}={v:.4f}" for k, v in means.items() if k != "n_queries")
        + f" over {means['n_queries']} queries -> {out}"
    )

    if not max_queries or save_subset:
        save_result(
            {
                "job": "first-stage",
                "system": retriever,
                "model": ds.index,
                "dataset": dataset,
                "first_stage": "",
                "phrasing": "",
                "variant": f"d{depth}",
                "subset": f"max{max_queries}" if max_queries else "",
            },
            {
                **means,
                "rel_threshold": ds.rel_threshold,
                "split": ds.split,
                "output": out,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            },
        )

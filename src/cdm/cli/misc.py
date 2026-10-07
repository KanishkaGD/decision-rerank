import os

import click

from cdm.datasets import DATASETS
from cdm.rerankers import SYSTEMS


@click.command("info")
def info():
    """Code revision, package versions, paths and hardware -- a quick check that the environment is complete."""
    from cdm import config
    from cdm.provenance import code_revision, hardware, package_versions

    click.echo(f"code:     {code_revision()}")
    for name, version in package_versions().items():
        click.echo(f"{name + ':':24}{version}")
    for name in ["CDM_ROOT", "RUNS_FOLDER", "OUTPUTS_FOLDER", "RESULTS_FOLDER"]:
        click.echo(f"{name + ':':24}{getattr(config, name)}")
    for var in ["HF_HOME", "PYSERINI_CACHE", "HF_HUB_OFFLINE", "SLURM_JOB_ID"]:
        click.echo(f"{var + ':':24}{os.environ.get(var, '')}")
    click.echo(f"hardware: {hardware()}")


@click.command("datasets")
def datasets():
    """List the registered datasets (src/cdm/datasets.py)."""
    for d in DATASETS.values():
        click.echo(f"{d.name:12} {d.split:5} {d.group:8} rel>={d.rel_threshold}  index={d.index}  topics={d.topics}")


@click.command("systems")
def systems():
    """List the registered rerankers (src/cdm/rerankers/__init__.py)."""
    for s in SYSTEMS.values():
        click.echo(f"{s.name:12} {s.kind:14} {s.model_id}  {s.params or ''}")


@click.command("prefetch")
@click.option("--datasets", "dataset_list", default=",".join(DATASETS), show_default=True)
@click.option("--systems", "system_list", default=",".join(SYSTEMS), show_default=True)
def prefetch(dataset_list, system_list):
    """
    Download Pyserini indexes, topics and qrels, and the rerankers' Hugging Face models, into the caches (PYSERINI_CACHE,
    HF_HOME). Run on the login node (slurm/prefetch.sh): compute nodes may have no internet.
    """
    from huggingface_hub import snapshot_download
    from pyserini.search.lucene import LuceneSearcher

    from cdm.first_stage import load_topics_and_qrels

    for name in filter(None, dataset_list.split(",")):
        ds = DATASETS[name]
        click.echo(f"[{name}] index {ds.index}")
        LuceneSearcher.from_prebuilt_index(ds.index)
        queries, qrels = load_topics_and_qrels(ds)
        click.echo(f"[{name}] {len(queries)} judged queries, {sum(map(len, qrels.values()))} judgments")
    for name in filter(None, system_list.split(",")):
        spec = SYSTEMS[name]
        if not spec.hf:
            click.echo(f"[{name}] skipped: {spec.model_id} is not a Hugging Face model (yet)")
            continue
        path = snapshot_download(spec.model_id, revision=spec.revision)
        click.echo(f"[{name}] {spec.model_id} -> {path}")


@click.command("results")
@click.option("--job", default="rerank", show_default=True)
@click.option("--metric", default="nDCG@10", show_default=True)
@click.option("--subsets", is_flag=True, help="Include quick --max-queries runs.")
def results(job, metric, subsets):
    """Print results.csv as a system x dataset table of one metric."""
    import pandas as pd

    from cdm.results_csv import load_results

    df = load_results()
    if df.empty:
        click.echo("results.csv is empty.")
        return
    df = df[df["job"] == job]
    if not subsets:
        df = df[df["subset"] == ""]
    if metric not in df.columns:
        raise click.ClickException(f"No column {metric}. Columns: {', '.join(df.columns)}")
    df = df.assign(
        value=pd.to_numeric(df[metric], errors="coerce"),
        row=df["system"]
        + df["variant"].map(lambda v: f" [{v}]" if v else "")
        + df["phrasing"].map(lambda p: f" ({p})" if p not in ("", "none") else ""),
    )
    table = df.pivot_table(index="row", columns="dataset", values="value", aggfunc="first")
    click.echo(table.round(4).to_string())

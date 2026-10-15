import os
from typing import Optional

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
        click.echo(
            f"{d.name:12} {d.split:5} {d.group:8} rel>={d.rel_threshold}  index={d.index}  topics={d.topics}  "
            f"first stage={d.default_first_stage}"
        )


@click.command("systems")
def systems():
    """List the registered rerankers (src/cdm/rerankers/__init__.py)."""
    for s in SYSTEMS.values():
        click.echo(f"{s.name:22} {s.kind:15} {s.model_id}  {s.params or ''}")


def _download(repo: str, revision: Optional[str], tries: int = 5) -> str:
    """
    snapshot_download with retries: large downloads to network file systems fail now and then with transient
    errors (e.g. "Resource temporarily unavailable" from hf_xet).
    Files already complete are not fetched again, so a retry resumes.
    """
    import time

    from huggingface_hub import snapshot_download

    for attempt in range(1, tries + 1):
        try:
            return snapshot_download(repo, revision=revision)
        except Exception as e:  # noqa: BLE001 -- any download error is worth a retry; the last one is raised
            if attempt == tries:
                raise
            wait = 30 * attempt
            click.echo(
                f"  {repo}: download failed ({type(e).__name__}: {str(e)[:200]}); retry {attempt}/{tries - 1} "
                f"in {wait} s",
                err=True,
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


@click.command("prefetch")
@click.option("--datasets", "dataset_list", default=",".join(DATASETS), show_default=True)
@click.option("--systems", "system_list", default=",".join(SYSTEMS), show_default=True)
@click.option(
    "--training-data",
    "training_list",
    default="",
    help="Fine-tuning sources to download (src/cdm/finetune_data.py TRAINING_SOURCES), comma-separated; default none.",
)
def prefetch(dataset_list, system_list, training_list):
    """
    Download Pyserini indexes, topics and qrels, official rerank pools (data/pools/), and the rerankers' Hugging Face
    models, into the caches (PYSERINI_CACHE, HF_HOME). Run on a machine with internet access: the
    runs themselves work offline (HF_HUB_OFFLINE=1).
    """
    from pyserini.search.lucene import LuceneSearcher

    from cdm.first_stage import download_pool, load_topics_and_qrels, read_pool

    for name in filter(None, dataset_list.split(",")):
        ds = DATASETS[name]
        click.echo(f"[{name}] index {ds.index}")
        LuceneSearcher.from_prebuilt_index(ds.index)
        queries, qrels = load_topics_and_qrels(ds)
        click.echo(f"[{name}] {len(queries)} judged queries, {sum(map(len, qrels.values()))} judgments")
        if ds.pool:
            path = download_pool(ds)
            pooled = {}
            for qid, docid, _ in read_pool(path):
                if qid in queries:
                    pooled.setdefault(qid, set()).add(docid)
            click.echo(
                f"[{name}] official pool {path}: {len(pooled)} of {len(queries)} judged queries, "
                f"{sum(map(len, pooled.values()))} pairs"
            )
    for name in filter(None, training_list.split(",")):
        from cdm.finetune_data import TRAINING_SOURCES, source_path

        src = TRAINING_SOURCES[name]
        for attempt in range(1, 6):
            try:
                click.echo(f"[{name}] {src.repo}/{src.filename} -> {source_path(src, local_only=False)}")
                break
            except Exception as e:  # noqa: BLE001 -- transient download errors, as in _download
                if attempt == 5:
                    raise
                click.echo(
                    f"  {name}: download failed ({type(e).__name__}: {str(e)[:200]}); retry {attempt}/4", err=True
                )
    downloaded = set()
    for name in filter(None, system_list.split(",")):
        spec = SYSTEMS[name]
        if not spec.hf:
            click.echo(f"[{name}] skipped: {spec.model_id} is not a Hugging Face model")
            continue
        repos = [(spec.model_id, spec.revision), *spec.extra_hf]
        if spec.params.get("backend") == "kev":  # a LoRA adapter: its base model is named in the checkpoint
            from kev.checkpoint import Checkpoint

            _download(spec.model_id, spec.revision)
            meta = Checkpoint(f"{spec.model_id}@{spec.revision}" if spec.revision else spec.model_id).meta
            repos.append((meta.base, meta.base_revision))
        for repo, revision in repos:
            if (repo, revision) not in downloaded:
                path = _download(repo, revision)
                downloaded.add((repo, revision))
                click.echo(f"[{name}] {repo} -> {path}")


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


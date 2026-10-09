import json
import os

import click

from cdm.datasets import DATASETS
from cdm.finetune_data import TRAINING_SOURCES

# Recommended mix (docs/notes/reranker-27b-plan.md): dense-retriever hard negatives and BM25 negatives (the
# candidate distribution we rerank), 1 + 15 per group.
DEFAULT_NEGATIVES = "tevatron-aug:8,tevatron-bm25:7"


@click.command("finetune-data")
@click.argument("name")
@click.option(
    "--negatives",
    default=DEFAULT_NEGATIVES,
    show_default=True,
    help=f"Negatives per group and source, 'source:k,...' ({', '.join(TRAINING_SOURCES)}). The first source also "
    "gives the queries and positives.",
)
@click.option("--queries", "n_queries", type=int, default=None, help="Number of training queries (default: all).")
@click.option(
    "--exclude",
    default=",".join(DATASETS),
    show_default=True,
    help="Datasets whose queries no training query may share (id, normalised text or bag of words).",
)
@click.option("--seed", type=int, default=42, show_default=True)
def finetune_data(name, negatives, n_queries, exclude, seed):
    """
    Build training groups NAME for `cdm finetune` (data/finetune/NAME/groups.jsonl.gz): MS MARCO train queries with
    one positive and hard negatives from the prefetched sources (`cdm prefetch --training-data ...`), minus every
    query shared with a test collection or msmarco-dev. CPU only.
    """
    from cdm.finetune_data import (
        TRAINING_SOURCES,
        LeakageFilter,
        build_groups,
        evaluation_queries,
        groups_path,
        parse_quota,
        source_path,
        write_groups,
    )
    from cdm.provenance import write_manifest

    quota = parse_quota(negatives)
    paths = {s: source_path(TRAINING_SOURCES[s]) for s in quota}
    excluded = [d for d in exclude.split(",") if d]
    leakage = LeakageFilter.from_queries(evaluation_queries(excluded))
    click.echo(f"Leakage filter: {len(leakage.qids)} query ids, {len(leakage.texts)} texts from {', '.join(excluded)}")
    groups, stats = build_groups(paths, quota, leakage, n_queries, seed)
    path = groups_path(name)
    write_groups(path, groups)
    n_words = [len(g["pos"]["text"].split()) for g in groups[:5000]]
    stats["mean_words_per_positive_first5000"] = sum(n_words) / max(len(n_words), 1)
    write_manifest(
        os.path.dirname(path),
        job="finetune-data",
        name=name,
        negatives=quota,
        n_queries=n_queries,
        exclude=excluded,
        seed=seed,
        sources={s: {**TRAINING_SOURCES[s].__dict__, "path": paths[s]} for s in quota},
        stats=stats,
    )
    click.echo(json.dumps({k: v for k, v in stats.items() if k != "removed_examples"}, indent=2))
    click.echo(f"Wrote {len(groups)} groups to {path}")


@click.command("finetune")
@click.argument("system")
@click.argument("data")
@click.option("--loss", type=click.Choice(["listwise", "pointwise", "both"]), default="listwise", show_default=True)
@click.option("--group-size", type=int, default=16, show_default=True, help="1 positive + (group size - 1) negatives.")
@click.option("--max-length", type=int, default=1024, show_default=True, help="Tokens per input (= the reader's).")
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--warmup-ratio", type=float, default=0.03, show_default=True)
@click.option("--scheduler", default="linear", show_default=True, help="transformers scheduler name.")
@click.option("--epochs", type=float, default=1.0, show_default=True)
@click.option("--groups-per-step", type=int, default=64, show_default=True, help="Effective batch in query groups.")
@click.option("--micro-groups", type=int, default=1, show_default=True, help="Query groups per forward pass.")
@click.option("--lora-r", type=int, default=32, show_default=True)
@click.option("--lora-alpha", type=int, default=64, show_default=True)
@click.option("--lora-dropout", type=float, default=0.05, show_default=True)
@click.option(
    "--dtype",
    type=click.Choice(["bfloat16", "float32"]),
    default="bfloat16",
    show_default=True,
    help="Frozen base weights (LoRA weights are fp32 either way; float32 for CPU tests).",
)
@click.option("--no-gradient-checkpointing", is_flag=True)
@click.option("--max-groups", type=int, default=None, help="Train on the first N groups only (smoke tests).")
@click.option("--max-steps", type=int, default=None, help="Stop after N optimizer steps (throughput probes).")
@click.option("--save-steps", type=int, default=200, show_default=True, help="0: no intermediate checkpoints.")
@click.option("--log-steps", type=int, default=10, show_default=True)
@click.option("--max-hours", type=float, default=None, help="Stop (with a checkpoint) after this long.")
@click.option("--out", default=None, help="Output directory (default: the system's adapter directory's parent).")
@click.option("--no-final", is_flag=True, help="Don't write <out>/final (probes).")
@click.option("--resume", is_flag=True, help="Continue from the newest checkpoint in the output directory.")
@click.option("--seed", type=int, default=42, show_default=True)
def finetune(system, data, out, resume, no_gradient_checkpointing, no_final, **hp):
    """
    LoRA reranker fine-tune of SYSTEM's base model on training groups DATA (`cdm finetune-data`), with Qwen3-Reranker's
    prompt and yes/no readout (src/cdm/finetune.py). SYSTEM is a registered reranker with an `adapter`
    (src/cdm/rerankers/__init__.py); the adapter is written where that entry expects it, so `cdm rerank SYSTEM ...`
    scores the result. Multi-GPU: run under torchrun (slurm/finetune.sbatch does).
    """
    from cdm.finetune import TrainConfig, train
    from cdm.finetune_data import groups_path
    from cdm.rerankers import get_system
    from cdm.rerankers.pointwise_lm import QWEN3_INSTRUCTION, QWEN3_SYSTEM

    spec = get_system(system)
    adapter = spec.params.get("adapter")
    if spec.kind != "qwen3-reranker" or not adapter:
        raise click.ClickException(f"{system} is not a reranker fine-tune of ours (a qwen3-reranker with an adapter).")
    if os.path.basename(adapter.rstrip("/")) != "final":
        raise click.ClickException(f"{system}: the registered adapter must be <out>/final, not {adapter}")
    path = data if data.endswith(".jsonl.gz") else groups_path(data)
    if not os.path.exists(path):
        raise click.ClickException(f"No training groups at {path}: build them with `cdm finetune-data {data}`.")
    cfg = TrainConfig(
        system=system,
        model_id=spec.model_id,
        revision=spec.revision,
        data=path,
        out=out or os.path.dirname(adapter.rstrip("/")),
        instruction=spec.params.get("instruction", QWEN3_INSTRUCTION),
        system_prompt=spec.params.get("system", QWEN3_SYSTEM),
        gradient_checkpointing=not no_gradient_checkpointing,
        write_final=not no_final,
        **hp,
    )
    result = train(cfg, resume=resume)
    click.echo(json.dumps(result))

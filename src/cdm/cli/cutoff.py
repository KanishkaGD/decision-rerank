import os

import click

DL = "dl19,dl20,dl21,dl22,dl23"
# Where a dataset's default cell is taken from the literature (bge-m3, Set-Encoder on DL19/20), the probabilities come
# from our efficiency run on the same candidates; the stored row keeps that variant.
FALLBACK_VARIANTS = ("", "a100")


@click.command("cutoff")
@click.option("--datasets", default=DL, show_default=True, help="Comma-separated; needs judged non-relevant passages.")
@click.option("--systems", default=None, help="Comma-separated; default: every system with stored probabilities.")
def cutoff(datasets, systems):
    """
    RQ2, Part A: keep every candidate with p >= tau (0.3, 0.5, 0.7) and score the set (precision, recall, F1, size),
    find the per-dataset oracle cutoff and the regret of tau = 0.5, and the F1 of fixed top-k sets. Reads the stored
    probabilities (scores.jsonl.gz) of the rerank rows in results.csv on each dataset's default candidates and writes
    one row per system and dataset (job = cutoff) to results.csv. No model runs.
    """
    from cdm.cutoff import cutoff_analysis, read_scores
    from cdm.datasets import get_dataset
    from cdm.provenance import code_revision
    from cdm.results_csv import RESULT_KEY_COLUMNS, load_results, save_result
    from cdm.runs import read_qrels, run_dir

    df = load_results()
    df = df[(df["job"] == "rerank") & (df["subset"] == "") & (df["output"] != "")]
    wanted = set(systems.split(",")) if systems else None
    revision = code_revision()
    code = revision.get("describe", revision.get("commit", ""))
    for dataset in datasets.split(","):
        ds = get_dataset(dataset)
        first_stage = ds.default_first_stage
        qrels = read_qrels(os.path.join(run_dir(dataset, first_stage), "qrels.tsv"))
        rows = df[(df["dataset"] == dataset) & (df["first_stage"] == first_stage)]
        # One analysis per system and question (S1Rank's and, for RQ2, Kanishka's "answer"): the phrasing is a key.
        for system, phrasing in sorted(set(zip(rows["system"], rows["phrasing"]))):
            if wanted is not None and system not in wanted:
                continue
            mine = rows[(rows["system"] == system) & (rows["phrasing"] == phrasing)]
            candidates = [mine[mine["variant"] == v] for v in FALLBACK_VARIANTS]
            source = next((c.iloc[0] for c in candidates if len(c)), None)
            if source is None:
                continue
            path = os.path.join(source["output"], "scores.jsonl.gz")
            if not os.path.exists(path):
                click.echo(f"  {system} {dataset}: no {path}, skipped", err=True)
                continue
            values = cutoff_analysis(read_scores(path, qrels, ds.rel_threshold))
            if not values:  # no probabilities (RankZephyr outputs a ranking only)
                continue
            key = {c: source[c] for c in RESULT_KEY_COLUMNS}
            key["job"] = "cutoff"
            meta = {"rel_threshold": ds.rel_threshold, "source_output": source["output"], "code": code}
            save_result(key, {**values, **meta})
            click.echo(
                f"{system:24} {phrasing:7} {dataset}{'/' + source['variant'] if source['variant'] else '':6} "
                f"F1@0.5={values['cut_F1@0.5']:.3f} oracle={values['cut_F1@oracle']:.3f} "
                f"(tau {values['cut_oracle_tau']:.3f}) size@0.5={values['cut_size@0.5']:.1f} "
                f"top10={values['topk_F1@10']:.3f}"
            )

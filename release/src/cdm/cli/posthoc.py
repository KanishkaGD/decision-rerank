import os

import click

# The trained rerankers whose native probabilities are scaled after the fact.
RERANKERS = "qwen3-reranker-0.6b,qwen3-reranker-8b,prism-0.8b,prism-4b,prism-9b,monot5-3b,bge-m3,set-encoder-large"
TEST = "dl19,dl20,dl21,dl22,dl23,trec-covid,nfcorpus,scifact,touche,dbpedia"
# Where a default cell comes from the literature, the probabilities come from our efficiency run on the same
# candidates (as in `cdm cutoff`); the stored row records it in source_variant.
FALLBACK_VARIANTS = ("", "a100")


@click.command("posthoc")
@click.option("--systems", default=RERANKERS, show_default=True, help="Comma-separated rerankers to scale.")
@click.option("--datasets", default=TEST, show_default=True, help="Comma-separated test collections.")
@click.option("--fit-dataset", default="msmarco-dev", show_default=True, help="Dev split the scaling is fitted on.")
@click.option("--fit-first-stage", default="bm25-d100-max1000", show_default=True, help="Its candidate set.")
def posthoc(systems, datasets, fit_dataset, fit_first_stage):
    """
    RQ2, after the fact: fit temperature and Platt scaling of each reranker's logit on the dev split (every candidate
    labelled, unjudged = non-relevant) and evaluate native, temperature-scaled and Platt-scaled probabilities on the
    test collections: ECE/Brier on judged pairs, ECE/Brier with unjudged = non-relevant, and the fixed cutoff of
    `cdm cutoff` on the collections that judge non-relevant passages. Reads stored outputs only; writes one row per
    system, dataset and method (job = posthoc, variant = method) to results.csv.
    """
    from cdm.cutoff import read_scores
    from cdm.datasets import get_dataset
    from cdm.posthoc import METHODS, evaluate_scaled, fit, scaled
    from cdm.provenance import code_revision
    from cdm.results_csv import RESULT_KEY_COLUMNS, load_results, save_result
    from cdm.runs import read_qrels, run_dir

    fit_ds = get_dataset(fit_dataset)
    if fit_ds.split == "test":
        raise click.ClickException(f"{fit_dataset} is a test split: fit on a dev split only (CLAUDE.md).")
    df = load_results()
    df = df[(df["job"] == "rerank") & (df["subset"] == "") & (df["output"] != "")]
    revision = code_revision()
    code = revision.get("describe", revision.get("commit", ""))
    dev_qrels = read_qrels(os.path.join(run_dir(fit_dataset, fit_first_stage), "qrels.tsv"))

    def source(system, dataset, first_stage):
        rows = df[(df["system"] == system) & (df["dataset"] == dataset) & (df["first_stage"] == first_stage)]
        for variant in FALLBACK_VARIANTS:
            match = rows[rows["variant"] == variant]
            if len(match) and os.path.exists(os.path.join(match.iloc[0]["output"], "scores.jsonl.gz")):
                return match.iloc[0]
        return None

    def save(src, dataset, method, values, extra):
        key = {c: src[c] for c in RESULT_KEY_COLUMNS}
        key.update(job="posthoc", dataset=dataset, variant=method)
        save_result(key, {**values, **extra, "source_variant": src["variant"], "source_output": src["output"],
                          "code": code})

    for system in systems.split(","):
        dev_src = source(system, fit_dataset, fit_first_stage)
        if dev_src is None:
            click.echo(f"{system}: no {fit_dataset} run on {fit_first_stage}; skipped", err=True)
            continue
        dev = read_scores(os.path.join(dev_src["output"], "scores.jsonl.gz"), dev_qrels, fit_ds.rel_threshold)
        params = fit(dev)
        n_pairs = sum(len(q.probs) for q in dev)
        n_rel = int(sum(q.relevant.sum() for q in dev))
        click.echo(f"{system}: fitted on {len(dev)} queries, {n_pairs} pairs, {n_rel} relevant: "
                   + ", ".join(f"{m} a={a:.3f} b={b:.3f}" for m, (a, b) in params.items()))
        for method in METHODS:
            a, b = params[method]
            extra = {"posthoc_a": a, "posthoc_b": b, "fit_dataset": fit_dataset, "fit_first_stage": fit_first_stage,
                     "fit_n_queries": len(dev), "fit_n_pairs": n_pairs, "fit_n_relevant": n_rel}
            # In-sample: how well the fit itself does on the dev split.
            save(dev_src, fit_dataset, method, evaluate_scaled(scaled(dev, a, b), with_cutoff=False), extra)
            for dataset in datasets.split(","):
                ds = get_dataset(dataset)
                if ds.split != "test":
                    continue
                src = source(system, dataset, ds.default_first_stage)
                if src is None:
                    continue
                qrels = read_qrels(os.path.join(run_dir(dataset, ds.default_first_stage), "qrels.tsv"))
                queries = read_scores(os.path.join(src["output"], "scores.jsonl.gz"), qrels, ds.rel_threshold)
                values = evaluate_scaled(scaled(queries, a, b), with_cutoff=ds.group == "trec-dl")
                save(src, dataset, method, values, extra)
                click.echo(f"  {method:11} {dataset:10} ECE={values.get('ECE', float('nan')):.3f} "
                           f"ECE_unj0={values['ECE_unjudged0']:.3f}"
                           + (f" F1@0.5={values['cut_F1@0.5']:.3f} regret={values['cut_regret']:.3f}"
                              if "cut_F1@0.5" in values else ""))

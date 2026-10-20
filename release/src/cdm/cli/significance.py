import click

TEST = "dl19,dl20,dl21,dl22,dl23,trec-covid,nfcorpus,scifact,touche,dbpedia"


@click.command("significance")
@click.option("--datasets", default=TEST, show_default=True, help="Comma-separated; Holm corrects across these.")
@click.option("--phrasing", default="answer", show_default=True, help="Question of the decision models (and base models).")
def significance(datasets, phrasing):
    """
    RQ1: paired t-tests over queries (nDCG@10 from the stored perquery.csv) for the triplet comparisons -- reranker
    fine-tune vs. the best decision fine-tune, that one vs. the base model, joint vs. isolated -- Holm-corrected across
    the datasets per comparison. Reads stored outputs only; writes one row per comparison and dataset (job =
    significance, system = the first system, variant = vs-<second system>, with "@<phrasing>" for any question but
    S1Rank's) to results.csv. Systems that ask no question (phrasing none) are compared with the decision and base
    models of the given question; the triplets' base rows are the paper's (cdm.paper_tables.TRIPLETS).
    """
    from cdm.datasets import get_dataset
    from cdm.provenance import code_revision
    from cdm.results_csv import RESULT_KEY_COLUMNS, load_results, save_result
    from cdm.significance import ALPHA, METRIC, comparisons, holm_flags, paired_t, read_perquery

    names = datasets.split(",")
    df = load_results()
    df = df[(df["job"] == "rerank") & (df["subset"] == "") & (df["variant"] == "") & (df["output"] != "")]
    df = df[df["phrasing"].isin([phrasing, "none", ""])]
    rows = {}
    for dataset in names:
        first_stage = get_dataset(dataset).default_first_stage
        for _, row in df[(df["dataset"] == dataset) & (df["first_stage"] == first_stage)].iterrows():
            rows[(row["system"], dataset)] = row

    def average(system):
        values = [float(rows[(system, d)][METRIC]) for d in names if (system, d) in rows]
        return sum(values) / len(values) if len(values) == len(names) else None

    systems = {s for s, _ in rows}
    revision = code_revision()
    code = revision.get("describe", revision.get("commit", ""))
    for a, b, comparison in comparisons({s: average(s) for s in systems}, phrasing):
        tested = [d for d in names if (a, d) in rows and (b, d) in rows]
        if not tested:
            click.echo(f"  {a} vs {b}: no common outputs, skipped", err=True)
            continue
        results = [paired_t(read_perquery(rows[(a, d)]["output"]), read_perquery(rows[(b, d)]["output"]))
                   for d in tested]
        flags = holm_flags([r["p"] for r in results], ALPHA)
        for dataset, result, significant in zip(tested, results, flags):
            key = {c: rows[(a, dataset)][c] for c in RESULT_KEY_COLUMNS}
            key.update(job="significance", variant=f"vs-{b}" + ("" if phrasing == "s1rank" else f"@{phrasing}"))
            save_result(key, {
                "sig_comparison": comparison, "sig_other": b, "sig_metric": METRIC, "sig_test": "paired-t",
                "sig_correction": f"holm-{len(tested)}", "sig_delta": result["delta"], "sig_t": result["t"],
                "sig_p": result["p"], "sig_holm": int(significant), "sig_n": result["n"], "code": code,
            })
        wins = sum(f and r["delta"] > 0 for f, r in zip(flags, results))
        losses = sum(f and r["delta"] < 0 for f, r in zip(flags, results))
        click.echo(f"{comparison:22} {a:22} vs {b:22} significant: {wins} better, {losses} worse of {len(tested)}")

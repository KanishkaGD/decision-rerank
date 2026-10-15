import click

# Modules, not their commands: importing a command named like its module (rerank.rerank) here would shadow the
# module attribute cdm.cli.rerank.
from cdm.cli import cutoff, finetune, first_stage, misc, posthoc, rerank, significance


@click.group()
def main():
    """
    System One decision models vs. reranker fine-tunes as rerankers (see README.md).
    """


for command in [
    misc.info,
    misc.datasets,
    misc.systems,
    misc.prefetch,
    first_stage.first_stage,
    rerank.rerank,
    cutoff.cutoff,
    posthoc.posthoc,
    significance.significance,
    finetune.finetune_data,
    finetune.finetune,
    misc.results,
]:
    main.add_command(command)

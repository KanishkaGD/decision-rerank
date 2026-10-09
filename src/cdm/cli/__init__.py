import click

# Modules, not their commands: importing a command named like its module (rerank.rerank) here would shadow the
# module attribute cdm.cli.rerank.
from cdm.cli import first_stage, misc, rerank


@click.group()
def main():
    """
    System One decision models vs. reranker fine-tunes as rerankers (see README.md). On the cluster, run these through the
    Slurm scripts (slurm/) rather than directly: the login node is not for computation.
    """


for command in [
    misc.info,
    misc.datasets,
    misc.systems,
    misc.prefetch,
    first_stage.first_stage,
    rerank.rerank,
    misc.results,
    misc.rename_system,
]:
    main.add_command(command)

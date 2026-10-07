#!/usr/bin/env bash
#SBATCH --job-name=cdm-first-stage
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%j_first-stage.out.tmp
# No --time/--account here: slurm/submit.sh passes both from slurm/config.sh (see CLAUDE.md). --output is relative
# to the submission directory, the repo root; start_job_log renames it and moves it to logs/done/ at the end.

# Retrieves and caches the first-stage candidates of ONE dataset (cdm first-stage): the run file, the query and
# passage texts, and the qrels, under runs/<dataset>/<retriever>/. Every reranker then reads this cache, so all of
# them see exactly the same candidates. Needs the Pyserini index prefetched (slurm/prefetch.sh).
#   slurm/submit.sh first-stage --datasets dl19,dl20        |  sbatch slurm/first-stage.sbatch <dataset> [cdm args]
set -euo pipefail
CDM_ROOT="${SLURM_SUBMIT_DIR}"
source "${CDM_ROOT}/slurm/log_command.sh"
log_job_command first-stage.sbatch "$@"
DATASET="${1:?Usage: sbatch slurm/first-stage.sbatch <dataset> [extra cdm first-stage args]}"
shift
start_job_log first-stage "${DATASET}"
source "${CDM_ROOT}/slurm/env.sh"

srun -n 1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" "${CONTAINER_CMD[@]}" cdm first-stage "${DATASET}" --threads "${SLURM_CPUS_PER_TASK:-8}" "$@"

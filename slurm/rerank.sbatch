#!/usr/bin/env bash
#SBATCH --job-name=cdm-rerank
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%j_rerank.out.tmp
# No --time/--account here: slurm/submit.sh passes both from slurm/config.sh (see CLAUDE.md). --output is relative
# to the submission directory, the repo root; start_job_log renames it and moves it to logs/done/ at the end.
# The GPU itself (--gres, --constraint) comes from GPU_GRES/GPU_CONSTRAINT in slurm/config.sh via slurm/submit.sh;
# a direct sbatch must pass them, e.g. sbatch --gres=gpu:1 --constraint=gpumem.48gib slurm/rerank.sbatch ...

# Reranks the cached first-stage candidates of ONE dataset with ONE system (cdm rerank), stores the raw score of
# every (query, passage, phrasing) under outputs/, and appends the metrics to results/results.csv.
#   cluster/submit.sh rerank <system> --datasets dl19       |  sbatch slurm/rerank.sbatch <system> <dataset> [cdm args]
set -euo pipefail
CDM_ROOT="${SLURM_SUBMIT_DIR}"
source "${CDM_ROOT}/slurm/log_command.sh"
log_job_command rerank.sbatch "$@"
SYSTEM="${1:?Usage: sbatch slurm/rerank.sbatch <system> <dataset> [extra cdm rerank args]}"
DATASET="${2:?Usage: sbatch slurm/rerank.sbatch <system> <dataset> [extra cdm rerank args]}"
shift 2
start_job_log rerank "${SYSTEM}_${DATASET}"
source "${CDM_ROOT}/slurm/env.sh"

srun -n 1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" "${CONTAINER_CMD[@]}" cdm rerank "${SYSTEM}" "${DATASET}" "$@"

#!/usr/bin/env bash
#SBATCH --job-name=cdm-prefetch
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%j_prefetch.out.tmp
# No --time/--account here: slurm/submit.sh passes both from slurm/config.sh (see CLAUDE.md). --output is relative
# to the submission directory, the repo root; start_job_log renames it and moves it to logs/done/ at the end.

# Batch counterpart of slurm/prefetch.interactive.sh, for downloads too big for the login node (the 39 GB MS MARCO v2
# passage index of DL 2021-2023). Compute nodes reach the internet (Hugging Face, the TypeSafe API: experiment log,
# 2026-10-06); if a download fails here, run slurm/prefetch.interactive.sh on the login node instead.
#   slurm/submit.sh prefetch --datasets dl21,dl22,dl23 --systems none  |  sbatch slurm/prefetch.sbatch [options]
set -euo pipefail
CDM_ROOT="${SLURM_SUBMIT_DIR}"
source "${CDM_ROOT}/slurm/log_command.sh"
log_job_command prefetch.sbatch "$@"
start_job_log prefetch
source "${CDM_ROOT}/slurm/config.sh"
export HF_HUB_OFFLINE=0
source "${CDM_ROOT}/slurm/env.sh"
prefetch_lists "$@"

srun -n 1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" "${CONTAINER_CMD[@]}" cdm prefetch --datasets "${PREFETCH_DATASETS}" --systems "${PREFETCH_SYSTEMS}" \
    --training-data "${PREFETCH_TRAINING_DATA}"

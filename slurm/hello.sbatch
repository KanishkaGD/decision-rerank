#!/usr/bin/env bash
#SBATCH --job-name=cdm-hello
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%j_hello.out.tmp
# No --time/--account here: slurm/submit.sh passes both from slurm/config.sh (see CLAUDE.md). --output is relative
# to the submission directory, the repo root; start_job_log renames it and moves it to logs/done/ at the end.

# Sanity check of submission, paths, logging and the container: prints the host, then `cdm info` (versions, paths,
# GPU) from inside the container.  Submit: slurm/submit.sh hello
set -euo pipefail
CDM_ROOT="${SLURM_SUBMIT_DIR}"
source "${CDM_ROOT}/slurm/log_command.sh"
log_job_command hello.sbatch "$@"
start_job_log hello
source "${CDM_ROOT}/slurm/env.sh"

echo "Hello from $(hostname), job ${SLURM_JOB_ID}"
srun -n 1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" "${CONTAINER_CMD[@]}" cdm info

#!/usr/bin/env bash
#SBATCH --job-name=cdm-finetune
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --output=logs/%j_finetune.out.tmp
# No --time/--account here: slurm/submit.sh passes both from slurm/config.sh (see CLAUDE.md). --output is relative
# to the submission directory, the repo root; start_job_log renames it and moves it to logs/done/ at the end.
# The GPUs (--gres gpu:N, --constraint) come from finetune_gres_for/finetune_constraint_for in slurm/config.sh via
# slurm/submit.sh; all N GPUs are on one node and each holds a full copy of the frozen base model (data parallel).

# LoRA reranker fine-tune of ONE system (a registered reranker with an adapter, src/cdm/rerankers/__init__.py) on ONE
# set of training groups (data/finetune/<data>/, `cdm finetune-data`), one process per GPU (torchrun). The job stops
# with a checkpoint before its time limit; pass --resume to continue (docs/notes/reranker-27b-plan.md).
#   slurm/submit.sh finetune <system> --datasets <data> [cdm finetune args]
#   sbatch --gres=gpu:4 --constraint=gpumem.80gib slurm/finetune.sbatch <system> <data> [cdm finetune args]
set -euo pipefail
CDM_ROOT="${SLURM_SUBMIT_DIR}"
source "${CDM_ROOT}/slurm/log_command.sh"
log_job_command finetune.sbatch "$@"
SYSTEM="${1:?Usage: sbatch slurm/finetune.sbatch <system> <data> [extra cdm finetune args]}"
DATA="${2:?Usage: sbatch slurm/finetune.sbatch <system> <data> [extra cdm finetune args]}"
shift 2
start_job_log finetune "${SYSTEM}_${DATA}"
source "${CDM_ROOT}/slurm/config.sh"
source "${CDM_ROOT}/slurm/env.sh"
NGPU="${SLURM_GPUS_ON_NODE:-1}"

srun -n 1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" "${CONTAINER_CMD[@]}" \
    torchrun --standalone --nproc_per_node="${NGPU}" --no-python cdm finetune "${SYSTEM}" "${DATA}" "$@"

#!/usr/bin/env bash
# Interactive counterpart of finetune.sbatch: run inside a shell from slurm/interactive.sh (with GPUs, see there).
# Usage: slurm/finetune.interactive.sh <system> <data> [extra cdm finetune args]
set -euo pipefail
CDM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SYSTEM="${1:?Usage: slurm/finetune.interactive.sh <system> <data> [extra cdm finetune args]}"
DATA="${2:?Usage: slurm/finetune.interactive.sh <system> <data> [extra cdm finetune args]}"
shift 2
source "${CDM_ROOT}/slurm/config.sh"
source "${CDM_ROOT}/slurm/env.sh"
NGPU="${SLURM_GPUS_ON_NODE:-1}"

"${CONTAINER_CMD[@]}" \
    torchrun --standalone --nproc_per_node="${NGPU}" --no-python cdm finetune "${SYSTEM}" "${DATA}" "$@"

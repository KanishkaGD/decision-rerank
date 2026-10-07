#!/usr/bin/env bash
# Interactive counterpart of rerank.sbatch: run inside a shell from slurm/interactive.sh (with a GPU, see there).
# Usage: slurm/rerank.interactive.sh <system> <dataset> [extra cdm rerank args]
set -euo pipefail
CDM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SYSTEM="${1:?Usage: slurm/rerank.interactive.sh <system> <dataset> [extra cdm rerank args]}"
DATASET="${2:?Usage: slurm/rerank.interactive.sh <system> <dataset> [extra cdm rerank args]}"
shift 2
source "${CDM_ROOT}/slurm/config.sh"
CDM_CONTAINER="$(container_for "${SYSTEM}")"
source "${CDM_ROOT}/slurm/env.sh"

"${CONTAINER_CMD[@]}" cdm rerank "${SYSTEM}" "${DATASET}" "$@"

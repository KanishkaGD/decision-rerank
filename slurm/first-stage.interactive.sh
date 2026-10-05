#!/usr/bin/env bash
# Interactive counterpart of first-stage.sbatch: run inside a shell from slurm/interactive.sh. Same command, no srun.
# Usage: slurm/first-stage.interactive.sh <dataset> [extra cdm first-stage args]
set -euo pipefail
CDM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${CDM_ROOT}/slurm/env.sh"
DATASET="${1:?Usage: slurm/first-stage.interactive.sh <dataset> [extra cdm first-stage args]}"
shift

"${CONTAINER_CMD[@]}" cdm first-stage "${DATASET}" --threads "${SLURM_CPUS_PER_TASK:-8}" "$@"

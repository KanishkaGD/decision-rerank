#!/usr/bin/env bash
# Interactive counterpart of hello.sbatch: run inside a shell from slurm/interactive.sh.
set -euo pipefail
CDM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${CDM_ROOT}/slurm/env.sh"

echo "Hello from $(hostname), job ${SLURM_JOB_ID:-<no Slurm allocation>}"
"${CONTAINER_CMD[@]}" cdm info

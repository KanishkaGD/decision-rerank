#!/usr/bin/env bash
# Reserves an interactive shell on a compute node, so you queue once instead of once per command. Inside it, run the
# slurm/*.interactive.sh scripts directly. Start it from the login node.
#
# Usage: slurm/interactive.sh [time_limit] [partition] [extra srun args...]
#   slurm/interactive.sh                                                  4h on general (CPU)
#   slurm/interactive.sh 02:00:00 gpu --gres=gpu:1 --constraint=gpumem.48gib
#   slurm/interactive.sh 01:00:00 gpu --gres=gpu:a100-sxm:1
set -euo pipefail
CDM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${CDM_ROOT}/slurm/config.sh"

TIME_LIMIT="${1:-04:00:00}"
PARTITION="${2:-general}"
shift "$(( $# < 2 ? $# : 2 ))"

exec srun --pty \
    --account="${SLURM_ACCOUNT}" \
    --partition="${PARTITION}" \
    --time="${TIME_LIMIT}" \
    --cpus-per-task=8 \
    --mem=32G \
    "$@" \
    bash -i

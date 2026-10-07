#!/usr/bin/env bash
# Runs ONE command in the container on a compute node via srun and waits for it -- the non-interactive way to try
# something quickly from the login node. The code is snapshotted first (slurm/snapshot.sh), as for batch jobs.
#
# Usage: slurm/exec.sh [--gpu] [--time hh:mm:ss] [--partition p] [--gres g] [--constraint c] [--mem m]
#                      [--nodelist n] -- <cmd...>        (--nodelist: check one node, e.g. after failures there)
#   slurm/exec.sh -- cdm info
#   slurm/exec.sh --gpu -- cdm rerank bge-m3 scifact --max-queries 20 --no-save
# Defaults: partition short (max 1h, higher priority), 00:20:00, 8 CPUs, 32G; --gpu adds GPU_GRES/GPU_CONSTRAINT from slurm/config.sh.
set -euo pipefail
CDM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${CDM_ROOT}/slurm/config.sh"

PARTITION="short"
TIME_LIMIT="00:20:00"
MEM="32G"
GRES_VALUE=""
CONSTRAINT=""
NODELIST=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --gpu) GRES_VALUE="${GPU_GRES}"; CONSTRAINT="${CONSTRAINT:-${GPU_CONSTRAINT}}" ;;
        --gres) GRES_VALUE="$2"; shift ;;
        --constraint) CONSTRAINT="$2"; shift ;;
        --time) TIME_LIMIT="$2"; shift ;;
        --partition) PARTITION="$2"; shift ;;
        --mem) MEM="$2"; shift ;;
        --nodelist) NODELIST=(--nodelist="$2"); shift ;;
        --) shift; break ;;
        *) echo "slurm/exec.sh: unknown option $1 (put the command after --)" >&2; exit 2 ;;
    esac
    shift
done
[ "$#" -gt 0 ] || { echo "Usage: slurm/exec.sh [options] -- <command...>" >&2; exit 2; }
GRES=()
[ -n "${GRES_VALUE}" ] && GRES+=(--gres="${GRES_VALUE}")
source "${CDM_ROOT}/slurm/snapshot.sh"
snapshot_code
[ -n "${CONSTRAINT}" ] && GRES+=(--constraint="${CONSTRAINT}")

# env.sh is sourced inside the allocation (so it sees the GPU and sets --nv), hence the bash -c.
exec srun \
    --account="${SLURM_ACCOUNT}" \
    --partition="${PARTITION}" \
    --time="${TIME_LIMIT}" \
    --cpus-per-task=8 \
    --mem="${MEM}" \
    --job-name=cdm-exec \
    "${GRES[@]}" "${NODELIST[@]}" \
    bash -c 'CDM_ROOT="$1"; shift; source "${CDM_ROOT}/slurm/env.sh" || exit 1; exec "${CONTAINER_CMD[@]}" "$@"' \
    _ "${CDM_ROOT}" "$@"

#!/usr/bin/env bash
# Opens an interactive shell on a COMPUTE node (slurm/interactive.sh) from your machine, in the remote repo root,
# for working by hand: run the slurm/*.interactive.sh scripts there. Syncs the code first. For humans only -- a
# coding agent should use cluster/exec.sh (one command) or cluster/submit.sh (batch jobs).
#
# Usage: cluster/interactive.sh [--no-sync] [time_limit] [partition] [extra srun args...]
#   cluster/interactive.sh                                              4h on general (CPU)
#   cluster/interactive.sh 02:00:00 gpu --gres=gpu:1 --constraint=gpumem.80gib
source "$(dirname "$0")/lib.sh"

SYNC=1
if [ "${1:-}" = "--no-sync" ]; then
    SYNC=0
    shift
fi

ensure_connection
[ "${SYNC}" -eq 1 ] && "${CDM_ROOT}/cluster/sync.sh" >/dev/null
ROOT="$(remote_root)"
exec ssh "${SSH_OPTS[@]}" -t "${CLUSTER_HOST}" "cd $(printf '%q' "${ROOT}") && bash slurm/interactive.sh $(printf '%q ' "$@")"

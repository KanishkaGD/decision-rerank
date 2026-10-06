#!/usr/bin/env bash
# Runs ONE command inside the container on a compute node and waits for it, streaming its output back -- for
# quick checks and small experiments that don't deserve a batch job. Syncs the code first. Goes through
# slurm/exec.sh (srun), so it queues like any job; the `short` partition (max 1h, higher priority) is the default.
#
# Usage: cluster/exec.sh [--no-sync] [--gpu] [--gres g] [--constraint c] [--time hh:mm:ss] [--partition p] [--mem m]
#        -- <command...>
#   cluster/exec.sh -- cdm info
#   cluster/exec.sh --gpu -- cdm rerank minilm-l6 scifact --max-queries 20 --no-save
#   cluster/exec.sh --gpu --time 00:30:00 -- python -c 'import torch; print(torch.cuda.get_device_name())'
#
# Runs with a remote terminal, so interrupting it (Ctrl-C, a timeout) cancels the srun job too. Output lines end in
# \r\n because of that terminal. Use cluster/submit.sh for anything long.
source "$(dirname "$0")/lib.sh"

SYNC=1
if [ "${1:-}" = "--no-sync" ]; then
    SYNC=0
    shift
fi

ensure_connection
[ "${SYNC}" -eq 1 ] && "${CDM_ROOT}/cluster/sync.sh" >/dev/null
remote_exec_tty bash slurm/exec.sh "$@"

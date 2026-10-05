#!/usr/bin/env bash
# Submits Slurm batch jobs from your machine: syncs the code (cluster/sync.sh), then runs slurm/submit.sh with the
# same arguments on the login node. Everything after the optional --no-sync is passed through unchanged.
#
# Usage: cluster/submit.sh [--no-sync] <job> [args...]
#   cluster/submit.sh hello
#   cluster/submit.sh first-stage --datasets dl19,dl20
#   cluster/submit.sh rerank minilm-l6 --datasets all --dry-run
#   cluster/submit.sh --help                 slurm/submit.sh's own help
#
# Prints the job ids ("Submitted batch job <id>"); follow them with cluster/status.sh and cluster/logs.sh <id>.
source "$(dirname "$0")/lib.sh"

SYNC=1
if [ "${1:-}" = "--no-sync" ]; then
    SYNC=0
    shift
fi

ensure_connection
[ "${SYNC}" -eq 1 ] && "${CDM_ROOT}/cluster/sync.sh"
remote_exec bash slurm/submit.sh "$@"

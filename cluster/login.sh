#!/usr/bin/env bash
# Light, cluster-related commands on the LOGIN node, in the remote repo root -- prefetching data, building the
# container, inspecting files. Not for computation: the FAITH docs forbid compute-heavy or long-running work on
# login nodes. Use cluster/exec.sh (compute node, blocking) or cluster/submit.sh (batch) for that.
#
# Usage: cluster/login.sh [--no-sync] <command...>
#   cluster/login.sh bash slurm/prefetch.sh --datasets dl19,scifact --models minilm-l6
#   cluster/login.sh bash container/build.sh
#   cluster/login.sh ls -la runs/
#   cluster/login.sh                      an interactive shell on the login node, in the repo root
source "$(dirname "$0")/lib.sh"

SYNC=1
if [ "${1:-}" = "--no-sync" ]; then
    SYNC=0
    shift
fi

ensure_connection
if [ "$#" -eq 0 ]; then
    ROOT="$(remote_root)"
    exec ssh "${SSH_OPTS[@]}" -t "${CLUSTER_HOST}" "cd $(printf '%q' "${ROOT}") && exec \$SHELL -l"
fi
[ "${SYNC}" -eq 1 ] && "${CDM_ROOT}/cluster/sync.sh" >/dev/null
remote_exec "$@"

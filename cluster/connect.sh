#!/usr/bin/env bash
# Opens, checks or closes the shared SSH connection that every other cluster/*.sh reuses.
#
# Usage: cluster/connect.sh            open it (asks for your password once, unless a Kerberos ticket is valid)
#        cluster/connect.sh --check    exit 0 if it is open, 1 if not
#        cluster/connect.sh --close    close it
#
# Run this yourself in a terminal: it is the one step that may need a password, so a coding agent can't do it.
# The connection stays open in the background for CONTROL_PERSIST (cluster/cluster.env, default 12h) after the
# last command that used it.
source "$(dirname "$0")/lib.sh"

case "${1:-}" in
    --check)
        if ssh "${SSH_OPTS[@]}" -O check "${CLUSTER_HOST}" 2>/dev/null; then
            echo "Connected to ${CLUSTER_HOST} ($(remote_root))."
            exit 0
        fi
        echo "Not connected to ${CLUSTER_HOST}. Run cluster/connect.sh." >&2
        exit 1
        ;;
    --close)
        ssh "${SSH_OPTS[@]}" -O exit "${CLUSTER_HOST}" 2>/dev/null || true
        echo "Closed the connection to ${CLUSTER_HOST}."
        exit 0
        ;;
    "" | --open) ;;
    *)
        echo "Usage: $0 [--check|--close]" >&2
        exit 2
        ;;
esac

if ssh "${SSH_OPTS[@]}" -O check "${CLUSTER_HOST}" 2>/dev/null; then
    echo "Already connected to ${CLUSTER_HOST}."
else
    # Interactive (BatchMode=no) so ssh can ask for the password; -f -N backgrounds it without running a command,
    # leaving just the shared connection. ControlMaster=auto (not yes) also replaces a stale socket left behind by
    # a crash or reboot.
    ssh "${SSH_BASE_OPTS[@]}" -o BatchMode=no -o ControlMaster=auto -f -N "${CLUSTER_HOST}"
    echo "Connected to ${CLUSTER_HOST}; the connection stays open for ${CONTROL_PERSIST} after its last use."
fi
echo "Remote repo root: $(remote_root)"

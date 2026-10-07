# Shared helpers for the cluster/*.sh wrappers. Source it, don't run it:
#   source "$(dirname "$0")/lib.sh"
#
# The wrappers run on YOUR machine (laptop/workstation), never on the cluster. Each one talks to the FAITH login
# node over a single shared SSH connection (ssh ControlMaster): FAITH accepts only password or Kerberos logins, no
# SSH keys, so cluster/connect.sh opens the connection once (password prompt, or none with a Kerberos ticket) and
# every later ssh/rsync call reuses it without asking again. That is what makes the wrappers usable from scripts
# and coding agents, which can't type a password.
#
# The login node only ever runs light, cluster-related commands (sbatch, squeue, rsync, downloads); the FAITH docs
# forbid compute-heavy work and autonomous agents there. Real work goes to compute nodes via slurm/submit.sh
# (batch jobs) or slurm/exec.sh (one blocking srun command).

set -euo pipefail

CDM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=cluster/cluster.env
source "${CDM_ROOT}/cluster/cluster.env"
if [ -f "${CDM_ROOT}/cluster/cluster.local.env" ]; then
    # shellcheck disable=SC1091
    source "${CDM_ROOT}/cluster/cluster.local.env"
fi

# Same ControlPath as the FAITH docs recommend for ~/.ssh/config, so a connection you opened by hand (plain
# `ssh faith`) is shared with these scripts too, and vice versa.
mkdir -p "${HOME}/.ssh/cm"
chmod 700 "${HOME}/.ssh/cm"
# ssh uses the FIRST value it gets for an option, so the base options leave out ControlMaster and BatchMode:
# SSH_OPTS (scripts) adds non-interactive ones, cluster/connect.sh its own interactive ones.
SSH_BASE_OPTS=(
    -o "ControlPath=${HOME}/.ssh/cm/%r@%h:%p"
    -o "ControlPersist=${CONTROL_PERSIST}"
    -o "ServerAliveInterval=60"
    -o "ConnectTimeout=15"
)
SSH_OPTS=(
    "${SSH_BASE_OPTS[@]}"
    # Only reuse a connection, never become one: a backgrounded master would keep the caller's stderr open for
    # ControlPersist. ensure_connection and cluster/connect.sh are the only places that open one.
    -o "ControlMaster=no"
    # Never prompt: without an open connection (or a Kerberos ticket), fail fast instead of hanging on a password
    # prompt nobody sees.
    -o "BatchMode=yes"
)

die() {
    echo "$(basename "$0"): $*" >&2
    exit 1
}

# Succeeds if the shared connection is up. Without one, tries a non-interactive login (works with a valid Kerberos
# ticket, `kinit <user>@...`), which then becomes the shared connection.
ensure_connection() {
    if ssh "${SSH_OPTS[@]}" -O check "${CLUSTER_HOST}" 2>/dev/null; then
        return 0
    fi
    if ssh "${SSH_BASE_OPTS[@]}" -o ControlMaster=auto -o BatchMode=yes "${CLUSTER_HOST}" true 2>/dev/null; then
        return 0
    fi
    die "no connection to '${CLUSTER_HOST}'. Run cluster/connect.sh first (asks for your password once), or get a
Kerberos ticket with kinit. Setup: cluster/ssh_config.example and README.md, 'Cluster'."
}

# Plain ssh over the shared connection, for the rare caller that needs its own command string.
cluster_ssh() {
    ssh "${SSH_OPTS[@]}" "${CLUSTER_HOST}" "$@"
}

# The repo root on the cluster, e.g. /shared/jdoe/calibrated-decision-models-for-ir.
remote_root() {
    if [ -n "${REMOTE_ROOT}" ]; then
        echo "${REMOTE_ROOT}"
        return
    fi
    # Cached per host and project name, so changing either in cluster.env doesn't reuse a stale path.
    local cache="${CDM_ROOT}/cluster/.remote_root" tag="${CLUSTER_HOST} ${PROJECT_NAME} "
    if [ -s "${cache}" ] && [[ "$(cat "${cache}")" == "${tag}"* ]]; then
        local cached
        cached="$(cat "${cache}")"
        echo "${cached#"${tag}"}"
        return
    fi
    local user
    user="$(cluster_ssh 'id -un')"
    [ -n "${user}" ] || die "could not determine your user name on ${CLUSTER_HOST}"
    echo "${tag}/shared/${user}/${PROJECT_NAME}" >"${cache}"
    echo "/shared/${user}/${PROJECT_NAME}"
}

# Runs one command with its arguments in the remote repo root, each argument shell-quoted, so
#   remote_exec bash slurm/submit.sh rerank minilm --datasets dl19
# arrives exactly as written. Exit status and stdout/stderr are the remote command's.
remote_exec() {
    local root
    root="$(remote_root)"
    cluster_ssh "cd $(printf '%q' "${root}") && $(printf '%q ' "$@")"
}

# Like remote_exec, but with a remote pseudo-terminal (ssh -tt): when the local side is interrupted (Ctrl-C, an
# agent's tool timeout), the remote command gets SIGHUP, so a blocking srun is cancelled instead of holding its
# allocation until the time limit.
remote_exec_tty() {
    local root
    root="$(remote_root)"
    ssh "${SSH_OPTS[@]}" -tt "${CLUSTER_HOST}" "cd $(printf '%q' "${root}") && $(printf '%q ' "$@")"
}

# Runs a raw command string in the remote repo root, for globs and pipes that must expand on the cluster, e.g.
#   remote_shell 'tail -n 50 logs/123_*.out'
# Never pass untrusted input here.
remote_shell() {
    local root
    root="$(remote_root)"
    cluster_ssh "cd $(printf '%q' "${root}") && $1"
}

# The -e argument for rsync, so transfers reuse the shared connection too.
rsync_ssh() {
    printf 'ssh'
    printf ' %q' "${SSH_OPTS[@]}"
}

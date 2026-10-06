#!/usr/bin/env bash
# One-time setup of the shared repository on the VM (docs/vm-setup.md), by a human with sudo:
#   - a group (default cdm) for everyone who works in the repository, and /srv/cdm owned by it;
#   - a clone at /srv/cdm/repo, readable and writable by the group (core.sharedRepository), on the integration branch;
#     task worktrees go to /srv/cdm/worktrees/<id> (scripts/task.sh);
#   - the collaboration settings in git config: integration branch, remote and remote branch for scripts/task.sh.
#
# Usage: sudo cluster/vm/setup-repo.sh <git-url> [--group cdm] [--branch main] [--remote-branch main] [--users "a b"]
#                                         [--shared-user]
#   --shared-user  several people work under ONE Linux user (the one running sudo): sets cdm.sharedUser, so claims
#                  and edits require each person's profile (cluster/vm/setup-person.sh). This is the chosen setup.
#   e.g. sudo cluster/vm/setup-repo.sh https://github.com/KanishkaGD/decision-rerank.git --remote-branch dev-michael \
#            --shared-user
set -euo pipefail
URL="${1:?usage: sudo cluster/vm/setup-repo.sh <git-url> [options]}"
shift
GROUP="cdm"
BRANCH="main"
REMOTE_BRANCH=""
USERS=""
BASE="/srv/cdm"
SHARED_USER=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --group) GROUP="$2"; shift ;;
        --branch) BRANCH="$2"; shift ;;
        --remote-branch) REMOTE_BRANCH="$2"; shift ;;
        --users) USERS="$2"; shift ;;
        --base) BASE="$2"; shift ;;
        --shared-user) SHARED_USER=1 ;;
        *) echo "setup-repo.sh: unknown option $1" >&2; exit 2 ;;
    esac
    shift
done
REMOTE_BRANCH="${REMOTE_BRANCH:-${BRANCH}}"
[ "$(id -u)" -eq 0 ] || { echo "Run with sudo (creates a group and /srv/cdm)." >&2; exit 1; }
OWNER="${SUDO_USER:-root}"

getent group "${GROUP}" >/dev/null || groupadd "${GROUP}"
for u in ${USERS} ${OWNER}; do
    [ "${u}" = root ] || usermod -aG "${GROUP}" "${u}"
done
mkdir -p "${BASE}/worktrees" "${BASE}/secrets"
chown "${OWNER}:${GROUP}" "${BASE}" "${BASE}/worktrees" "${BASE}/secrets"
chmod 2775 "${BASE}" "${BASE}/worktrees"
chmod 2750 "${BASE}/secrets"

if [ ! -d "${BASE}/repo/.git" ]; then
    sudo -u "${OWNER}" git clone --branch "${REMOTE_BRANCH}" "${URL}" "${BASE}/repo"
    sudo -u "${OWNER}" git -C "${BASE}/repo" checkout -B "${BRANCH}"
fi
cd "${BASE}/repo"
sudo -u "${OWNER}" git config core.sharedRepository group
sudo -u "${OWNER}" git config cdm.mainBranch "${BRANCH}"
sudo -u "${OWNER}" git config cdm.remote origin
sudo -u "${OWNER}" git config cdm.remoteBranch "${REMOTE_BRANCH}"
[ "${SHARED_USER}" -eq 1 ] && sudo -u "${OWNER}" git config cdm.sharedUser true
sudo -u "${OWNER}" git branch --set-upstream-to "origin/${REMOTE_BRANCH}" "${BRANCH}" 2>/dev/null || true
chgrp -R "${GROUP}" "${BASE}/repo"
chmod -R g+rwX "${BASE}/repo"
find "${BASE}/repo" -type d -exec chmod g+s {} +
echo "Repository ready: ${BASE}/repo (group ${GROUP}; integration branch ${BRANCH} <-> origin/${REMOTE_BRANCH})."
echo "Members of ${GROUP} must log in again for the group to apply. Next: docs/vm-setup.md, steps 3-6."

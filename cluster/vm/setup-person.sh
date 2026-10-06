#!/usr/bin/env bash
# Several people share ONE Linux user on the VM (docs/vm-setup.md). This writes a profile for one person,
# ~/people/<name>.env, that each person sources before starting a shell's work or Claude Code:
#
#   source ~/people/<name>.env
#
# It keeps apart what would otherwise be shared under one Linux user:
#   CDM_WHO                       the name on task claims; the guard and scripts/task.sh check ownership against it
#   GIT_AUTHOR_* / GIT_COMMITTER_* commit authorship
#   CLAUDE_CONFIG_DIR             the person's own Claude Code login, settings, memory and session history
#   GH_CONFIG_DIR                 the person's own GitHub login (gh), used by git for pushes (gh auth setup-git)
#
# Usage: cluster/vm/setup-person.sh <name> "<Full Name>" <email>
#   e.g. cluster/vm/setup-person.sh michael "Michael Dinzinger" michael.dinzinger@uni-passau.de
set -euo pipefail
NAME="${1:?usage: cluster/vm/setup-person.sh <name> \"<Full Name>\" <email>}"
FULL="${2:?full name missing}"
EMAIL="${3:?email missing}"
[[ "${NAME}" =~ ^[a-z][a-z0-9-]{1,30}$ ]] || { echo "name: lowercase letters, digits, '-' (e.g. michael)" >&2; exit 2; }

mkdir -p ~/people "${HOME}/.claude-${NAME}" "${HOME}/.config/gh-${NAME}"
chmod 700 ~/people "${HOME}/.claude-${NAME}" "${HOME}/.config/gh-${NAME}"
PROFILE=~/people/${NAME}.env
cat >"${PROFILE}" <<EOF
# Profile of ${FULL} on the shared VM user (cluster/vm/setup-person.sh). Source it: source ${PROFILE}
export CDM_WHO=$(printf '%q' "${NAME}")
export GIT_AUTHOR_NAME=$(printf '%q' "${FULL}")
export GIT_AUTHOR_EMAIL=$(printf '%q' "${EMAIL}")
export GIT_COMMITTER_NAME=$(printf '%q' "${FULL}")
export GIT_COMMITTER_EMAIL=$(printf '%q' "${EMAIL}")
export CLAUDE_CONFIG_DIR="\${HOME}/.claude-${NAME}"
export GH_CONFIG_DIR="\${HOME}/.config/gh-${NAME}"
# Shows whose session this is in the prompt, e.g. "[michael] cdm@vm:~$"
case "\${PS1:-}" in "[${NAME}]"*) ;; *) PS1="[${NAME}] \${PS1:-\\\\u@\\\\h:\\\\w\\\\$ }" ;; esac
EOF
chmod 600 "${PROFILE}"

# git pushes over HTTPS ask gh for a token; gh reads GH_CONFIG_DIR, so each person pushes with their own login.
if command -v gh >/dev/null; then
    git config --global credential.https://github.com.helper '!gh auth git-credential'
fi

echo "Wrote ${PROFILE}. ${FULL} now runs, once:"
echo "  source ${PROFILE}"
echo "  gh auth login            # own GitHub account (stored in ${HOME}/.config/gh-${NAME})"
echo "  claude                   # own Claude login (stored in ${HOME}/.claude-${NAME}), then /status to check"
echo "and from then on starts every session with: source ${PROFILE}"

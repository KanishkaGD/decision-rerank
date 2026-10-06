#!/usr/bin/env bash
# Task claims and worktrees for several people and agents working in one repository (docs/collaboration.md).
#
# Every unit of work is a task with a short id (lowercase, digits, dashes: "jev-dl19-runs"). A task gets its own
# branch (task/<id>) and its own git worktree (../worktrees/<id> next to the main checkout), and a claim that every
# other checkout sees at once: claims live in the repository's shared git directory, not in a tracked file, so
# claiming needs no commit and two checkouts can't both claim the same id (mkdir is atomic).
#
# Usage (from any checkout of the repository):
#   scripts/task.sh list                          active claims: id, who, branch, since, description
#   scripts/task.sh start <id> "<description>"    claim <id>, create branch task/<id> and its worktree from main
#   scripts/task.sh status                        the claim of the current worktree (exit 1 if there is none)
#   scripts/task.sh finish [--keep-worktree]      in a task worktree: check, rebase on main, fast-forward main,
#                                                 push main (if a remote is configured), release the claim
#   scripts/task.sh release <id>                  drop a claim without merging (abandoned work); keeps the branch
#
# Configuration (git config, shared by all worktrees):
#   cdm.mainBranch   integration branch (default: main)
#   cdm.remote       remote to push the integration branch to after finish (default: none, local only)
#   cdm.remoteBranch branch on that remote (default: same as cdm.mainBranch)
#   CDM_WHO          who you are in claims (default: git config user.name, else $USER)
set -euo pipefail

die() { echo "task.sh: $*" >&2; exit 1; }

git rev-parse --git-dir >/dev/null 2>&1 || die "not inside a git repository"
COMMON_DIR="$(cd "$(git rev-parse --git-common-dir)" && pwd)"
CLAIMS="${COMMON_DIR}/cdm-claims"
mkdir -p "${CLAIMS}"
MAIN_BRANCH="$(git config cdm.mainBranch || echo main)"
REMOTE="$(git config cdm.remote || true)"
REMOTE_BRANCH="$(git config cdm.remoteBranch || echo "${MAIN_BRANCH}")"
WHO="${CDM_WHO:-$(git config user.name || echo "${USER:-unknown}")}"
# The main checkout is the first entry of `git worktree list`; it stays on the integration branch.
MAIN_CHECKOUT="$(git worktree list --porcelain | sed -n '1s/^worktree //p')"
WORKTREES="$(dirname "${MAIN_CHECKOUT}")/worktrees"

valid_id() { [[ "$1" =~ ^[a-z0-9][a-z0-9-]{1,48}$ ]] || die "task id '$1': use 2-49 characters a-z, 0-9, '-'"; }

current_task() {
    local branch
    branch="$(git rev-parse --abbrev-ref HEAD)"
    [[ "${branch}" == task/* ]] && echo "${branch#task/}"
}

field() { sed -n "s/^$2=//p" "${CLAIMS}/$1/claim" 2>/dev/null; }

cmd="${1:-list}"
shift || true
case "${cmd}" in
    list)
        found=0
        for d in "${CLAIMS}"/*/; do
            [ -f "${d}claim" ] || continue
            id="$(basename "${d}")"
            found=1
            printf '%-28s %-18s %-24s %s  %s\n' "${id}" "$(field "${id}" who)" "$(field "${id}" branch)" \
                "$(field "${id}" since)" "$(field "${id}" description)"
        done
        [ "${found}" -eq 1 ] || echo "No active tasks."
        ;;

    start)
        id="${1:-}"
        desc="${2:-}"
        [ -n "${id}" ] && [ -n "${desc}" ] || die 'usage: scripts/task.sh start <id> "<description>"'
        valid_id "${id}"
        git show-ref --verify --quiet "refs/heads/task/${id}" && die "branch task/${id} already exists; pick a new id"
        mkdir "${CLAIMS}/${id}" 2>/dev/null || die "task '${id}' is already claimed by $(field "${id}" who)"
        {
            echo "who=${WHO}"
            echo "branch=task/${id}"
            echo "worktree=${WORKTREES}/${id}"
            echo "since=$(date '+%Y-%m-%d %H:%M')"
            echo "description=${desc}"
        } >"${CLAIMS}/${id}/claim"
        mkdir -p "${WORKTREES}"
        if ! git worktree add -b "task/${id}" "${WORKTREES}/${id}" "${MAIN_BRANCH}" >/dev/null; then
            rm -rf "${CLAIMS:?}/${id}"
            die "could not create the worktree; claim released"
        fi
        # Untracked per-machine files every checkout needs: secrets and local cluster settings.
        for f in .env cluster/cluster.local.env .claude/settings.local.json; do
            [ -f "${MAIN_CHECKOUT}/${f}" ] && mkdir -p "$(dirname "${WORKTREES}/${id}/${f}")" &&
                ln -sfn "${MAIN_CHECKOUT}/${f}" "${WORKTREES}/${id}/${f}"
        done
        echo "Claimed '${id}'. Work in: ${WORKTREES}/${id}   (branch task/${id}, from ${MAIN_BRANCH})"
        ;;

    status)
        id="$(current_task)" || die "this checkout is not on a task/<id> branch (scripts/task.sh start ...)"
        [ -f "${CLAIMS}/${id}/claim" ] || die "branch task/${id} has no active claim (finished or released)"
        cat "${CLAIMS}/${id}/claim"
        ;;

    finish)
        keep=0
        [ "${1:-}" = "--keep-worktree" ] && keep=1
        id="$(current_task)" || die "run finish inside a task worktree (branch task/<id>)"
        [ -f "${CLAIMS}/${id}/claim" ] || die "task '${id}' has no active claim"
        [ -z "$(git status --porcelain --untracked-files=no)" ] || die "commit or stash your changes first"
        # One finish at a time: two agents fast-forwarding main concurrently would race.
        lock="${COMMON_DIR}/cdm-finish.lock"
        mkdir "${lock}" 2>/dev/null || die "another finish is running (remove ${lock} if it is stale)"
        trap 'rmdir "${lock}" 2>/dev/null || true' EXIT
        # Bring main up to date with the remote first (someone may have finished from another machine). A
        # remote-tracking ref, not FETCH_HEAD: FETCH_HEAD is per worktree, the main checkout wouldn't see it.
        if [ -n "${REMOTE}" ] && git ls-remote --exit-code -q "${REMOTE}" "refs/heads/${REMOTE_BRANCH}" >/dev/null; then
            tracking="refs/remotes/${REMOTE}/${REMOTE_BRANCH}"
            git fetch -q "${REMOTE}" "+refs/heads/${REMOTE_BRANCH}:${tracking}" || die "fetch from ${REMOTE} failed"
            git -C "${MAIN_CHECKOUT}" merge -q --ff-only "${tracking}" ||
                die "${MAIN_BRANCH} has diverged from ${REMOTE}/${REMOTE_BRANCH}; reconcile it in ${MAIN_CHECKOUT}"
        fi
        git rebase -q "${MAIN_BRANCH}" ||
            die "rebase on ${MAIN_BRANCH} has conflicts: resolve them (git status), git rebase --continue, finish again"
        "$(git rev-parse --show-toplevel)/scripts/check.sh" || die "checks failed; fix them and finish again"
        [ -z "$(git -C "${MAIN_CHECKOUT}" status --porcelain --untracked-files=no)" ] ||
            die "the main checkout ${MAIN_CHECKOUT} has uncommitted changes; nobody should edit there"
        git -C "${MAIN_CHECKOUT}" merge -q --ff-only "task/${id}" || die "fast-forward of ${MAIN_BRANCH} failed"
        if [ -n "${REMOTE}" ]; then
            git -C "${MAIN_CHECKOUT}" push -q "${REMOTE}" "${MAIN_BRANCH}:${REMOTE_BRANCH}" ||
                die "push to ${REMOTE}/${REMOTE_BRANCH} failed; ${MAIN_BRANCH} is merged locally, push it by hand"
        fi
        rm -rf "${CLAIMS:?}/${id}"
        echo "Finished '${id}': ${MAIN_BRANCH} is now $(git rev-parse --short HEAD)${REMOTE:+, pushed to ${REMOTE}/${REMOTE_BRANCH}}."
        if [ "${keep}" -eq 0 ]; then
            here="$(git rev-parse --show-toplevel)"
            cd "${MAIN_CHECKOUT}"
            git worktree remove "${here}" && git branch -q -d "task/${id}" &&
                echo "Removed the worktree ${here} and the branch task/${id}."
        fi
        ;;

    release)
        id="${1:-}"
        [ -n "${id}" ] || die "usage: scripts/task.sh release <id>"
        [ -d "${CLAIMS}/${id}" ] || die "no active claim '${id}'"
        rm -rf "${CLAIMS:?}/${id}"
        echo "Released '${id}' (branch task/${id} and its worktree are kept; delete them when no longer needed)."
        ;;

    *) die "unknown command '${cmd}' (list, start, status, finish, release)" ;;
esac

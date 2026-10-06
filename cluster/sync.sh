#!/usr/bin/env bash
# Copies the local working tree to the remote repo root on /shared, so jobs run exactly the code you have here,
# committed or not. Every other cluster/*.sh that starts work calls this first (unless given --no-sync).
#
# Usage: cluster/sync.sh [--dry-run]
#
# What goes where:
#   - Everything git would track goes up (rsync honours .gitignore). --delete removes remote files you deleted
#     locally, but never touches gitignored ones (data/, runs/, outputs/, logs/, the container image, ...).
#   - results/ and paper/ never go up: results/ is written on the cluster and comes back with cluster/fetch.sh;
#     the paper isn't needed there. Both are excluded, which also protects them from --delete.
#   - .env (secrets such as HF_TOKEN, gitignored) goes up separately, readable only by you.
#   - REVISION records the git commit and whether the tree had uncommitted changes (REVISION.patch holds them),
#     since .git itself stays here; every job copies it into its output manifest (src/cdm/provenance.py).
source "$(dirname "$0")/lib.sh"

DRY_RUN=()
[ "${1:-}" = "--dry-run" ] && DRY_RUN=(--dry-run --itemize-changes)

ensure_connection
ROOT="$(remote_root)"
cluster_ssh "mkdir -p $(printf '%q' "${ROOT}")"

# Provenance of this sync, written next to the code that is uploaded.
if git -C "${CDM_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
    {
        echo "commit=$(git -C "${CDM_ROOT}" rev-parse HEAD 2>/dev/null || echo none)"
        echo "describe=$(git -C "${CDM_ROOT}" describe --always --dirty 2>/dev/null || echo none)"
        # Untracked, non-ignored files are synced and used too, so they count as uncommitted changes.
        if [ -n "$(git -C "${CDM_ROOT}" status --porcelain)" ]; then
            echo "dirty=true"
        else
            echo "dirty=false"
        fi
        echo "synced_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "synced_from=$(hostname)"
    } >"${CDM_ROOT}/REVISION"
    {
        git -C "${CDM_ROOT}" diff HEAD 2>/dev/null || true
        git -C "${CDM_ROOT}" ls-files --others --exclude-standard -z | while IFS= read -r -d '' f; do
            git -C "${CDM_ROOT}" diff --no-index -- /dev/null "${f}" 2>/dev/null || true
        done
    } >"${CDM_ROOT}/REVISION.patch"
else
    echo "commit=none" >"${CDM_ROOT}/REVISION"
    : >"${CDM_ROOT}/REVISION.patch"
fi

rsync -az --delete ${DRY_RUN[@]+"${DRY_RUN[@]}"} \
    -e "$(rsync_ssh)" \
    --include='/REVISION' --include='/REVISION.patch' \
    --filter=':- .gitignore' \
    --exclude='.git/' \
    --exclude='/results/' \
    --exclude='/paper/' \
    --exclude='/cluster/.remote_root' \
    "${CDM_ROOT}/" "${CLUSTER_HOST}:${ROOT}/"

if [ -f "${CDM_ROOT}/.env" ] && [ "${#DRY_RUN[@]}" -eq 0 ]; then
    rsync -az --chmod=F600 -e "$(rsync_ssh)" "${CDM_ROOT}/.env" "${CLUSTER_HOST}:${ROOT}/.env"
fi

# File modes don't always survive the trip (e.g. files written by tools that don't set the executable bit).
# The shared directories (data, caches, outputs, results, logs, snapshots, the container image) live once in the
# project directory and are symlinked into this checkout's code directory, so all checkouts use the same data and
# write to the same results.csv (src/cdm/results_csv.py locks it). They are gitignored or excluded above, so
# rsync --delete never touches them.
if [ "${#DRY_RUN[@]}" -eq 0 ]; then
    SHARED="$(remote_shared_root)"
    if [ "${SHARED}" != "${ROOT}" ]; then
        remote_shell "set -e; S=$(printf '%q' "${SHARED}")
            mkdir -p \"\$S\"/{data,runs,outputs,results,logs/done,snapshots,checkpoints,container}
            for d in data runs outputs results logs snapshots checkpoints; do
                [ -L \"\$d\" ] || { [ -e \"\$d\" ] && rmdir \"\$d\" 2>/dev/null; true; }
                [ -e \"\$d\" ] && [ ! -L \"\$d\" ] && { echo \"sync: \$d is a real directory in this checkout, not a link to \$S/\$d; move its content there\" >&2; exit 1; }
                ln -sfn \"\$S/\$d\" \"\$d\"
            done
            [ -L container/cdm.sif ] || ln -sfn \"\$S/container/cdm.sif\" container/cdm.sif"
    fi
    remote_shell 'chmod +x slurm/*.sh container/*.sh 2>/dev/null || true; mkdir -p logs/done results'
fi

echo "Synced $(sed -n 's/^describe=//p' "${CDM_ROOT}/REVISION" 2>/dev/null || echo '?') to ${CLUSTER_HOST}:${ROOT}"

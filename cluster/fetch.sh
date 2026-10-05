#!/usr/bin/env bash
# Copies what the jobs produced from the cluster back to your machine. Never deletes anything locally.
#
# Usage: cluster/fetch.sh                    results/ (results.csv, per-query metrics, tables) -- the default
#        cluster/fetch.sh --logs             also logs/done/ (gitignored locally)
#        cluster/fetch.sh --path <relpath>   any other path under the remote repo root, e.g.
#                                            outputs/dl19/bm25/minilm-l6 (raw scores; can be large)
#        cluster/fetch.sh --dry-run ...      show what would be copied
#
# results/ is tracked in git: review `git diff results/` after fetching, then commit it.
source "$(dirname "$0")/lib.sh"
ensure_connection
ROOT="$(remote_root)"

PATHS=("results/")
EXTRA=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --logs) PATHS+=("logs/done/") ;;
        --path)
            [ -n "${2:-}" ] || die "--path needs an argument"
            [[ "$2" != /* && "$2" != *..* ]] || die "--path must be relative to the repo root"
            PATHS+=("${2%/}/")
            shift
            ;;
        --dry-run) EXTRA+=(--dry-run --itemize-changes) ;;
        *) die "unknown option $1" ;;
    esac
    shift
done

for p in "${PATHS[@]}"; do
    mkdir -p "${CDM_ROOT}/${p}"
    # *.lock: the advisory lock files of results.csv (src/cdm/file_lock.py), meaningless outside the cluster.
    rsync -az ${EXTRA[@]+"${EXTRA[@]}"} --exclude='*.lock' -e "$(rsync_ssh)" \
        "${CLUSTER_HOST}:${ROOT}/${p}" "${CDM_ROOT}/${p}"
    echo "Fetched ${p}"
done

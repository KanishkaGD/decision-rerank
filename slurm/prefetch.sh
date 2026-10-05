#!/usr/bin/env bash
# Downloads everything jobs need into the caches under data/, on the LOGIN node (compute nodes may have no
# internet, and the jobs run with HF_HUB_OFFLINE=1): Pyserini's prebuilt indexes, topics and qrels, and the
# Hugging Face models of the rerankers. Downloading is the kind of light, cluster-related work the login node is
# for. From your machine: cluster/login.sh bash slurm/prefetch.sh ...
#
# Usage: slurm/prefetch.sh [--datasets all|ds1,ds2] [--systems all|s1,s2]
#   (defaults: every dataset and system in slurm/config.sh; `all` adds the dev datasets too)
set -euo pipefail
CDM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${CDM_ROOT}/slurm/config.sh"
export HF_HUB_OFFLINE=0
source "${CDM_ROOT}/slurm/env.sh"

DATASET_LIST="$(IFS=,; echo "${DATASETS[*]},${DEV_DATASETS[*]}")"
SYSTEM_LIST="$(IFS=,; echo "${SYSTEMS[*]}")"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --datasets) [ "$2" = all ] || DATASET_LIST="$2"; shift ;;
        --systems) [ "$2" = all ] || SYSTEM_LIST="$2"; shift ;;
        *) echo "slurm/prefetch.sh: unknown option $1" >&2; exit 2 ;;
    esac
    shift
done

"${CONTAINER_CMD[@]}" cdm prefetch --datasets "${DATASET_LIST}" --systems "${SYSTEM_LIST}"

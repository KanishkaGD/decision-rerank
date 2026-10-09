#!/usr/bin/env bash
# Downloads everything jobs need into the caches under data/ (jobs run with HF_HUB_OFFLINE=1): Pyserini's prebuilt
# indexes, topics and qrels, the official rerank pools, and the rerankers' Hugging Face models.
#
# Where: small downloads (models, pools, the BEIR and MS MARCO v1 indexes) are light, cluster-related work the login
# node is for, so run this script there directly. The MS MARCO v2 passage index of DL 2021-2023 is 39 GB to download
# and unpack; fetch that one with the batch counterpart on a CPU node instead:
#   slurm/submit.sh prefetch --datasets dl21,dl22,dl23 --systems none     (slurm/prefetch.sbatch)
#
# Usage: slurm/prefetch.interactive.sh [--datasets all|none|ds1,ds2] [--systems all|none|s1,s2]
#                                     [--training-data src1,src2]   (fine-tuning data, ~4 GB: use the batch job)
#   (defaults: every dataset and system in slurm/config.sh, the dev datasets included)
set -euo pipefail
CDM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${CDM_ROOT}/slurm/config.sh"
export HF_HUB_OFFLINE=0
source "${CDM_ROOT}/slurm/env.sh"
prefetch_lists "$@"

"${CONTAINER_CMD[@]}" cdm prefetch --datasets "${PREFETCH_DATASETS}" --systems "${PREFETCH_SYSTEMS}" \
    --training-data "${PREFETCH_TRAINING_DATA}"

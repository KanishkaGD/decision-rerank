# Central, and only, place for the Slurm account, the dataset/system lists that slurm/submit.sh loops over, and the
# --time budget of every (job, dataset, system) combination. Source it, don't run it.
#
# No slurm/*.sbatch carries its own #SBATCH --time or --account: slurm/submit.sh computes both from here and
# passes them to sbatch. A direct `sbatch slurm/<job>.sbatch ...` gets your default account and the partition's
# default time limit unless you pass them yourself (see CLAUDE.md).

# Find yours with: sacctmgr show associations where user=$USER format=Account%20,User%20,Partition%20,QOS%30
SLURM_ACCOUNT="chair-ds"

# GPU requests of slurm/submit.sh (GPU jobs) and slurm/exec.sh --gpu. A memory-tier feature instead of a GPU type,
# so a job can land on any GPU with enough memory; it also keeps jobs off the old titanblack and the AMD rx_9060_xt
# nodes, which current PyTorch/CUDA wheels don't run on. Check the available features with
#   sinfo -N -o "%N %G %f"
# Memory tiers on FAITH (2026-10-06): 6, 8, 11, 16, 20, 24, 48, 80, 141 GiB. gpumem.48gib selects the A100-SXM,
# H200, L40 and L40S nodes (in the short partition: L40 and L40S); 24gib selects the same nodes.
# For latency measurements (plan: one fixed GPU), submit with a fixed type instead: --gres gpu:a100-sxm:1.
GPU_GRES="gpu:1"
GPU_CONSTRAINT="gpumem.48gib"

# The GPU memory feature for one system. The 27B models (Clef: ~55 GB of bf16 weights; Kev-27B: ~51 GB plus ~15 GB
# of batching buffers) need the A100-80GB/H200 nodes, which are only in the gpu partition (slurm/exec.sh needs
# --partition gpu for them). Qwen3-Reranker-8B in fp32 is ~32 GB of weights: 48 GB is enough.
gpu_constraint_for() {
    case "$1" in
        clef-isolated | clef-joint | kev-27b-*) echo "gpumem.80gib" ;;
        *) echo "${GPU_CONSTRAINT}" ;;
    esac
}

# Dataset names as registered in src/cdm/datasets.py (`cdm datasets` lists them with their Pyserini ids). "all" in
# `slurm/submit.sh ... --datasets all` means this list. Every dataset is reranked on the BM25 top-100
# (docs/roadmap.md, "Setup").
DATASETS=(
    "dl19"       # TREC DL 2019 passage, MS MARCO v1
    "dl20"       # TREC DL 2020 passage, MS MARCO v1
    "dl21"       # TREC DL 2021-2023 passage, MS MARCO v2 (138M passages); official top-100 files = BM25 top-100
    "dl22"
    "dl23"
    "trec-covid" # BEIR
    "nfcorpus"
    "scifact"
    "touche"     # BEIR webis-touche2020
    "dbpedia"    # BEIR dbpedia-entity
)
DEV_DATASETS=(
    "msmarco-dev" # MS MARCO passage dev (small), 6,980 queries: fine-tuning (docs/roadmap.md); never reported
)

# Reranker names as registered in src/cdm/rerankers/__init__.py (`cdm systems` lists them).
SYSTEMS=(
    "bge-m3"             # cross-encoders: S1Rank's two baselines and Qwen3-Reranker-8B
    "monot5-3b"
    "qwen3-reranker-8b"
    "set-encoder-large"  # cross-encoder that scores the candidates jointly (own container, container_for)
    "qwen3.7-flash"      # LLM judge via OpenRouter (API_SYSTEMS)
    # System One models: <name>-isolated (one candidate per request), <name>-joint (all candidates in one state)
    "jev-isolated"       # TypeSafe API; runs on CPU (API_SYSTEMS)
    "jev-joint"
    "laya"               # open weights, isolated only (its context is too short for a list)
    "clef-flash-isolated" # Cloudflare, 9B
    "clef-flash-joint"
    "clef-isolated"      # Cloudflare, 27B: needs an 80 GB GPU (gpu_constraint_for)
    "clef-joint"
    "kev-4b-isolated"    # open weights, Jev-like (github.com/jaredpalmer/kev)
    "kev-4b-joint"
    "kev-27b-isolated"   # needs an 80 GB GPU (gpu_constraint_for)
    "kev-27b-joint"
)
# Systems served by an external API: slurm/submit.sh runs them on the general (CPU) partition, no GPU.
API_SYSTEMS=("jev-isolated" "jev-joint" "qwen3.7-flash")

# Grid cells we take from published papers instead of computing (results/literature.csv; docs/roadmap.md):
# slurm/submit.sh skips them unless --include-literature. monoT5-3B is still computed on DL19/20, as the check that
# our candidate sets reproduce the published numbers, and for its probabilities (calibration, RQ3).
#   Usage: from_literature <system> <dataset>   (exit status 0 = taken from the literature)
from_literature() {
    case "$1:$2" in
        bge-m3:dl19 | bge-m3:dl20 | bge-m3:trec-covid | bge-m3:nfcorpus | bge-m3:scifact | bge-m3:touche | \
            bge-m3:dbpedia) return 0 ;;
        monot5-3b:trec-covid | monot5-3b:nfcorpus | monot5-3b:scifact | monot5-3b:touche | monot5-3b:dbpedia) return 0 ;;
        set-encoder-large:dl19 | set-encoder-large:dl20) return 0 ;;
        *) return 1 ;;
    esac
}

# The container image a system runs in (relative to the repo root). Set-Encoder needs lightning-ir, which needs
# transformers < 5, while Laya, Clef and Kev need transformers >= 5.17: it gets its own image.
#   Usage: container_for <system>
container_for() {
    case "$1" in
        set-encoder-*) echo "container/lightning-ir.sif" ;;
        *) echo "container/cdm.sif" ;;
    esac
}

# --time budgets, from measured run times plus a margin; update them with a comment citing the job id.
# Partition maxima: general and gpu 48h, short 1h.
first_stage_time_for() {
    local dataset="$1"
    case "${dataset}" in
        msmarco-dev) echo "02:00:00" ;; # 6,980 queries
        *) echo "01:00:00" ;; # DL 2021-2023 official pools: 4-5 min (jobs 4357123-4357125)
    esac
}

rerank_time_for() {
    local dataset="$1" system="$2"
    case "${dataset}:${system}" in
        # 100 candidates for <= 82 queries (DL19-23, TREC-COVID, Touche). Measured on DL21-23: Kev-27B isolated
        # 29-44 min (4357189-4357191), Clef 27B 7-18 min (4357165-4357170), everything else under 10 min.
        dl*:clef-isolated | dl*:clef-joint | dl*:kev-27b-* | trec-covid:clef-* | trec-covid:kev-27b-* | \
            touche:clef-* | touche:kev-27b-*) echo "01:30:00" ;;
        dl* | trec-covid:* | touche:*) echo "00:45:00" ;;
        # 300-400 queries (NFCorpus, SciFact, DBPedia): about 5x, unmeasured.
        *:clef-isolated | *:clef-joint | *:kev-27b-*) echo "06:00:00" ;;
        msmarco-dev:*) echo "08:00:00" ;; # ~700k query-passage pairs at depth 100 (pass --max-queries)
        *) echo "03:00:00" ;;
    esac
}

# Prints the --time budget for one job.  Usage: time_for <job> [dataset] [system]
time_for() {
    case "$1" in
        hello) echo "00:02:00" ;; # job 4345829: 49 s
        prefetch) echo "06:00:00" ;; # MS MARCO v2 index and 23 models: 52 min until a failure (4356461)
        first-stage) first_stage_time_for "$2" ;;
        rerank) rerank_time_for "$2" "$3" ;;
        *)
            echo "slurm/config.sh: no --time budget for job '$1'" >&2
            return 1
            ;;
    esac
}

# The --datasets/--systems arguments of slurm/prefetch.{sbatch,interactive.sh} -> PREFETCH_DATASETS and
# PREFETCH_SYSTEMS, comma-separated as `cdm prefetch` takes them. all (the default) = DATASETS + DEV_DATASETS and
# SYSTEMS above; none = nothing.  Usage: prefetch_lists "$@"
prefetch_lists() {
    PREFETCH_DATASETS="$(IFS=,; echo "${DATASETS[*]},${DEV_DATASETS[*]}")"
    PREFETCH_SYSTEMS="$(IFS=,; echo "${SYSTEMS[*]}")"
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --datasets) PREFETCH_DATASETS="$2"; shift ;;
            --systems) PREFETCH_SYSTEMS="$2"; shift ;;
            *) echo "prefetch: unknown option $1" >&2; return 2 ;;
        esac
        shift
    done
    [ "${PREFETCH_DATASETS}" = all ] && PREFETCH_DATASETS="$(IFS=,; echo "${DATASETS[*]},${DEV_DATASETS[*]}")"
    [ "${PREFETCH_SYSTEMS}" = all ] && PREFETCH_SYSTEMS="$(IFS=,; echo "${SYSTEMS[*]}")"
    [ "${PREFETCH_DATASETS}" = none ] && PREFETCH_DATASETS=""
    [ "${PREFETCH_SYSTEMS}" = none ] && PREFETCH_SYSTEMS=""
    return 0
}

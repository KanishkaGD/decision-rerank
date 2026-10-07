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
        clef-pointwise | clef-listwise | kev-27b-*) echo "gpumem.80gib" ;;
        *) echo "${GPU_CONSTRAINT}" ;;
    esac
}

# Dataset names as registered in src/cdm/datasets.py (`cdm datasets` lists them with their Pyserini ids).
# "all" in `slurm/submit.sh ... --datasets all` means this list. msmarco-dev is the dev split for selecting the
# question phrasing (WP3) and is never reported as a test result.
# Candidates: the official rerank pool where there is one (TREC DL, MS MARCO dev; up to 1000 per query), else our
# BM25 top-100 (BEIR) -- `cdm datasets` shows which.
DATASETS=(
    "dl19"       # TREC DL 2019 passage (in-domain), official pool
    "dl20"       # TREC DL 2020 passage (in-domain), official pool
    "dl21"       # TREC DL 2021-2023 passage: MS MARCO v2 (138M passages), official top-100 pools; prefetch the
    "dl22"       #   39 GB msmarco-v2-passage index with a batch job (slurm/prefetch.sbatch)
    "dl23"
    "trec-covid" # BEIR
    "nfcorpus"   # BEIR
    "scifact"    # BEIR, small: the sanity-check dataset
    "fiqa"       # BEIR
    "scidocs"    # BEIR
)
DEV_DATASETS=(
    "msmarco-dev" # MS MARCO passage dev (small), 6,980 queries, official top1000.dev pool
)

# Reranker names as registered in src/cdm/rerankers/__init__.py (`cdm systems` lists them).
SYSTEMS=(
    # pointwise cross-encoders
    "minilm-l6"
    "bge-m3"
    "mxbai-base"
    "monot5-base"          # Kanishka's baselines, and their larger sizes for the 48/80 GB GPUs
    "monot5-3b"
    "qwen3-reranker-0.6b"
    "qwen3-reranker-4b"
    "qwen3-reranker-8b"
    # listwise cross-encoder (own container, container_for; at most 100 candidates: pass --depth 100 on 1000-pools)
    "set-encoder-base"
    "set-encoder-large"
    # pointwise LLM judge via OpenRouter (API_SYSTEMS)
    "qwen3.7-flash"
    # System One models: <name>-pointwise (one candidate per request) and <name>-listwise (all in one state)
    "jev-pointwise"        # TypeSafe API; runs on CPU (API_SYSTEMS)
    "jev-listwise"
    "laya"                 # open weights, pointwise only (its context is too short for a list)
    "laya-multilingual"
    "clef-flash-pointwise" # Cloudflare, 9B
    "clef-flash-listwise"
    "clef-pointwise"       # Cloudflare, 27B: needs an 80 GB GPU (gpu_constraint_for)
    "clef-listwise"
    "kev-0.8b-pointwise"   # open weights, Jev-like (github.com/jaredpalmer/kev)
    "kev-0.8b-listwise"
    "kev-4b-pointwise"
    "kev-4b-listwise"
    "kev-9b-pointwise"
    "kev-9b-listwise"
    "kev-27b-pointwise"    # needs an 80 GB GPU (gpu_constraint_for)
    "kev-27b-listwise"
)
# Systems served by an external API: slurm/submit.sh runs them on the general (CPU) partition, no GPU.
API_SYSTEMS=("jev-pointwise" "jev-listwise" "qwen3.7-flash")

# The container image a system runs in (relative to the repo root). Set-Encoder needs lightning-ir, which needs
# transformers < 5, while Laya, Clef and Kev need transformers >= 5.17: it gets its own image.
#   Usage: container_for <system>
container_for() {
    case "$1" in
        set-encoder-*) echo "container/lightning-ir.sif" ;;
        *) echo "container/cdm.sif" ;;
    esac
}

# --time budgets. Unvalidated first guesses: replace each with the observed run time plus a margin once a job's
# log (or `sacct -j <jobid>`) shows it, and say so in a comment. Partition maxima: general 48h, gpu 28h,
# short 1h.
first_stage_time_for() {
    local dataset="$1"
    case "${dataset}" in
        msmarco-dev) echo "04:00:00" ;; # 6,980 queries; the official pool: ~7M pairs to read and order
        *) echo "01:00:00" ;;
    esac
}

rerank_time_for() {
    local dataset="$1" system="$2"
    case "${dataset}:${system}" in
        msmarco-dev:jev-pointwise) echo "16:00:00" ;; # 100 API requests per query, ~16 in parallel
        msmarco-dev:*) echo "08:00:00" ;; # ~700k query-passage pairs at depth 100 (pass --max-queries / --depth)
        # The official DL pools: 41k (DL19) / 50k (DL20) pairs. Kanishka's runs: Kev-0.8B ~8 min for both on an A100;
        # Qwen3.7-Flash ~64 pairs/s through OpenRouter (~15 min per year); Jev 101k requests.
        # 27B models on all 41k/50k pairs: unmeasured, generous (gpu partition max 48h).
        dl19:*-27b* | dl20:*-27b* | dl19:clef-pointwise | dl20:clef-pointwise | dl19:clef-listwise | \
            dl20:clef-listwise) echo "16:00:00" ;;
        dl19:clef-flash* | dl20:clef-flash*) echo "08:00:00" ;;
        dl19:qwen3-reranker-8b | dl20:qwen3-reranker-8b) echo "08:00:00" ;;
        dl19:* | dl20:*) echo "04:00:00" ;;
        # The official DL 2021-2023 pools: top-100 for 53 / 76 / 82 judged queries (5k-8k pairs). Unmeasured.
        dl2[123]:*-27b* | dl2[123]:clef-pointwise | dl2[123]:clef-listwise) echo "04:00:00" ;;
        dl2[123]:*) echo "02:00:00" ;;
        scidocs:* | fiqa:*) echo "03:00:00" ;; # 1,000 / 648 queries
        *) echo "01:00:00" ;;
    esac
}

# Prints the --time budget for one job.  Usage: time_for <job> [dataset] [system]
time_for() {
    case "$1" in
        hello) echo "00:02:00" ;; # job 4345829: 49 s
        prefetch) echo "06:00:00" ;; # unmeasured; the MS MARCO v2 passage index is 39 GB to download and unpack
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

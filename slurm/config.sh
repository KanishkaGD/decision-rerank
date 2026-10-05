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
#   cluster/login.sh sinfo -N -o "%N %G %f"
# For latency measurements (plan: one fixed GPU), submit with a fixed type instead: --gres gpu:a100-sxm:1.
GPU_GRES="gpu:1"
GPU_CONSTRAINT="gpumem.40gib"

# Dataset names as registered in src/cdm/datasets.py (`cdm datasets` lists them with their Pyserini ids).
# "all" in `slurm/submit.sh ... --datasets all` means this list. msmarco-dev is the dev split for selecting the
# question phrasing (WP3) and is never reported as a test result.
DATASETS=(
    "dl19"       # TREC DL 2019 passage (in-domain)
    "dl20"       # TREC DL 2020 passage (in-domain)
    "trec-covid" # BEIR
    "nfcorpus"   # BEIR
    "scifact"    # BEIR, small: the sanity-check dataset
    "fiqa"       # BEIR
    "scidocs"    # BEIR
)
DEV_DATASETS=(
    "msmarco-dev" # MS MARCO passage dev (small), 6,980 queries
)

# Reranker names as registered in src/cdm/rerankers/__init__.py (`cdm systems` lists them).
SYSTEMS=(
    "minilm-l6"
    "bge-m3"
    "mxbai-base"
    "laya"  # stub until WP0 confirms the API
    "jev"   # stub, needs API access
)

# --time budgets. Unvalidated first guesses: replace each with the observed run time plus a margin once a job's
# log (or `cluster/status.sh <jobid>`) shows it, and say so in a comment. Partition maxima: general 48h, gpu 28h,
# short 1h.
first_stage_time_for() {
    local dataset="$1"
    case "${dataset}" in
        msmarco-dev) echo "02:00:00" ;; # 6,980 queries
        *) echo "01:00:00" ;;
    esac
}

rerank_time_for() {
    local dataset="$1" system="$2"
    case "${dataset}:${system}" in
        msmarco-dev:*) echo "08:00:00" ;; # ~700k query-passage pairs at depth 100
        scidocs:* | fiqa:*) echo "03:00:00" ;; # 1,000 / 648 queries
        *) echo "01:00:00" ;;
    esac
}

# Prints the --time budget for one job.  Usage: time_for <job> [dataset] [system]
time_for() {
    case "$1" in
        hello) echo "00:02:00" ;;
        first-stage) first_stage_time_for "$2" ;;
        rerank) rerank_time_for "$2" "$3" ;;
        *)
            echo "slurm/config.sh: no --time budget for job '$1'" >&2
            return 1
            ;;
    esac
}

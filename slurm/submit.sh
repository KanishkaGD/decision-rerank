#!/usr/bin/env bash
# The entry point for batch jobs: one independent `sbatch` call per dataset, each with --account and --time from
# slurm/config.sh (the only place they are set). No array jobs: every dataset is its own job, queued and
# time-limited separately. Run it on the login node, from the repo root or anywhere else.
#
# Usage: slurm/submit.sh hello
#        slurm/submit.sh prefetch [--datasets ...] [--systems ...]   (downloads on a CPU node: slurm/prefetch.sbatch)
#        slurm/submit.sh first-stage [--datasets all|ds1,ds2,...] [options] [extra cdm args]
#        slurm/submit.sh rerank <system> [--datasets all|ds1,ds2,...] [options] [extra cdm args]
#        slurm/submit.sh --list | --help
#
# Options handled here (anything else is passed to `cdm` unchanged, e.g. --depth 50 or --phrasings p1,p2):
#   --datasets all|ds1,...   default all = DATASETS in slurm/config.sh; dev datasets (msmarco-dev) only by name
#   --after <id>[,<id>...]   start only after those jobs succeeded (sbatch --dependency=afterok), e.g. rerank
#                            after the first-stage job that caches its candidates
#   --cpu                    rerank on the general (CPU) partition, no GPU
#   --gres <spec>            rerank on a specific GPU instead of GPU_CONSTRAINT, e.g. gpu:a100-sxm:1 (latency runs)
#   --time <hh:mm:ss>        override config.sh's budget (then fix config.sh if the budget was wrong)
#   --dry-run                print the sbatch commands instead of running them
set -euo pipefail

CDM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${CDM_ROOT}" # #SBATCH --output and SLURM_SUBMIT_DIR are relative to the submission directory
source "${CDM_ROOT}/slurm/config.sh"
mkdir -p logs/done

# Printed at the top of every job's log (slurm/log_command.sh): sbatch passes the environment on to the job.
export CDM_SUBMIT_COMMAND="slurm/submit.sh$( [ "$#" -gt 0 ] && printf ' %q' "$@" )"

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; }

case "${1:-}" in
    "" | -h | --help) usage; exit 0 ;;
    --list)
        echo "Datasets:     ${DATASETS[*]}"
        echo "Dev datasets: ${DEV_DATASETS[*]}"
        echo "Systems:      ${SYSTEMS[*]}"
        exit 0
        ;;
esac

JOB="$1"
shift
SYSTEM=""
case "${JOB}" in
    hello | first-stage | prefetch) ;;
    rerank)
        SYSTEM="${1:?Usage: slurm/submit.sh rerank <system> [--datasets ...] (systems: ${SYSTEMS[*]})}"
        shift
        ;;
    *) echo "slurm/submit.sh: unknown job '${JOB}' (hello, prefetch, first-stage, rerank)" >&2; exit 2 ;;
esac

DATASET_LIST=("${DATASETS[@]}")
DATASETS_ARG=""
AFTER=""
CPU=0
GRES=""
TIME_OVERRIDE=""
DRY_RUN=0
PASSTHROUGH=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --datasets)
            DATASETS_ARG="$2"
            [ "$2" = "all" ] || IFS=',' read -r -a DATASET_LIST <<<"$2"
            shift
            ;;
        --after) AFTER="$2"; shift ;;
        --cpu) CPU=1 ;;
        --gres) GRES="$2"; shift ;;
        --time) TIME_OVERRIDE="$2"; shift ;;
        --dry-run) DRY_RUN=1 ;;
        *) PASSTHROUGH+=("$1") ;;
    esac
    shift
done

submit() {
    if [ "${DRY_RUN}" -eq 1 ]; then
        printf '%q ' sbatch "$@"
        echo
    else
        sbatch "$@"
    fi
}

# Freeze the code for these jobs (slurm/snapshot.sh): a queued job starts later, maybe after src/ has changed again.
if [ "${DRY_RUN}" -eq 0 ]; then
    source "${CDM_ROOT}/slurm/snapshot.sh"
    snapshot_code
fi

COMMON=(--account="${SLURM_ACCOUNT}")
[ -n "${AFTER}" ] && COMMON+=(--dependency="afterok:${AFTER//,/:}")

if [ "${JOB}" = "hello" ]; then
    submit "${COMMON[@]}" --time="${TIME_OVERRIDE:-$(time_for hello)}" slurm/hello.sbatch
    exit 0
fi
if [ "${JOB}" = "prefetch" ]; then
    # One job for all downloads: they share the caches, and the big ones (indexes) dominate anyway.
    submit "${COMMON[@]}" --time="${TIME_OVERRIDE:-$(time_for prefetch)}" slurm/prefetch.sbatch \
        ${DATASETS_ARG:+--datasets "${DATASETS_ARG}"} "${PASSTHROUGH[@]}"
    exit 0
fi

for dataset in "${DATASET_LIST[@]}"; do
    case "${JOB}" in
        first-stage)
            submit "${COMMON[@]}" --time="${TIME_OVERRIDE:-$(time_for first-stage "${dataset}")}" \
                slurm/first-stage.sbatch "${dataset}" "${PASSTHROUGH[@]}"
            ;;
        rerank)
            RESOURCES=()
            if [ "${CPU}" -eq 1 ] || [[ " ${API_SYSTEMS[*]} " == *" ${SYSTEM} "* ]]; then
                RESOURCES=(--partition=general)
            elif [ -n "${GRES}" ]; then
                RESOURCES=(--gres="${GRES}")
            else
                RESOURCES=(--gres="${GPU_GRES}" --constraint="$(gpu_constraint_for "${SYSTEM}")")
            fi
            submit "${COMMON[@]}" "${RESOURCES[@]}" \
                --time="${TIME_OVERRIDE:-$(time_for rerank "${dataset}" "${SYSTEM}")}" \
                slurm/rerank.sbatch "${SYSTEM}" "${dataset}" "${PASSTHROUGH[@]}"
            ;;
    esac
done

# Source this, don't run it -- with CDM_ROOT already set to the repo root on /shared:
#   CDM_ROOT="${SLURM_SUBMIT_DIR}"                       (in a .sbatch: $0 is Slurm's spooled copy there)
#   CDM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"         (everywhere else)
#   source "${CDM_ROOT}/slurm/env.sh"
#
# The one place that defines where data, caches and outputs live and how the container is called, shared by every
# .sbatch, its .interactive.sh counterpart, slurm/exec.sh and slurm/prefetch.sh -- so they can't drift apart.
# Afterwards, run a command in the container with:  "${CONTAINER_CMD[@]}" cdm <subcommand> ...

: "${CDM_ROOT:?set CDM_ROOT to the repo root before sourcing slurm/env.sh}"

CONTAINER="${CDM_ROOT}/container/cdm.sif"
[ -f "${CONTAINER}" ] || {
    echo "slurm/env.sh: ${CONTAINER} is missing -- build it first: cluster/login.sh bash container/build.sh" >&2
    return 1 2>/dev/null || exit 1
}

# Everything persistent lives on /shared under the repo root (gitignored). Caches are filled by slurm/prefetch.sh on
# the login node, since compute nodes may not reach the internet.
export CDM_DATA="${CDM_ROOT}/data"
export HF_HOME="${CDM_DATA}/hf_cache"
export PYSERINI_CACHE="${CDM_DATA}/pyserini"
export XDG_CACHE_HOME="${CDM_DATA}/cache"
export CDM_RUNS="${CDM_ROOT}/runs"        # cached first-stage candidates (src/cdm/runs.py)
export CDM_OUTPUTS="${CDM_ROOT}/outputs"  # raw reranker scores, per-query metrics, manifests
export CDM_RESULTS="${CDM_ROOT}/results"  # results.csv (tracked in git, fetched with cluster/fetch.sh)
# Jobs only read the caches: offline mode makes a model or dataset missing from them fail fast with a clear error,
# instead of hanging on a download a compute node can't make. slurm/prefetch.sh sets HF_HUB_OFFLINE=0.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "${HF_HOME}" "${PYSERINI_CACHE}" "${XDG_CACHE_HOME}" "${CDM_RUNS}" "${CDM_OUTPUTS}" "${CDM_RESULTS}"

# HF_TOKEN, API keys etc. from .env (gitignored; cluster/sync.sh copies it up with mode 600). Passed as a file, not
# as --env KEY=value arguments, so the secrets don't show up in the process list of a shared compute node.
ENV_ARGS=()
if [ -f "${CDM_ROOT}/.env" ]; then
    ENV_ARGS=(--env-file "${CDM_ROOT}/.env")
fi

# GPU only when Slurm gave us one; --nv on a CPU node would just print warnings.
NV_ARGS=()
if [ -n "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}" ] || { [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "NoDevFileFound" ]; }; then
    NV_ARGS=(--nv)
fi

# Apptainer passes the host environment through by default; the --env list makes the variables the code depends
# on explicit. Binding the repo root at the same path keeps every path identical inside and outside.
# PYTHONPATH puts the repo's src/ ahead of the copy installed in the image: code changes take effect with the next
# cluster/sync.sh, and the container only needs a rebuild when dependencies (pyproject.toml) change. Batch jobs use
# the snapshot slurm/submit.sh made at submission time (CDM_CODE_DIR) instead of the live src/.
CDM_CODE_DIR="${CDM_CODE_DIR:-${CDM_ROOT}}"
CONTAINER_CMD=(
    apptainer exec "${NV_ARGS[@]}"
    --bind "${CDM_ROOT}:${CDM_ROOT}"
    --pwd "${CDM_ROOT}"
    --env "PYTHONPATH=${CDM_CODE_DIR}/src,CDM_CODE_DIR=${CDM_CODE_DIR},CDM_ROOT=${CDM_ROOT},CDM_DATA=${CDM_DATA},HF_HOME=${HF_HOME},PYSERINI_CACHE=${PYSERINI_CACHE},XDG_CACHE_HOME=${XDG_CACHE_HOME},CDM_RUNS=${CDM_RUNS},CDM_OUTPUTS=${CDM_OUTPUTS},CDM_RESULTS=${CDM_RESULTS},HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
    "${ENV_ARGS[@]}"
    "${CONTAINER}"
)

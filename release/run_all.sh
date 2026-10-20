#!/usr/bin/env bash
# Every step behind the paper's numbers, in order. Each step is idempotent: `cdm rerank` skips a configuration that
# results/results.csv already has (pass --overwrite to recompute), so the script can be resumed after an interruption.
#
#   bash run_all.sh [step ...]        steps: prefetch first-stage finetune rerank analyses tables (default: all)
#
# Commands run directly when `cdm` is importable from src/ (a Python environment with pyproject.toml's dependencies),
# or inside the Apptainer images when CDM_CONTAINER=1 (container/cdm.sif, container/lightning-ir.sif for the
# Set-Encoder). On a cluster, run the steps -- or the single `cdm rerank` calls -- as batch jobs; the commands are the
# same. GPUs: 48 GB suffice for most systems; the 27B models, the 8B/9B models in fp32 and the fine-tune (2 GPUs) need
# 80 GB. Jev runs through the TypeSafe API (TYPESAFE_API_KEY, see .env.example) and needs no GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT}"
[ -f .env ] && set -a && source .env && set +a

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CDM_ROOT="${ROOT}" CDM_DATA="${CDM_DATA:-${ROOT}/data}" CDM_RUNS="${CDM_RUNS:-${ROOT}/runs}"
export CDM_OUTPUTS="${CDM_OUTPUTS:-${ROOT}/outputs}" CDM_RESULTS="${CDM_RESULTS:-${ROOT}/results}"
export HF_HOME="${HF_HOME:-${CDM_DATA}/hf_cache}" PYSERINI_CACHE="${PYSERINI_CACHE:-${CDM_DATA}/pyserini}"

TEST="dl19 dl20 dl21 dl22 dl23 trec-covid nfcorpus scifact touche dbpedia"
DL="dl19 dl20 dl21 dl22 dl23"
# The triplets (base model | decision fine-tunes, joint and isolated | reranker fine-tunes), then the other systems.
TRIPLETS="qwen3-0.6b-prompted kev-0.6b-joint kev-0.6b-isolated qwen3-reranker-0.6b
          qwen3-8b-prompted kev-8b-joint kev-8b-isolated qwen3-reranker-8b
          qwen3.5-0.8b-prompted kev-0.8b-joint kev-0.8b-isolated prism-0.8b
          qwen3.5-4b-prompted kev-4b-joint kev-4b-isolated prism-4b
          qwen3.5-9b-prompted kev-9b-joint kev-9b-isolated clef-flash-joint clef-flash-isolated prism-9b
          qwen3.5-9b-reranker
          qwen3.8-27b-prompted kev-27b-joint kev-27b-isolated clef-joint clef-isolated"
OTHERS="bge-m3 monot5-3b set-encoder-large jev-joint jev-isolated laya"
# Rerankers whose probabilities are scaled after the fact; fitted on 1,000 MS MARCO dev queries.
POSTHOC="qwen3-reranker-0.6b qwen3-reranker-8b prism-0.8b prism-4b prism-9b monot5-3b bge-m3 set-encoder-large"
FINETUNED="qwen3.5-9b-reranker"

cdm() {
    local image="container/cdm.sif"
    case " $* " in *" set-encoder-"*) image="container/lightning-ir.sif" ;; esac
    if [ "${CDM_CONTAINER:-0}" = 1 ]; then
        apptainer exec --nv --env "PYTHONPATH=${PYTHONPATH}" "${image}" cdm "$@"
    else
        python -m cdm.cli "$@"
    fi
}
torchrun_cdm() {
    if [ "${CDM_CONTAINER:-0}" = 1 ]; then
        apptainer exec --nv --env "PYTHONPATH=${PYTHONPATH}" container/cdm.sif torchrun "$@" --no-python cdm finetune \
            "${FINETUNE_ARGS[@]}"
    else
        torchrun "$@" --no-python python -m cdm.cli finetune "${FINETUNE_ARGS[@]}"
    fi
}

step_prefetch() {
    # Needs internet; everything after it runs offline. MS MARCO v2's passage index (DL21-23) is 39 GB.
    cdm prefetch --training-data tevatron-aug,tevatron-bm25
}

step_first_stage() {
    export HF_HUB_OFFLINE=1
    for d in ${TEST}; do cdm first-stage "${d}"; done
    # The dev candidates for post-hoc calibration and checkpoint selection: BM25 top-100 of the first 1,000 queries.
    cdm first-stage msmarco-dev --max-queries 1000
}

step_finetune() {
    # Ours-9B: a LoRA adapter on Qwen3.5-9B, 100k MS MARCO training queries, 1 positive + 15 negatives each, listwise
    # and pointwise loss, one epoch on 2 GPUs (bf16). Checkpoints every 200 steps under checkpoints/qwen3.5-9b-reranker/.
    export HF_HUB_OFFLINE=1
    cdm finetune-data msmarco-mix-100k --queries 100000
    FINETUNE_ARGS=("${FINETUNED}" msmarco-mix-100k --loss both)
    torchrun_cdm --standalone --nproc_per_node=2
    # Checkpoint selection on 200 msmarco-dev queries: keep `final` unless an earlier checkpoint is more than 1 point
    # better on MRR@10 (it was: checkpoint-800, set in SELECTED_CHECKPOINT in src/cdm/rerankers/__init__.py).
    for ckpt in checkpoints/${FINETUNED}/checkpoint-*; do
        n="${ckpt##*-}"
        cdm rerank "${FINETUNED}" msmarco-dev --first-stage bm25-d100-max1000 --max-queries 200 \
            --adapter "${ckpt}" --variant "ckpt${n}"
    done
}

step_rerank() {
    export HF_HUB_OFFLINE=1
    for s in ${TRIPLETS} ${OTHERS}; do
        for d in ${TEST}; do cdm rerank "${s}" "${d}"; done
    done
    # RankZephyr on TREC DL (BEIR and RankGPT-4 are taken from the literature: results/literature.csv).
    for d in ${DL}; do cdm rerank rankzephyr-7b "${d}"; done
    for s in ${POSTHOC}; do cdm rerank "${s}" msmarco-dev --first-stage bm25-d100-max1000; done
}

step_analyses() {
    # Read the stored outputs only (CPU): significance (RQ1), the fixed cutoff and count test, post-hoc scaling (RQ2).
    cdm significance
    cdm cutoff
    cdm posthoc --systems "$(echo ${POSTHOC} | tr ' ' ,)"
}

step_tables() {
    python -m cdm.paper_tables
}

STEPS=("$@")
[ "${#STEPS[@]}" -gt 0 ] || STEPS=(prefetch first-stage finetune rerank analyses tables)
for step in "${STEPS[@]}"; do
    echo "== ${step}"
    "step_${step//-/_}"
done

# calibrated-decision-models-for-ir

Experiments and paper for *Decide, then Rank: Calibrated System One Models as Rerankers* (working title). The plan
-- research questions, setup and order of work -- is [docs/roadmap.md](docs/roadmap.md).

## Layout

The repository lives on the FAITH cluster's shared storage (`/shared/<user>/...`, visible to login and compute
nodes) and is used from a login node (`freia`/`frigga`). Jobs read and write next to the code.

```
slurm/          Slurm jobs (.sbatch + .interactive.sh pairs), submit.sh, exec.sh, config.sh, env.sh, snapshot.sh
container/      Apptainer image with the dependencies (cdm.def, build.sh)
src/cdm/        the Python package and `cdm` CLI: first stage, rerankers, evaluation, results.csv
tests/          pytest
results/        results.csv and generated tables -- tracked in git, written by the jobs in place
paper/          LaTeX sources (LNCS), built with `make`
docs/           research plan (owner's, read-only), experiment log, notes, intent statements
```

Gitignored, next to the code: `data/` (official pools, Pyserini and Hugging Face caches), `runs/` (cached
first-stage candidates), `outputs/` (raw reranker scores), `logs/`, `snapshots/` (the code of each submission),
`container/cdm.sif`.

**The login node is not for computation** (FAITH docs): only light, cluster-related commands run there -- editing,
git, `sbatch`/`squeue`/`sacct`, building the container, small downloads. Everything else goes through Slurm
(`slurm/exec.sh`, `slurm/submit.sh`). Coding agents may work here only in a session you supervise (CLAUDE.md).

## Setup, once

On the login node, in the repo root (and the container again whenever `pyproject.toml` or `container/cdm.def`
changes):

```bash
cp .env.example .env && chmod 600 .env             # HF_TOKEN, TYPESAFE_AI_API_KEY, OPENROUTER_API_KEY
bash container/build.sh                           # container/cdm.sif (fakeroot, ~15 min)
bash container/build.sh lightning-ir              # container/lightning-ir.sif, for the Set-Encoder only
bash slurm/prefetch.interactive.sh --datasets dl19,dl20,scifact --systems bge-m3,monot5-3b
                                                  # indexes, topics/qrels, official pools, models -> data/
slurm/submit.sh prefetch --datasets dl21,dl22,dl23 --systems none
                                                  # the 39 GB MS MARCO v2 passage index: a CPU batch job
slurm/submit.sh hello                             # sanity check: Slurm, logging, container
slurm/exec.sh --gpu -- cdm info                   # ... and a GPU inside the container
```

Unit tests (light, fine on the login node): `pytest` in any Python 3.10-3.13 environment with `click numpy pandas
ir-measures pytest`, or in the container on a compute node:
`slurm/exec.sh -- bash -c 'PYTHONPATH=$PYTHONPATH:$CDM_ROOT/data/devtools python -m pytest -q tests'`.

## Running experiments

From the login node, in the repo root. `slurm/submit.sh` and `slurm/exec.sh` snapshot `src/` first, so you can keep
editing while jobs wait in the queue.

```bash
# Quick, blocking check on a compute node (short partition, <= 1h):
slurm/exec.sh -- cdm info
slurm/exec.sh -- cdm first-stage scifact --max-queries 20
slurm/exec.sh --gpu -- cdm rerank bge-m3 scifact --first-stage bm25-d100-max20 --no-save

# Batch jobs: one job per dataset, --time and --account from slurm/config.sh
slurm/submit.sh first-stage --datasets all                       # -> Submitted batch job 101 ... 110
slurm/submit.sh rerank monot5-3b --datasets dl19,dl20 --after 101,102
slurm/submit.sh rerank kev-27b-joint --datasets dl21,dl22,dl23 --phrasings s1rank --dry-run   # print only

# Follow them
squeue --me
sacct -j 101,102 --format=JobID,JobName%28,State,Elapsed,Timelimit,ExitCode,MaxRSS,NodeList
tail -f logs/101_*.out                # logs/<jobid>_<job>_<args>_<ts>.out; logs/done/ once finished

# Results: written to results/results.csv in place
cdm results --metric nDCG@10          # system x dataset table (slurm/exec.sh -- cdm results ... without a local env)
git diff results/ && git commit results/
```

What the jobs produce:

| Path | Content |
| --- | --- |
| `runs/<dataset>/bm25-d100/`, `official-d100/` (DL21-23) | first-stage cache, the BM25 top-100: `candidates.jsonl.gz` (with passage texts), `run.trec.gz`, `qrels.tsv`, `manifest.json` |
| `outputs/<dataset>/<first-stage>/<system>/<phrasing>/` | `scores.jsonl.gz` (score and probability per query and passage), `responses.jsonl.gz` (API systems: every raw response with tokens and latency), `run.trec.gz`, `perquery.csv`, `manifest.json` |
| `results/results.csv` | one row per configuration: nDCG@10, R@10, R@100, RR@10, AP, ECE, Brier, latency p50/p95, truncation, API requests/tokens/cost (API systems), provenance |

Every manifest records the command, code revision (and whether it had uncommitted changes, with the patch), package
versions, hardware and configuration.

### The `cdm` CLI

| Command | What it does |
| --- | --- |
| `cdm datasets` / `cdm systems` | registered datasets (`src/cdm/datasets.py`) and rerankers (`src/cdm/rerankers/`) |
| `cdm prefetch` | download indexes, topics, qrels and models into the caches |
| `cdm first-stage <dataset>` | the BM25 top-100 (Pyserini search, or DL21-23's official file), cached for all rerankers |
| `cdm rerank <system> <dataset>` | rerank the cache, store raw scores, evaluate, write results.csv |
| `cdm results` | print results.csv as a table |
| `cdm info` | versions, paths, GPU |

**Candidates and systems** are fixed in [docs/roadmap.md](docs/roadmap.md), "Setup": the BM25 top-100 everywhere (for
DL21-23 read from the official top-100 files, which are exactly that set), so that published numbers on the same
candidates can stand in for runs; those are in `results/literature.csv`, and `slurm/submit.sh` skips their cells.
`cdm datasets` and `cdm systems` list what is registered.

| Group | Systems | Notes |
| --- | --- | --- |
| Cross-encoders | `bge-m3`, `monot5-3b`, `qwen3-reranker-8b` | fp32 + TF32 (`src/cdm/rerankers/cross_encoder.py`, `pointwise_lm.py`) |
| Cross-encoder scoring the candidates jointly | `set-encoder-large` | lightning-ir, own container (`container/lightning-ir.def`); at most 100 candidates |
| Listwise LLM | `rankzephyr-7b` | generates a permutation, sliding window 20/10 (rank_llm's prompt); BEIR from the literature |
| LLM judge | `qwen3.7-flash` | OpenRouter, P(yes) from log-probs; needs `OPENROUTER_API_KEY` |
| System One | `jev-*`, `clef-flash-*`, `clef-*`, `kev-4b-*`, `kev-27b-*` | each `-joint` (all candidates in one state) and `-isolated` (one per request), both pointwise |
| System One | `laya` | isolated only (context too short for a list) |

The System One models share one reranker (`src/cdm/rerankers/system_one.py`) and get identical requests with
S1Rank's question (`--phrasings s1rank`): Jev through the TypeSafe API (`TYPESAFE_AI_API_KEY`, runs on the CPU
partition; with `JEV_PROVIDER=openrouter` in `.env`, the same model through OpenRouter, `OPENROUTER_API_KEY`), Laya,
Clef and Kev on a GPU (the 27B models need an 80 GB GPU). A joint request that is too long for the model is split in
halves.

## Slurm details

Conventions are in [CLAUDE.md](CLAUDE.md).

| Job | Batch (`slurm/submit.sh ...`) | Interactive counterpart |
| --- | --- | --- |
| hello | `hello` | `slurm/hello.interactive.sh` (inside `slurm/interactive.sh`) |
| prefetch | `prefetch [--datasets ...] [--systems ...]` | `slurm/prefetch.interactive.sh` (on the login node) |
| first stage | `first-stage [--datasets ...]` | `slurm/first-stage.interactive.sh <dataset>` |
| rerank | `rerank <system> [--datasets ...] [--cpu \| --gres gpu:a100-sxm:1]` | `slurm/rerank.interactive.sh <system> <dataset>` |

- `slurm/config.sh`: account (`chair-ds`), dataset and system lists, GPU request (`gpumem.48gib` -- verify with
  `sinfo -N -o "%N %G %f"`), `--time` budgets (first guesses; correct them from real runs).
- `slurm/env.sh`: paths, caches and the container call, shared by all jobs. `slurm/snapshot.sh`: the per-submission
  code snapshot and its `REVISION`.
- Partitions (FAITH docs): `general` CPU, 48h max; `gpu`, 28h max; `short`, 1h max with higher priority.
  Storage: `/shared/$USER` (Ceph, all nodes, slow for many small files, two-week snapshots, no backup),
  `/local/$USER` (node-local scratch, clean up), `/home` (login nodes only).
- Rerank latency for the paper needs one fixed GPU type: `slurm/submit.sh rerank <system> --gres gpu:a100-sxm:1`.

## Paper

```bash
cd paper && make && make check
```

`paper/main.tex` includes one file per section from `paper/sections/`; RQ boxes and badges come from
`paper/rq-macros.tex`. Tables are generated from `results/results.csv` and `results/literature.csv` into `paper/tables/` by `src/cdm/paper_tables.py` (`make tables`; `make` reruns it when the results change).

## Working together

Two people work in this repository with ordinary git branches: develop on a feature or dev branch, merge into
`master` regularly in small working steps, run the unit tests (`pytest`) before
merging, and never rewrite `master`'s history.

## Next steps

See [docs/roadmap.md](docs/roadmap.md), "Order of work"; the `experiment-status` skill (or
`python3 .claude/skills/experiment-status/scripts/status.py`) shows what is done, running and missing.

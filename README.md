# calibrated-decision-models-for-ir

Experiments and paper for *Decide, then Rank: Calibrated System One Models as Rerankers and Retrieval Controllers*
(working title). The plan, research questions and work packages are in [docs/research-plan.md](docs/research-plan.md).

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
bash slurm/prefetch.interactive.sh --datasets dl19,dl20,scifact --systems minilm-l6,monot5-base
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
slurm/exec.sh --gpu -- cdm rerank minilm-l6 scifact --first-stage bm25-d100-max20 --no-save

# Batch jobs: one job per dataset, --time and --account from slurm/config.sh
slurm/submit.sh first-stage --datasets all                       # -> Submitted batch job 101 ... 110
slurm/submit.sh rerank monot5-base --datasets dl19,dl20 --after 101,102
slurm/submit.sh rerank kev-27b-listwise --datasets dl21,dl22,dl23 --dry-run   # print the sbatch calls only

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
| `runs/<dataset>/official-d1000/`, `official-d100/`, `bm25-d100/` | first-stage cache (official pool or our BM25): `candidates.jsonl.gz` (with passage texts), `run.trec.gz`, `qrels.tsv`, `manifest.json` |
| `outputs/<dataset>/<first-stage>/<system>/<phrasing>/` | `scores.jsonl.gz` (score and probability per query and passage), `responses.jsonl.gz` (API systems: every raw response with tokens and latency), `run.trec.gz`, `perquery.csv`, `manifest.json` |
| `results/results.csv` | one row per configuration: nDCG@10, R@10, R@100, RR@10, AP, ECE, Brier, latency p50/p95, truncation, API requests/tokens/cost (API systems), provenance |

Every manifest records the command, code revision (and whether it had uncommitted changes, with the patch), package
versions, hardware and configuration.

### The `cdm` CLI

| Command | What it does |
| --- | --- |
| `cdm datasets` / `cdm systems` | registered datasets (`src/cdm/datasets.py`) and rerankers (`src/cdm/rerankers/`) |
| `cdm prefetch` | download indexes, topics, qrels and models into the caches |
| `cdm first-stage <dataset>` | the official rerank pool, or BM25 top-100 with Pyserini, cached for all rerankers |
| `cdm rerank <system> <dataset>` | rerank the cache, store raw scores, evaluate, write results.csv |
| `cdm results` | print results.csv as a table |
| `cdm info` | versions, paths, GPU |

**Candidates.** Every dataset's default first stage is its official rerank pool where the track published one, else
our BM25 top-100 (owner's decision, 2026-10-07; `cdm datasets` shows which): TREC DL 2019/2020 use the official
`msmarco-passagetest{2019,2020}-top1000` pools (up to 1000 per query, unranked, so ordered by our BM25 score;
`--depth 100` then reranks the BM25 top 100 of the pool), TREC DL 2021-2023 the official `202x_passage_top100`
pools (ranked BM25 runs on MS MARCO v2; texts from the `msmarco-v2-passage` index), MS MARCO dev `top1000.dev`, and
BEIR, which publishes
no pools, uses Pyserini's BM25 flat top-100 (the de-facto standard). `cdm first-stage dl19` builds
`runs/dl19/official-d1000/` from the pool `cdm prefetch` downloaded; `--retriever bm25` forces our own BM25.

**Systems.** `cdm systems` lists them. Kanishka's selection (branch `master`) is included, with larger sizes for the
48/80 GB GPUs. Every system outputs one score per candidate; IR terms for what it sees: **pointwise** = one
(query, passage) pair at a time, **listwise** = the whole candidate list of the query at once (here: still one score
per candidate, as in S1Rank's "joint pointwise" and the Set-Encoder; not a generated permutation as in RankGPT).

| Group | Systems | Mode | Notes |
| --- | --- | --- | --- |
| Cross-encoders | `minilm-l6`, `bge-m3`, `mxbai-base` | pointwise | sentence-transformers |
| Trained yes/no LMs | `monot5-base`, `monot5-3b`, `qwen3-reranker-0.6b`, `-4b`, `-8b` | pointwise | Kanishka's prompts and readout; fp32 + TF32 (`src/cdm/rerankers/pointwise_lm.py`) |
| Set-Encoder | `set-encoder-base`, `set-encoder-large` | listwise | lightning-ir, own container; at most 100 candidates (`--depth 100` on the 1000-candidate pools) |
| LLM judge | `qwen3.7-flash` | pointwise | OpenRouter, P(yes) from log-probs, Kanishka's prompt; needs `OPENROUTER_API_KEY` |
| System One | `jev-*`, `clef-flash-*`, `clef-*`, `kev-0.8b-*`, `kev-4b-*`, `kev-9b-*`, `kev-27b-*` | `-pointwise` and `-listwise` | the RQ4 comparison |
| System One | `laya`, `laya-multilingual` | pointwise only | context too short for a list |

Status: the cross-encoders work; the monoT5, Qwen3-Reranker, Set-Encoder, Kev and Qwen3.7-Flash rerankers are new
and not yet run on the cluster. The System One models share one reranker (`src/cdm/rerankers/system_one.py`) and get
identical requests: Jev through the TypeSafe API (needs `TYPESAFE_AI_API_KEY` in `.env`, runs on the CPU partition),
Laya, Clef and Kev on a GPU (the 27B models, `clef-pointwise`/`-listwise` and `kev-27b-*`, need an 80 GB GPU). A listwise request that is too long for the
model is split in halves. The phrasings in `src/cdm/phrasings.py` are a draft until WP3 freezes them, except
`s1rank` (S1Rank's question, for replication).

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
`paper/rq-macros.tex`. Tables are generated from `results/results.csv` into `paper/tables/`.

## Working together

Two people work in this repository with ordinary git branches: develop on a feature or dev branch, merge into
`master` regularly in small working steps, run the unit tests (`pytest`) before
merging, and never rewrite `master`'s history.

## Next steps

1. Rebuild the container (Kev, flash-linear-attention), prefetch the official pools and the new models, build the
   DL19/20 official-pool caches, and run all systems on them; compare with Kanishka's numbers on the same pool.
   Then prefetch the MS MARCO v2 index (batch job) for DL21-23 and the remaining datasets, and run the first stage
   on all of them.
2. WP2: all systems on DL19/DL20 (the calibration sets with judged non-relevant documents), then the BEIR subset;
   replicate S1Rank's Jev numbers as a sanity check; report the go/no-go decision on Laya before any WP4 work.
3. WP0 landscape note (S1Rank, the Decision Index, Laya and Clef model cards) in `docs/notes/wp0-landscape.md`.
4. Move `paper/` to ACM sigconf for the WSDM 2027 short paper (`docs/intent/venue.md`) and add `cdm table`.

# calibrated-decision-models-for-ir

Experiments and paper for *Decide, then Rank: Calibrated System One Models as Rerankers and Retrieval Controllers*
(working title). The plan, research questions and work packages are in [docs/research-plan.md](docs/research-plan.md).

## Layout

```
cluster/        run on YOUR machine: talk to the FAITH login node over one shared SSH connection
slurm/          run on the cluster: Slurm jobs (.sbatch + .interactive.sh pairs), submit.sh, config.sh, env.sh
container/      Apptainer image with the dependencies (cdm.def, build.sh)
src/cdm/        the Python package and `cdm` CLI: first stage, rerankers, evaluation, results.csv
tests/          pytest
results/        results.csv and generated tables -- tracked in git, fetched from the cluster
paper/          LaTeX sources (LNCS), built with `make`
docs/           research plan (owner's, read-only), experiment log, notes, intent statements
```

Gitignored and only on the cluster under `/shared/<user>/calibrated-decision-models-for-ir/`: `data/` (Pyserini and
Hugging Face caches), `runs/` (cached first-stage candidates), `outputs/` (raw reranker scores), `logs/`,
`container/cdm.sif`.

## Setup

**Local (your machine).** Python 3.10-3.13 and Java 21 (for Pyserini, only needed to run the first stage locally):

```bash
poetry install --extras dev     # or: pip install -e '.[dev]'
pytest
```

**Cluster access, once.** The FAITH cluster accepts only password or Kerberos logins (no SSH keys) and is reachable
from the university network (office, Eduroam, EduVPN). Add the `faith` host from
[cluster/ssh_config.example](cluster/ssh_config.example) to `~/.ssh/config`, then:

```bash
cp .env.example .env            # optional: HF_TOKEN
cluster/connect.sh              # asks for your password once; the connection stays open 12h after last use
cluster/connect.sh --check
```

Every `cluster/*.sh` reuses that connection without prompting, which is what lets scripts and coding agents drive
the cluster. They never compute on the login node -- the FAITH docs forbid that -- but submit to Slurm.

**Cluster environment, once** (and the container again whenever `pyproject.toml` changes):

```bash
cluster/sync.sh                                   # copy the code to /shared/<user>/calibrated-decision-models-for-ir
cluster/login.sh bash container/build.sh          # build container/cdm.sif on the login node (fakeroot)
cluster/login.sh bash slurm/prefetch.sh           # Pyserini indexes + topics/qrels, Hugging Face models -> data/
cluster/submit.sh hello                           # sanity check: Slurm, logging, container
cluster/exec.sh --gpu -- cdm info                 # ... and a GPU inside the container
```

`slurm/prefetch.sh` takes `--datasets dl19,scifact` / `--systems minilm-l6` to fetch less; the full set includes
the MS MARCO passage index (several GB).

## Running experiments

All from your machine. Each command that starts work syncs the code first (`--no-sync` skips that).

```bash
# Quick, blocking check on a compute node (short partition, <= 1h):
cluster/exec.sh -- cdm info
cluster/exec.sh -- cdm first-stage scifact --max-queries 20
cluster/exec.sh --gpu -- cdm rerank minilm-l6 scifact --first-stage bm25-d100-max20 --no-save

# Batch jobs: one job per dataset, --time and --account from slurm/config.sh
cluster/submit.sh first-stage --datasets all                       # -> Submitted batch job 101 ... 107
cluster/submit.sh rerank minilm-l6 --datasets all --after 101,102,103,104,105,106,107
cluster/submit.sh rerank bge-m3 --datasets dl19,dl20 --dry-run     # print the sbatch calls only

# Follow them
cluster/status.sh                     # squeue --me
cluster/status.sh 101 102             # sacct: state, elapsed, exit code, memory
cluster/logs.sh 101 -f                # follow a log (logs/<jobid>_<job>_<args>_<ts>.out)

# Results
cluster/fetch.sh                      # results/ -> here; then git diff results/ && git commit
cdm results --metric nDCG@10          # system x dataset table
cluster/fetch.sh --path outputs/dl19/bm25-d100/minilm-l6    # raw scores of one run, if needed
```

What the jobs produce:

| Path (on the cluster) | Content |
| --- | --- |
| `runs/<dataset>/bm25-d100/` | first-stage cache: `candidates.jsonl.gz` (with passage texts), `run.trec.gz`, `qrels.tsv`, `manifest.json` |
| `outputs/<dataset>/<first-stage>/<system>/<phrasing>/` | `scores.jsonl.gz` (score and probability per query and passage), `responses.jsonl.gz` (API systems: every raw response with tokens and latency), `run.trec.gz`, `perquery.csv`, `manifest.json` |
| `results/results.csv` | one row per configuration: nDCG@10, R@10, R@100, RR@10, AP, ECE, Brier, latency p50/p95, truncation, API requests/tokens/cost (API systems), provenance |

Every manifest records the command, code revision (and whether it had uncommitted changes), package versions,
hardware and configuration.

### The `cdm` CLI

| Command | What it does |
| --- | --- |
| `cdm datasets` / `cdm systems` | registered datasets (`src/cdm/datasets.py`) and rerankers (`src/cdm/rerankers/`) |
| `cdm prefetch` | download indexes, topics, qrels and models into the caches |
| `cdm first-stage <dataset>` | BM25 top-100 with Pyserini, cached for all rerankers |
| `cdm rerank <system> <dataset>` | rerank the cache, store raw scores, evaluate, write results.csv |
| `cdm results` | print results.csv as a table |
| `cdm info` | versions, paths, GPU |

Status of the systems: the three cross-encoders work. The System One models share one reranker
(`src/cdm/rerankers/system_one.py`) and get identical requests: Jev through the TypeSafe API (`jev`: all candidates in
one request; `jev-isolated`: one per request; needs `TYPESAFE_AI_API_KEY` in `.env`, runs on the CPU partition), Laya
on a GPU (`laya`, `laya-multilingual`; one candidate per request, its context is too short for more) and Cloudflare's
Clef on a GPU (`clef-flash`, `clef`, each also `-isolated`; `clef` needs an 80 GB GPU). The phrasings in
`src/cdm/phrasings.py` are a draft until WP3 freezes them, except `s1rank` (S1Rank's question, for replication).

## Slurm details

`slurm/` runs on the cluster; `cluster/` wraps it. Conventions are in [CLAUDE.md](CLAUDE.md).

| Job | Batch (`cluster/submit.sh ...`) | Interactive counterpart (inside `cluster/interactive.sh`) |
| --- | --- | --- |
| hello | `hello` | `slurm/hello.interactive.sh` |
| first stage | `first-stage [--datasets ...]` | `slurm/first-stage.interactive.sh <dataset>` |
| rerank | `rerank <system> [--datasets ...] [--cpu \| --gres gpu:a100-sxm:1]` | `slurm/rerank.interactive.sh <system> <dataset>` |

- `slurm/config.sh`: account (`chair-ds`), dataset and system lists, GPU request (`gpumem.48gib` -- verify with
  `cluster/login.sh sinfo -N -o "%N %G %f"`), `--time` budgets (first guesses; correct them from real runs).
- `slurm/env.sh`: paths, caches and the container call, shared by all jobs.
- Partitions (FAITH docs): `general` CPU, 48h max; `gpu`, 28h max; `short`, 1h max with higher priority.
  Storage: `/shared/$USER` (Ceph, all nodes, slow for many small files, two-week snapshots, no backup),
  `/local/$USER` (node-local scratch, clean up), `/home` (login nodes only).
- Rerank latency for the paper needs one fixed GPU type: `cluster/submit.sh rerank <system> --gres gpu:a100-sxm:1`.

## Paper

```bash
cd paper && make && make check
```

`paper/main.tex` includes one file per section from `paper/sections/`; RQ boxes and badges come from
`paper/rq-macros.tex`. Tables are generated from `results/results.csv` into `paper/tables/`.

## Working together

The project runs on a shared VM (`docs/vm-setup.md`: repository, SSH, Kerberos keytab with automatic tickets) where
several people and agents work in parallel. Every change is a claimed task in its own git worktree, merged back to
`main` as soon as it works (`docs/collaboration.md`):

```bash
scripts/task.sh list                                   # who works on what
scripts/task.sh start jev-dl19-runs "Jev on DL19/20"   # claim + branch task/<id> + worktree
scripts/task.sh finish                                 # in the worktree: rebase, scripts/check.sh, merge, push
```

A Claude Code hook (`scripts/agent_guard.py`, `.claude/settings.json`) blocks edits outside claimed task worktrees,
force-pushes and direct pushes to `main`.

## Next steps

`docs/handoff.md` (state of the project) and `docs/tasks/backlog.md` (open work, in order).

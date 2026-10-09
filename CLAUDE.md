# Project conventions

Research project: do System One decision models (Jev, Clef, Kev, Laya) make better rerankers than reranker fine-tunes
of the same backbone, and is their calibration of use? The plan of record is
`docs/roadmap.md` (research questions, setup, order of work); the owner's decisions are in `docs/intent/`; what was
tried, including failures, goes to `docs/experiment-log.md` (append-only); the paper lives in `paper/`.
`docs/research-plan.md` is the owner's original proposal (read-only); where it differs, `docs/roadmap.md` holds.

## Working together

Two people (and their agents) work in this repository with ordinary git branches:

- Work on a feature or dev branch (`feature/<topic>`, `dev-<name>`), never directly on `master`.
- Merge your changes into `master` regularly -- small, working steps rather than long-lived branches. Run
  the unit tests (`pytest`) before merging.
- Never force-push or rewrite `master` or someone else's branch. Pull/merge `master` into your branch before merging
  back, so conflicts are resolved on your branch.

## Where things run

- **The repository lives on the cluster**, under `/shared` (Ceph, visible to the login and compute nodes); you work
  in it on a FAITH login node (`freia`/`frigga`). Data, caches, runs, outputs and logs sit next to the code
  (gitignored), and `results/results.csv` is written in place.
- **The login node is not for computation.** The FAITH docs: "Do not use the login nodes to spawn autonomous agents
  operating without supervision" and "must not be used to run computationally intensive or long-running
  non-interactive tasks". So an agent works here only in a session the user supervises, and runs only light commands
  itself: reading and editing files, git, `sbatch`/`squeue`/`sacct`/`sinfo`, a quick `pytest` of the unit tests,
  small downloads. Never load a model, run a reranker or first stage, or install large packages on the login node,
  not even "just to check"; that goes to a compute node.
- Which script for what (all run from the login node, in the repo root):
  - `slurm/exec.sh [--gpu] -- <cmd>` -- one blocking command in the container on a compute node (`short`
    partition): quick checks, `--max-queries 20` trial runs.
  - `slurm/submit.sh <job> ...` -- batch jobs. `--dry-run` shows the `sbatch` calls without submitting; use it before
    submitting many jobs.
  - `squeue --me`, `sacct -j <jobid>`, `tail -f logs/<jobid>_*` -- follow jobs (logs move to `logs/done/` at exit).
  - Login node only: `bash container/build.sh`, `bash slurm/prefetch.interactive.sh ...` for small downloads
    (big ones, e.g. the MS MARCO v2 index: `slurm/submit.sh prefetch ...`).
  - `slurm/interactive.sh` -- an interactive shell on a compute node, for the human only (needs a terminal).
- `slurm/submit.sh` and `slurm/exec.sh` snapshot `src/` per submission (`snapshots/`, `slurm/snapshot.sh`), with
  `REVISION`/`REVISION.patch` recording the commit and the uncommitted changes (untracked files included), so queued
  jobs run the code they were submitted with; every manifest records the revision and copies the patch. Commit
  before runs whose numbers go into the paper.
- The container (`container/cdm.sif`) holds only the dependencies; the code comes from `src/` (or the snapshot) via
  `PYTHONPATH`. Rebuild (`bash container/build.sh`) only after changing `pyproject.toml` or `container/cdm.def`.
- Jobs run with `HF_HUB_OFFLINE=1`: download new models/datasets with the prefetch scripts first.

## Slurm scripts (`slurm/`)

- Every `slurm/<job>.sbatch` has a `slurm/<job>.interactive.sh` counterpart running the same container command
  without `srun` and without `#SBATCH` lines. Change both in the same commit.
- Environment and container call are defined once, in `slurm/env.sh` (`"${CONTAINER_CMD[@]}" cdm ...`); scripts don't
  set paths or `apptainer` flags themselves.
- No array jobs. Each `.sbatch` takes exactly one dataset (and one system) as positional arguments (except
  `prefetch.sbatch`, one download job for everything);
  `slurm/submit.sh` loops and submits one job per dataset.
- `--time` and `--account` are set only in `slurm/config.sh` (`time_for`, `SLURM_ACCOUNT`) and passed by
  `slurm/submit.sh`; no `#SBATCH --time`/`--account` anywhere. When a job's real run time is known, update its
  budget there with a comment citing the job id.
- Every `.sbatch` calls `log_job_command <job>.sbatch "$@"` first, then `start_job_log <job> <label>` (both in
  `slurm/log_command.sh`), so its log starts with the exact command and is named
  `logs/<jobid>_<job>_<label>_<timestamp>.out` (moved to `logs/done/` at exit).
- A new job: add the `.sbatch` + `.interactive.sh` pair, a case in `slurm/submit.sh`, a budget in `slurm/config.sh`,
  and a line in README.md.

## Code (`src/cdm/`)

- Datasets are registered in `src/cdm/datasets.py`, rerankers in `src/cdm/rerankers/__init__.py`, phrasings in
  `src/cdm/phrasings.py`. A new dataset or system also goes into `slurm/config.sh`.
- All metrics go through `cdm.results_csv.save_result` (locked read-modify-write; many jobs write concurrently).
  Never edit `results/results.csv` by hand; to rename a system use `cdm rename-system` and log it. Its key columns are `RESULT_KEY_COLUMNS`; anything that changes a result
  and isn't one of them belongs in `variant`.
- Every output directory gets a `manifest.json` (`cdm.provenance.write_manifest`).
- Tests: `pytest` (tests/). Formatting: `black` and `isort` (line length 120).

## Experiment rules

- One setup (`docs/roadmap.md`, "Setup"): BM25 top-100 candidates, S1Rank's question (`s1rank`) for all System One
  models. Never tune or fit anything on test splits: post-hoc calibration (RQ2) is fitted on `msmarco-dev`, and a
  different question would be chosen on `msmarco-dev` only and get a new phrasing id (never edit a frozen one).
- Don't recompute what is published on the same candidates: such cells come from `results/literature.csv`
  (`from_literature` in `slurm/config.sh`). Literature numbers are labelled as such in the paper.
- System One modes are named in S1Rank's terms: `<model>-joint` / `<model>-isolated`, both pointwise. "Listwise" means
  a reranker that outputs a ranking (RankGPT, RankZephyr).
- Store raw model outputs (score and probability per query and passage), not just rankings.
- Fix seeds; record library, model and checkpoint versions (the manifests do this -- keep it that way).
- Phase 2 of the roadmap (control signal, fine-tuning, BRIGHT) starts after the WSDM submission.

## Paper (`paper/`)

- Target: WSDM 2027 short paper (ACM sigconf, 4 pages); the sources are still LNCS until the move (roadmap step 5).
  One file per section in `paper/sections/`; RQ wording only in `paper/rq-macros.tex`; build and check with `make` /
  `make check` in `paper/`.
- Numbers in the paper come from `results/results.csv` and `results/literature.csv` via generated tables in
  `paper/tables/`, never typed by hand.
- Draft markers: `\TODO{}`, `\PENDING{}` (waits for a result), `\FLAG{}` (claim to verify).

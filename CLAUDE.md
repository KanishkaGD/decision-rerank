# Project conventions

Research project: calibrated System One decision models (Laya, Jev, Clef) as rerankers and retrieval controllers. The
plan is `docs/research-plan.md` (do not edit it); decisions and failed attempts go to `docs/experiment-log.md`; the
paper lives in `paper/`. The current state and next steps are in `docs/handoff.md`; open work in
`docs/tasks/backlog.md`.

## Working together (read first)

Several people and agents share this repository. Follow `docs/collaboration.md`; in short:

- Start with `scripts/task.sh list` and `scripts/task.sh status`. Claim before you change anything:
  `scripts/task.sh start <id> "<description>"`, then work only in the worktree it creates.
- On the VM, two people share one Linux user; `CDM_WHO` (from `~/people/<name>.env`) says whose session you are.
  If it isn't set, stop and ask -- never set it yourself. Don't touch tasks claimed by the other person.
- Never edit in the main checkout or on `main`; never force-push or rewrite history. A PreToolUse hook
  (`scripts/agent_guard.py`) enforces this -- don't work around it.
- When the task works, `scripts/task.sh finish`: rebase on `main`, `scripts/check.sh`, fast-forward `main`, push,
  release. Don't let branches live longer than the task.
- Don't stop with uncommitted work; commit and update `docs/handoff.md` in your task if you leave it unfinished.

## Where things run

- **You (the agent) run on the shared VM** (`docs/vm-setup.md`) or a user's machine, in a task worktree of this repo.
  Never on the cluster's login node: the FAITH docs forbid autonomous agents and compute-heavy work there.
- The cluster is reached only through `cluster/*.sh`, which send single commands over SSH to the login node and from
  there to Slurm. They share one SSH connection per VM user, opened automatically with the Kerberos ticket that a
  cron job keeps fresh from a keytab (VM), or by the user with `cluster/connect.sh` (laptop). If a script fails with
  "no connection", stop and tell the user (on the VM: check `klist`, docs/vm-setup.md "Maintenance") -- never ask for
  or handle the password, and never read, copy or print the keytab (`/srv/cdm/secrets/`).
- Each checkout syncs its code to its own `checkouts/<user>-<worktree>/` on `/shared`; data, caches, outputs,
  results, logs and the container are shared by all (`cluster/cluster.env`).
- Which wrapper for what:
  - `cluster/exec.sh [--gpu] -- <cmd>` -- one blocking command in the container on a compute node (`short`
    partition): quick checks, `--max-queries 20` trial runs.
  - `cluster/submit.sh <job> ...` -- batch jobs (runs `slurm/submit.sh` remotely). `--dry-run` shows the `sbatch`
    calls without submitting; use it before submitting many jobs.
  - `cluster/status.sh`, `cluster/logs.sh <jobid>` -- follow jobs. `cluster/fetch.sh` -- copy `results/` back.
  - `cluster/login.sh <cmd>` -- light login-node commands only: `bash slurm/prefetch.sh ...`,
    `bash container/build.sh`, `ls`, `sinfo`, `sacct`.
  - `cluster/interactive.sh` -- for the human only (needs a terminal).
- Every wrapper that starts work first runs `cluster/sync.sh` (rsync of the working tree, respecting `.gitignore`;
  `results/` and `paper/` stay local). Jobs therefore run uncommitted code; `REVISION`/`REVISION.patch` record the
  commit and the uncommitted changes (untracked files included). `slurm/submit.sh` snapshots `src/` with them per
  submission (`snapshots/`), so queued jobs run the code they were submitted with; every manifest records the
  revision and copies the patch. Commit before runs whose numbers go into the paper.
- The container (`container/cdm.sif`) holds only the dependencies; the code comes from the synced `src/` via
  `PYTHONPATH`. Rebuild (`cluster/login.sh bash container/build.sh`) only after changing `pyproject.toml`.
- Jobs run with `HF_HUB_OFFLINE=1`: download new models/datasets with `slurm/prefetch.sh` on the login node first.

## Slurm scripts (`slurm/`)

- Every `slurm/<job>.sbatch` has a `slurm/<job>.interactive.sh` counterpart running the same container command
  without `srun` and without `#SBATCH` lines. Change both in the same commit.
- Environment and container call are defined once, in `slurm/env.sh` (`"${CONTAINER_CMD[@]}" cdm ...`); scripts don't
  set paths or `apptainer` flags themselves.
- No array jobs. Each `.sbatch` takes exactly one dataset (and one system) as positional arguments;
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
  Never edit `results/results.csv` by hand. Its key columns are `RESULT_KEY_COLUMNS`; anything that changes a result
  and isn't one of them belongs in `variant`.
- Every output directory gets a `manifest.json` (`cdm.provenance.write_manifest`).
- Tests: `pytest` (tests/). Formatting: `black` and `isort` (line length 120).

## Experiment rules (from the research plan)

- Never tune anything on test splits. Phrasing and primitive are selected on `msmarco-dev` only (WP3); a phrasing is
  frozen (`frozen=True`) before any test-set run and never edited afterwards -- add a new id instead.
- Store raw model outputs (score and probability per query, passage and phrasing), not just rankings.
- Fix seeds; record library, model and checkpoint versions (the manifests do this -- keep it that way).
- Numbers from prior work are unverified until replicated; label them as such in the paper.
- Report back to the user after WP2 with the go/no-go decision before starting WP4.

## Paper (`paper/`)

- LNCS for now (venue undecided). One file per section in `paper/sections/`; RQ wording only in
  `paper/rq-macros.tex`; build and check with `make` / `make check` in `paper/`.
- Numbers in the paper come from `results/results.csv` via generated tables in `paper/tables/`, never typed by hand.
- Draft markers: `\TODO{}`, `\PENDING{}` (waits for a result), `\FLAG{}` (claim to verify).

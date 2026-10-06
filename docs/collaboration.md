# Collaboration rules for people and agents

Several people (Michael and a colleague) and their coding agents work in this repository at the same time, on a
shared VM (docs/vm-setup.md). These rules keep them from overwriting each other's work and keep `main` the one
version everyone builds on. `CLAUDE.md` points here; agents must follow these rules exactly. The guardrail hook
(`scripts/agent_guard.py`, wired in `.claude/settings.json`) enforces the most important ones.

## The rules

1. **One task, one claim, one worktree.** Before changing anything, claim a task:
   `scripts/task.sh start <id> "<one-line description>"`. This creates branch `task/<id>` and its own worktree
   (`/srv/cdm/worktrees/<id>` on the VM) from the current `main`, and records the claim where every checkout sees it
   immediately. Work only in that worktree. `scripts/task.sh list` shows who is working on what; check it before
   starting, and don't claim work that overlaps an active task -- ask its owner instead.
   - Task ids: short, lowercase, dashes (`jev-dl19-runs`, `wp0-landscape-note`, `fix-ece-binning`).
   - A task is one unit you can finish and merge within a day or two: a feature, a fix, one batch of experiments, one
     paper section. Split bigger work.
2. **Never edit in the main checkout** (`/srv/cdm/repo`, always on `main`). It only moves forward through
   `scripts/task.sh finish`. The hook blocks edits there and edits on any branch that isn't a claimed `task/<id>`.
3. **Merge back as soon as the task is done and works.** `scripts/task.sh finish` (in the task worktree) rebases the
   branch on the newest `main`, runs `scripts/check.sh` (shell syntax, Python compile, unit tests), fast-forwards
   `main`, pushes it, releases the claim and removes the worktree. If the rebase conflicts, resolve the conflicts in
   the worktree and run finish again. Branches are not meant to live longer than the task; no long-lived personal
   branches.
4. **Commit small and often** inside the task branch, with messages that say what and why. The final history of
   `main` is linear (rebase + fast-forward), so every commit should leave the code working.
5. **Abandoning a task:** `scripts/task.sh release <id>` drops the claim and keeps the branch; mention it in
   `docs/tasks/backlog.md` if someone should pick it up.
6. **Never force-push, never rewrite `main`, never delete someone else's branch or worktree.** The hook blocks the
   common ways.

## Where things are written

| What | Where | Notes |
| --- | --- | --- |
| Who is working on what, right now | `scripts/task.sh list` (claims in `.git/cdm-claims/`, untracked) | Never edit the claims by hand |
| Open work nobody has claimed yet | `docs/tasks/backlog.md` | Add and remove lines in the task that creates or finishes the work |
| Decisions, failed attempts, surprising results | `docs/experiment-log.md` | Newest first; job ids and commits; the plan requires it |
| The owner's answers to open questions | `docs/intent/` | One file per topic, plus a line in the experiment log |
| Notes, landscape checks, analyses | `docs/notes/<topic>.md` | |
| How to run things | `README.md` | Update it in the same task that changes a command |
| Conventions for agents | `CLAUDE.md` | Change only when a rule changes, and tell the others |
| Metrics | `results/results.csv` via `cdm.results_csv.save_result` | Never by hand; see below |
| The paper | `paper/` | Numbers only from generated tables |
| The research plan | `docs/research-plan.md` | Never edit; propose changes to the owner |
| Handoffs between sessions | `docs/handoff.md` | Rewrite it at the end of a session that leaves work half-done |

Two people editing `docs/experiment-log.md` at once is normal: entries go at the top, so a rebase conflict there is
resolved by keeping both entries, newest first.

## The cluster

- All checkouts use the same FAITH account and the same project directory on `/shared`. Each checkout syncs its code
  to its own `checkouts/<user>-<worktree>/` directory, so worktrees never overwrite each other's code; data, caches,
  outputs, logs, the container and `results/results.csv` are shared (cluster/cluster.env explains the layout).
- `results/results.csv` on the cluster is the authoritative copy (many jobs append to it, locked). Fetch it with
  `cluster/fetch.sh` inside your task and commit it there. On a rebase conflict in `results/results.csv`, don't merge
  by hand: take the newer file (`cluster/fetch.sh` again after the rebase) -- the cluster copy contains everyone's
  rows.
- The container image is shared: rebuild it (`cluster/login.sh bash container/build.sh`) only in a task that changes
  `pyproject.toml` or `container/cdm.def`, and say so in `scripts/task.sh list` (description) while it builds, since
  everyone's next job uses the new image.
- GPU and API budget are shared: before submitting more than ~20 jobs or an API run expected to cost over $10, check
  `cluster/status.sh` and ask the owner.
- Jobs keep running when a session ends. Write the job ids into the experiment log (or the handoff) before you stop,
  so whoever comes next can collect the results.

## For agents in particular

- Start every session with `scripts/task.sh list` and `scripts/task.sh status`. If you are in the main checkout,
  start or resume a task first; don't work around the hook.
- Don't stop with uncommitted changes in a task worktree. Either finish the task, or commit what you have, write the
  state into `docs/handoff.md` (in the task branch) and tell the person you work for.
- Don't edit files another active task is clearly working on (check the descriptions in `scripts/task.sh list`). If
  you must, say so to your person, who coordinates with the other.
- The guardrail can be switched off for one session with `CDM_GUARD=off` in the environment -- only a human decides
  that.

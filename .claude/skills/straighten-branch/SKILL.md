---
name: straighten-branch
description: >-
  Straighten this research repository after the plan changed: bring the roadmap, decisions, README, CLAUDE.md, paper
  skeleton, Slurm grid, registries and skills in line with the current plan, and remove what was planned before but
  is now dropped -- without rewriting history (the experiment log stays). Use this whenever the user changes the
  setup, scope or terminology ("we drop X", "let's simplify", "rename Y", "clean the docs", "straighten the
  roadmap", "the plan changed"), after a batch of decisions, before merging a branch, or when docs and code seem to
  disagree.
---

# Straighten the branch

Plans in this project change quickly (models added and dropped, candidate sets, terminology). Each change leaves
traces in a dozen places: `docs/roadmap.md`, `docs/intent/`, README, CLAUDE.md, the paper's RQ macros and sections,
`slurm/config.sh`, the registries in `src/cdm/`, `results/literature.csv`, tests, and the skills. A reader -- human
or agent -- should find one consistent, current plan, not a layered history of what we once wanted. That is what
straightening restores.

**What stays:** the history files. `docs/experiment-log.md` is append-only (it records what was tried and why, which
the plan must not); `docs/research-plan.md` is the owner's original proposal (read-only; only the owner removes or
changes it); `results/results.csv` keeps old rows (rename systems with `cdm rename-system`, never by hand).

## 1. Pin down the current plan

Collect the decisions of this conversation and of `docs/intent/` into a short list: what is in (datasets, systems,
candidates, metrics, research questions, order of work) and what was dropped or renamed. If a decision is ambiguous,
ask the user before sweeping -- a straightening built on a guess spreads the guess everywhere.

## 2. Update the sources of truth first

In this order, since later files point to earlier ones:

1. `docs/intent/*.md` -- current decisions only, each with its date; no "we used to" paragraphs.
2. `docs/roadmap.md` -- research questions, setup, order of work (phases and steps). Present tense, as the plan is.
3. Code: registries (`src/cdm/datasets.py`, `src/cdm/rerankers/__init__.py`, `src/cdm/phrasings.py`), the grid in
   `slurm/config.sh` (SYSTEMS, DATASETS, `from_literature`, budgets), `results/literature.csv`. Remove dropped
   entries rather than commenting them out; delete code paths nothing uses any more.
4. CLAUDE.md (conventions, experiment rules), README (setup, systems table, next steps -> roadmap).
5. Paper skeleton: `paper/rq-macros.tex` (RQ wording), `paper/main.tex` (title, abstract TODO, section list),
   `paper/sections/*`, `paper/sources.bib`. Build with `make` and `make check` in `paper/`.
6. Skills in `.claude/skills/` that encode the plan (e.g. `experiment-status`).

## 3. Record what was removed

Append every dropped name or outdated term as a regular expression to `deprecated.txt` (next to this file), with a
dated comment. The check below then finds stragglers now and in every later sweep.

## 4. Check

```bash
python3 .claude/skills/straighten-branch/scripts/check.py
```

It reports stale terms from `deprecated.txt` in all tracked files except the history files, mismatches between
`slurm/config.sh` and the registries, literature baselines that `from_literature` would recompute, and doc pointers
to missing files. Fix until it prints `ok` everywhere; for a hit that is legitimately current (rare), narrow the
regular expression instead of leaving the warning. Then run the unit tests (`pytest`) and, if the paper changed,
`make check` in `paper/`. All of this is light and fine on the login node.

## 5. Log and commit

Add one entry to `docs/experiment-log.md` naming the decisions and what was removed or renamed (this is where the
history goes). Commit on the working branch with a message that lists the decisions. Jobs already queued with old
names or settings: list them for the user (cancel and resubmit, or let them finish and rename with
`cdm rename-system`) -- don't cancel on your own.

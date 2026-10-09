---
name: experiment-status
description: >-
  Status report for the decision-rerank experiments: which runs are finished (results/results.csv), what the
  colleague's branches (Kanishka, master) have already run, which Slurm jobs are queued, running, failed or stuck
  behind a failed dependency (with the cause from the log), how much of the planned system x dataset grid is covered,
  and what to run next with ready-to-paste commands. Use this whenever the user asks about experiment progress or
  status ("what's running?", "did my jobs finish?", "why did job 123 fail?", "where are we?"), about results or
  what is already done ("have we run X on dl21?", "what did Kanishka run?", "compare with Kanishka"), or about
  planning ("what should I run next?", "what's missing for the paper?") -- also at the start of a session in this
  repo, and after submitting or before resubmitting jobs.
---

# Experiment status

The project has two people on two branches with different code: Michael (`dev-michael`, package `cdm`, Slurm jobs
on the FAITH cluster, results in `results/results.csv`) and Kanishka (`master`, package `decisionrerank`, his own A100,
results committed as `results/*.md` and `*.csv`), plus published numbers we use instead of runs
(`results/literature.csv`). Status questions therefore need three sources that nobody sees at
once: Slurm, our results file, and the other branch. A bundled script collects all of them; your job is to read its
report, look up causes, and turn it into a short, decision-ready answer.

## 1. Collect the facts

Run the script from the repo root:

```bash
python3 .claude/skills/experiment-status/scripts/status.py            # everything, after a git fetch
python3 .claude/skills/experiment-status/scripts/status.py --section jobs --days 2   # only Slurm, recent
```

Sections: `jobs` (Slurm: states, pending reasons, failures with log path and error lines, run times of finished
jobs), `results` (our nDCG@10, ECE/Brier, API cost), `coverage` (SYSTEMS x DATASETS from `slurm/config.sh` on the
BM25 top-100, each cell done / lit / run / queued / STUCK / FAILED / not run), `next` (submit commands for the missing
cells), `others` (every remote branch but ours: recent commits, result files, Kanishka's nDCG@10). It only uses git, `squeue`/`sacct` and file reads, so it is
fine on the login node -- never run models, `cdm rerank` or the first stage there to "check" something; that goes
through `slurm/exec.sh` or `slurm/submit.sh` (CLAUDE.md, "Where things run").

## 2. Look into problems before planning

- **Failed jobs**: the report shows the last error lines; read more of the log (`logs/done/<id>_*.out`) when the
  cause isn't obvious. Classify: transient (network, `os error 11`, node failure: resubmit), time limit (raise the
  budget in `slurm/config.sh` with a comment citing the job id), out of memory (GPU constraint or `--mem`), or a bug
  (fix the code first).
- **STUCK jobs** wait on a failed job and will never start. They have to be cancelled (`scancel <ids>`) and
  resubmitted after the cause is fixed, with new `--after` ids. Say so explicitly -- a queue full of pending jobs
  looks like progress but isn't.
- **Run times** of completed jobs: if they are far below or near their limit, suggest the budget change.

## 3. Read the plan context

To say what comes next, read `docs/roadmap.md` -- research questions, the fixed setup and the order of work -- and,
for recent events, the newest entries of `docs/experiment-log.md`. Then apply the roadmap's rules:

- **Order**: Phase 1 (paper: candidate sets, sanity checks, grid runs, analysis, writing) before Phase 2 (control
  signal, fine-tuning, BRIGHT), which starts after the WSDM submission.
- **The grid is the plan.** `slurm/config.sh` SYSTEMS x DATASETS on the BM25 top-100 is everything the paper needs;
  cells marked `lit` come from `results/literature.csv` and must not be recomputed. The `next` section prints the
  submit commands for the missing cells (first stage first, reranks with `--after` its job ids).
- **Sanity checks before trusting a dataset**: our BM25 first stage must equal Pyserini's 2CR score
  (`results/literature.csv`, role `reference`); monoT5-3B on DL19/20 within about one point of 71.83 / 68.89;
  `jev-joint` close to S1Rank's numbers. A miss means the candidates differ -- investigate before running more.
- **Fixed question**: System One models run with `--phrasings s1rank` only.
- **RQ1 (triplets)** needs, per backbone, the base model, the decision fine-tune in both modes (joint and isolated)
  and the reranker fine-tune on every dataset; the context systems (Jev, Clef, Laya, trained and listwise rerankers)
  complete the context table. **RQ2** (calibration) adds the reranker fine-tunes on a msmarco-dev subset for
  post-hoc scaling (`docs/roadmap.md`, step 7).
- **Kanishka's numbers** (branch `master`) rerank the 1000-candidate official DL19/20 pools, not our candidates:
  not directly comparable; useful as context and for Phase 2 (his QPP work).
- **Cost**: Jev (TypeSafe, or OpenRouter) costs money -- about $0.005 (joint) and $0.017 (isolated)
  per 1k query-passage pairs for Jev, measured on DL21-23; estimate before proposing and let the user decide.
  A missing API key in `.env` means the run can't start.
- **Provenance**: numbers meant for the paper must come from a commit (`dirty=false` in the job's REVISION). If the
  tree is dirty, say so and suggest committing before submitting.

## 4. Report

Keep it short and lead with what needs the user's attention. Use this structure, leaving out empty parts:

```markdown
## Status at a glance
2-4 bullets: what finished, what is running/queued, what is broken.

## Needs attention
Each failed or stuck item: cause (from the log), fix, and the exact commands.

## Results so far
Our key numbers (nDCG@10, and ECE where relevant), next to Kanishka's where comparable; mark large deviations.

## Coverage
One or two sentences, or the relevant slice of the grid -- not the whole 27 x 10 table unless asked.

## What to run next
A ranked list. Each item: why (which RQ or roadmap step it serves), the exact command(s), rough time or API cost.
```

Commands use the project's scripts: `slurm/submit.sh rerank <system> --datasets <ds> [--after <ids>]
[--phrasings s1rank]`, `slurm/submit.sh first-stage --datasets <ds>`, `slurm/submit.sh prefetch
--datasets ... --systems ...`; check a batch with `--dry-run` first. Don't submit, cancel or spend API money on
your own: propose, and act once the user agrees.

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
results committed as `results/*.md` and `*.csv`). Status questions therefore need three sources that nobody sees at
once: Slurm, our results file, and the other branch. A bundled script collects all of them; your job is to read its
report, look up causes, and turn it into a short, decision-ready answer.

## 1. Collect the facts

Run the script from the repo root:

```bash
python3 .claude/skills/experiment-status/scripts/status.py            # everything, after a git fetch
python3 .claude/skills/experiment-status/scripts/status.py --section jobs --days 2   # only Slurm, recent
```

Sections: `jobs` (Slurm: states, pending reasons, failures with log path and error lines, run times of finished
jobs), `results` (our nDCG@10, ECE/Brier, API cost), `coverage` (SYSTEMS x DATASETS from `slurm/config.sh`, each
cell done / run / queued / STUCK / FAILED / not run), `others` (every remote branch but ours: recent commits, result
files, Kanishka's nDCG@10 next to our closest system). It only uses git, `squeue`/`sacct` and file reads, so it is
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

To say what comes next, read (briefly, only what you need): the newest entries of `docs/experiment-log.md`,
"Next steps" in `README.md`, the decisions in `docs/intent/*.md`, and the work-package order in
`docs/research-plan.md`. Then apply what the plan and CLAUDE.md require:

- **Order**: WP2 baselines (DL19/20 on the official pools, then DL21-23 and BEIR) before WP3 (phrasing) and WP4
  (fine-tuning); the go/no-go on Laya is reported to the owner before WP4.
- **Test-set hygiene**: System One models run on test sets only with frozen phrasings (`s1rank` so far); draft
  phrasings run on `msmarco-dev` only (WP3). Flag any planned run that would break this.
- **RQ4 (pointwise vs. listwise)** needs `<model>-pointwise` and `<model>-listwise` for Jev, Clef and Kev on the
  same candidates and depth, plus the cross-encoders (pointwise) and Set-Encoder (listwise, at most 100 candidates:
  `--depth 100` on the 1000-candidate DL19/20 pools, so the other systems need a `--depth 100` run there too).
- **Comparison with Kanishka**: monoT5-base, Qwen3-Reranker-0.6B and Qwen3.7-Flash use his exact prompts, so on
  DL19/20 (official pool, all candidates) our numbers should match his within about 0.01 nDCG@10 -- a larger gap
  is a finding to investigate, not noise. Laya/Jev/Kev use a different question and state, so differences there
  are expected. Don't propose re-running something of his unless it is such a replication check or the setup
  differs in a way that matters.
- **Cost**: Jev (TypeSafe) and Qwen3.7-Flash (OpenRouter) cost money; estimate before proposing (Jev listwise about
  $2 per 1k queries at 100 candidates, pointwise similar; Kanishka paid $2.06 for 101k pairs) and let the user
  decide. API keys live in `.env`; a missing key means the run can't start.
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
A ranked list. Each item: why (which RQ/WP it serves), the exact command(s), rough time or API cost.
```

Commands use the project's scripts: `slurm/submit.sh rerank <system> --datasets <ds> [--after <ids>]
[--phrasings s1rank] [--depth 100]`, `slurm/submit.sh first-stage --datasets <ds>`, `slurm/submit.sh prefetch
--datasets ... --systems ...`; check a batch with `--dry-run` first. Don't submit, cancel or spend API money on
your own: propose, and act once the user agrees.

---
name: project-report
description: >-
  Project report for the decision-rerank paper: one Markdown overview of where the paper stands -- title, RQs,
  headline and claims, results per RQ (nDCG@10, joint vs. isolated, calibration, cost and efficiency), sanity checks,
  the roadmap with what is done, TODOs and draft markers in the paper, open questions and limitations -- published as
  a Claude artifact (Markdown) and kept in docs/notes/. Use it whenever the user wants an overview or summary of the
  paper project to share or discuss ("project report", "status of the paper", "overview for my meeting with
  Kanishka/my colleague", "what do we claim so far?", "update the report"), before a meeting, and after a batch of
  results or decisions changed the picture. For job/queue status alone use the experiment-status skill instead.
---

# Project report

The report is for people, mainly the two authors before a meeting: what the paper claims, what the numbers say, what
is decided, what is left. It is a snapshot; the plan of record stays `docs/roadmap.md`, decisions stay in
`docs/intent/`, the numbers stay in `results/`.

## 1. Collect the facts

```bash
python3 .claude/skills/project-report/scripts/facts.py      # login node is fine: csv, git, squeue only
```

It prints, as Markdown: title, venue and RQs (`paper/rq-macros.tex`); the owner's decisions
(`docs/intent/paper-focus.md`); the main table (joint mode, literature cells starred, best per column bold); RQ2
differences with window splits; RQ3 ECE; API cost, literature efficiency numbers and the fixed-GPU efficiency rows
(variant `a100*`); the roadmap; draft markers (`\TODO`, `\PENDING`, `\FLAG`) with file:line; limitations and open
items (`docs/notes/paper-notes.md`); jobs and git state.

Also read, for context the script does not parse: `docs/notes/paper-notes.md` (sanity checks, findings, source
register), the newest entries of `docs/experiment-log.md`, and -- if jobs are running or failed -- a quick
`experiment-status` run for what is in flight.

## 2. Write the report

Write `docs/notes/decision-rerank-status.md` (overwrite; git keeps the history). Numbers only from the facts output
or `results/*.csv` -- never typed from memory, never rounded differently from the tables. Mark literature numbers
as such. Keep it to what a reader needs in a 30-minute meeting:

```markdown
# Calibrated System One Models as Rerankers -- status <date>

> One paragraph: the paper in one sentence, the headline, where we are (grid done? what is running?), deadline.

## Research questions and headline
RQ1-RQ3 verbatim; the headline and the claims we can make now, each with its evidence and how firm it is
(supported / partly / open). Separate claims from hopes.

## Results
### RQ1 -- effectiveness       main table (joint mode) from the facts, plus 2-3 sentences of reading
### RQ2 -- joint vs. isolated   difference table + window splits, 2-3 sentences
### RQ3 -- calibration          ECE table (or the relevant slice), 2-3 sentences
### Cost and efficiency         measured API cost, literature cost/latency, efficiency run (or "pending")
### Sanity checks               one line each, pass/fail

## Decisions so far            short list with dates (from docs/intent/)
## TODOs                        roadmap steps not done, with owner if known; paper sections still \TODO/\PENDING
## Open questions for the meeting   3-6 concrete questions that need both authors
## Risks and limitations        the ones that could change a claim
```

Write plainly and briefly; tables over prose for numbers. If a claim rests on literature numbers or on unequal
hardware, say so next to it.

## 3. Publish

Publish the Markdown file as an artifact (this skill asks for Markdown, so a `.md` publish is correct here):
load the `artifact-design` skill first, then call the Artifact tool with `file_path` =
`docs/notes/decision-rerank-status.md`, `icon` = `report` on the first publish, and a one-sentence `description`.
Later updates go to the same artifact: its URL is recorded below -- read it first (`action: "read"`), then publish
with that `url`. After the first publish, write the URL into this file.

Artifact URL: https://claude.ai/artifact/6CdmvqL9JtfJ6AqFCKNFoh (published 2026-10-08)

Then commit the report (`docs/notes/decision-rerank-status.md`) with a short message, and give the user the link
and the 3-5 points that matter most for the meeting.

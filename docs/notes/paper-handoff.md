# Handoff: the ECIR short paper after the runs with the second question

Rewritten 2026-10-13 (the version of 2026-10-12, written for the session that drafted the paper, is in git history).
For a **new session that continues the paper**. Read this file first. The experiments for ECIR are done; do not redo
them. The paper is drafted, fits the page limit, and its RQ2 story has changed since the first draft (section 3).

## 1. Facts of the submission

- **ECIR 2027 short paper**: up to 6 pages plus references, Springer LNCS (`paper/` already uses `llncs.cls`,
  `splncs04.bst`), double-anonymous (no names, no repository URL, no self-identifying citations; a prior arXiv
  version is allowed but must not be cited). EasyChair, conference id ecir27.
- **Deadline 2026-10-12 23:59 AoE = 2026-10-13 13:59 CEST.** The abstract is registered.
- Build: `cd paper && make`; `make check` (errors, undefined references, overfull boxes); `make tables` regenerates
  `paper/tables/*.tex` via `python -m cdm.paper_tables`. On the login node `make` works, but `cdm.paper_tables` and
  `pytest` need pandas, which is only in the container:
  `slurm/exec.sh -- bash -c 'python -m cdm.paper_tables && PYTHONPATH=$PYTHONPATH:$CDM_ROOT/data/devtools python -m pytest -q -p no:cacheprovider tests'`
  (80 tests pass at `d5ff9f2`).
- **Page budget is tight.** At `d5ff9f2` the body ends exactly at the bottom of page 6 (page 6 = Table 3 + end of
  Sect. 5 + conclusion; references on pages 7-9). Every added line must be paid for elsewhere. Check with
  `pdftotext -f 6 -l 7 -layout main.pdf - | grep -n -E "Conclusion|References"`.
- Do not put `\FloatBarrier` back before the conclusion: with Table 3 at the top of page 6 it pushes the whole
  conclusion to page 7.

## 2. State of the repository

- Branch `dev-michael`, pushed to `origin`. Latest paper commits: `9a2ca92` (whole text revised for the second
  question), `d5ff9f2` (count error in Table 3 and Sect. 5). **Not yet merged into `master`** (Kanishka's branch;
  run `pytest` and merge `master` into `dev-michael` first, CLAUDE.md).
- `paper/main.pdf` is tracked in git since `9a2ca92` (force-added). If the owner does not want it tracked:
  `git rm --cached paper/main.pdf`.
- Kanishka revises on Overleaf. The repo draft is now the more recent full revision; his Overleaf edits need merging
  with it by hand. `docs/notes/paper-changes-answer-question.md` (the change list written for him) is **outdated**:
  the repo text has gone further (section 4 below).
- One draft marker left: `\TODO{anonymised link}` in `paper/sections/01_introduction.tex` (footnote for code and raw
  outputs; needs an anonymous repository URL, e.g. anonymous.4open.science, before submission).

## 3. What changed with the new results (read this before touching Sect. 5)

**The second question.** Kanishka's question, phrasing id `answer`: "Does document `{doc}` answer the `query`?" (base
models: systems `<name>-prompted-answer`). It was run on TREC DL 2019-2023 only, for every decision and base model
of the calibration table, in the same mode as the table row (experiment log, 2026-10-12 and 2026-10-13). RQ1 (ranking,
Tables 1 and 2) keeps S1Rank's question everywhere. Reranker fine-tunes ask no question, so their values do not
change. The question was fixed before the runs but **not chosen on development data** (owner's override of the
CLAUDE.md rule); the conclusion says so.

**The old RQ2 claim is gone.** Until 2026-10-12 the paper said "decision models are not better calibrated than
reranker fine-tunes of the same model" (`docs/intent/paper-focus.md` 14). That holds only with S1Rank's question.
With the stricter question the decision fine-tunes beat the published reranker fine-tunes from 4B on (ECE Kev-8B
0.113 vs Qwen3-Reranker-8B 0.257; Kev-4B 0.132 vs Prism-4B 0.218). Only 0.6B and 0.8B still favour the reranker.

**The RQ2 story now (owner, 2026-10-13; `docs/intent/paper-focus.md` 20):**
1. **Calibration depends on the wording.** The second question improves every decision and base model a lot (Jev
   ECE 0.224 → 0.075, Kev-8B 0.243 → 0.113), with hardly any change in ranking. So the calibration of decision models
   is not robust to the question. Caveat, stated in the text: the untuned base models move as much (Qwen3-8B
   0.383 → 0.231), so this is not special to decision fine-tuning; reranker fine-tunes have no question to word.
2. **Calibration buys little in practice.**
   - Fixed cutoff p ≥ 0.5: the reranker fine-tune keeps the better set on every foundation model, with either
     question (it ranks better). Described without numbers (numbers: `paper/tables/calibration-full.tex`, arXiv).
   - Count test (sum of a query's probabilities over its judged passages vs. its number of relevant ones; mean
     absolute error in passages; always guessing the median count: 8.6). With S1Rank's question no decision model is
     clearly better than the median guess (best: Clef 8.4). With the second question they are, and they beat the
     published reranker fine-tunes (Kev-8B 7.6 vs 12.9, Kev-4B 8.4 vs 11.5), **but** the untuned Qwen3.8-27B counts
     best (5.8) and Ours-9B beats Clef-Flash (7.3 vs 8.1).
   - Do not write "decision models do not benefit from calibration at all" or that the count test proxies ranking:
     it measures a use of calibration that ranking cannot give (how many passages to keep). The defensible claim is
     "decision fine-tuning is not what makes counting work".
3. **Our reranker fine-tune matches the best decision model of its size**: Ours-9B ECE 0.111 vs Clef-Flash 0.112
   (second question), and it ranks much better. Calibration comes from the training loss (yes/no BCE), not from
   decision fine-tuning.

**Abstract, introduction and conclusion** carry this framing: RQ1 negative; RQ2 "better calibrated only with a
question that matches the relevance definition, depends on the wording, does not make them select better passages,
matched by our reranker fine-tune; the untuned 27B counts best". The introduction calls both answers "mostly
negative".

## 4. The paper as it stands (`d5ff9f2`)

- **Title**: *Do System One Models Make Better Rerankers?* RQs in `paper/rq-macros.tex` only.
- **Wording rules** (owner): plain, short sentences; "foundation model" or "base model", not "backbone" (the PDF has
  none left); no dash asides.
- **Sections** (`paper/sections/`):
  - 01 introduction: hype, triplet design, two answers, footnote with `\TODO{anonymised link}`.
  - 02 related work: three short paragraphs (compressed for space).
  - 03 setup: triplets (incl. Ours-9B recipe), collections, decision models + the second question (one sentence),
    metrics (nDCG@10 with Holm-corrected t-tests, ECE, fixed cutoff F1, count error, short sanity checks). Removed
    for space: definitions of regret and post-hoc scaling.
  - 04 reranking: RQ1; joint vs. isolated by state length; Jev and Laya merged into the first paragraph.
  - 05 calibration: lead paragraph (S1Rank's question), "The question matters", "Uses of calibration".
  - 06 conclusion: answer plus one limitation sentence (second question not chosen on dev data; calibration only on
    TREC DL).
- **Tables** (generated; never edit by hand):
  - Table 1 `triplets.tex` (full width; DL20/DL22 hidden but in Avg; significance marks).
  - Table 2 `context-short.tex` (other systems).
  - Table 3 `calibration.tex`, two panels: groups "Relevant?" (S1Rank's question: ECE, Brier, Count) and "Answers?"
    (second question: ECE, Count). Rerankers' values repeated in brackets under "Answers?" and emphasised with the
    rest (best bold, second underlined, per foundation model). "Other systems" holds only Jev (monoT5, bge,
    Set-Encoder dropped for space). Generator: `calibration_ece_brier_table` in `src/cdm/paper_tables.py`; the count
    values come from the `job = cutoff` rows (`load_rerank_column(..., job="cutoff")`), the second question from
    phrasings `{"answer", "none"}`.
  - `calibration-full.tex`, `cutoff.tex`, `posthoc.tex`, `context.tex`: generated, not in the ECIR paper (arXiv).

## 5. Open items

- [ ] Anonymous repository link for the introduction's footnote (owner).
- [ ] Merge `dev-michael` into `master` after `pytest` (and after merging `master` in).
- [ ] Bring Kanishka's Overleaf edits and the repo draft together (owner decides which side is the base now).
- [ ] Owner may want to re-check the cuts made for space (related work, setup definitions, Table 3's other systems).
- [ ] **arXiv only, after ECIR:** our 27B reranker fine-tune (`qwen3.8-27b-reranker`). Training job 4429626
      COMPLETED (1-04:44:02); all checkpoints are in `checkpoints/qwen3.8-27b-reranker/` (200-1562, `final`). The
      resume job 4429627 ran on 2026-10-13 for 5 min and exited ("Already trained (1562/1562 steps)"). Next: checkpoint selection on msmarco-dev and scoring
      (`docs/notes/reranker-27b-plan.md`, section 5), then add it to the tables of the arXiv version.
- [ ] arXiv version: restore what was cut for ECIR (`calibration-full.tex`, cutoff/post-hoc numbers, regret,
      monoT5/bge/Set-Encoder in Table 3, the 27B fine-tune).

## 6. Rules that still apply (CLAUDE.md)

- Numbers in the paper come only from generated tables; numbers quoted in the text must be in those tables.
- Never edit `results/results.csv` by hand. Literature numbers are labelled as such.
- No computation on the login node; `slurm/exec.sh` / `slurm/submit.sh`.
- Commit on `dev-michael` with the attribution lines from the session's system reminder; push only when the owner
  asks (pushes of `dev-michael` have been approved so far).

## 7. Read next

- `docs/intent/paper-focus.md`, items 13-20 (the owner's decisions for this paper).
- `docs/experiment-log.md`, entries of 2026-10-12 and 2026-10-13 (the second question, all numbers).
- `paper/sections/05_calibration.tex` and `paper/tables/calibration.tex`.
- `docs/notes/paper-changes-answer-question.md`: number tables per foundation model for both questions (the text
  proposals in it are outdated).

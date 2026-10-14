# Handoff: the arXiv paper

**Update 2026-10-14 (branch `feature/arxiv-acm`).** The paper is now in the two-column ACM layout (acmart sigconf,
nonacm) with the ECIR narrative, the `answer` question in every result and S1Rank's question only in the ablation
(Sect. 7, `tables/question-ablation.tex`); Ours-27B is in every table (ties Kev-27B in ranking, worse calibrated).
Sections: 00 abstract, 01 introduction (teaser `figures/teaser.tex`), 02 related work, 03 decision models as
rerankers, 04 setup, 05 RQ1, 06 RQ2, 07 question ablation, 08 discussion and limitations, 09 conclusion. Open:
authors and affiliations, subtitle (optional), repository link, acknowledgements (`\TODO`s in `main.tex` and the
introduction), Kanishka's Overleaf edits, merge into `dev-michael`/`master`. Sections 2-3 below describe the state
before this update (two questions in Table 3, LNCS) and are kept for the history.

Rewritten 2026-10-13, after the ECIR 2027 short paper was submitted (tag `ecir2027-submission`; the ECIR handoff is
in the history of this file). For a **new session that continues the paper**: read this file, then `docs/roadmap.md`
("Order of work"). The experiments of the ECIR paper are done; do not redo them. The arXiv paper is the ECIR paper
plus what was cut for the 6-page limit and our 27B reranker fine-tune.

## 1. Build and checks

- `cd paper && make`; `make check` (errors, undefined references, overfull boxes); `make tables` regenerates
  `paper/tables/*.tex` via `python -m cdm.paper_tables`. On the login node `make` works, but `cdm.paper_tables` and
  `pytest` need pandas, which is only in the container:
  `slurm/exec.sh -- bash -c 'python -m cdm.paper_tables && PYTHONPATH=$PYTHONPATH:$CDM_ROOT/data/devtools python -m pytest -q -p no:cacheprovider tests'`
- No page limit any more: the space cuts listed in section 3 can be undone.
- `paper/main.pdf` is force-added (it is in `.gitignore`); `git rm --cached paper/main.pdf` if it should not be tracked.

## 2. The RQ2 story (read this before touching Sect. 5)

**The second question.** Kanishka's question, phrasing id `answer`: "Does document `{doc}` answer the `query`?" (base
models: systems `<name>-prompted-answer`). It was run on TREC DL 2019-2023 only, for every decision and base model
of the calibration table, in the same mode as the table row (experiment log, 2026-10-12 and 2026-10-13). RQ1 (ranking,
Tables 1 and 2) keeps S1Rank's question everywhere. Reranker fine-tunes ask no question, so their values do not
change. The question was fixed before the runs but **not chosen on development data** (owner's override of the
CLAUDE.md rule); the conclusion says so.

**"Not better calibrated" holds only with S1Rank's question.**
With the stricter question the decision fine-tunes beat the published reranker fine-tunes from 4B on (ECE Kev-8B
0.113 vs Qwen3-Reranker-8B 0.257; Kev-4B 0.132 vs Prism-4B 0.218). Only 0.6B and 0.8B still favour the reranker.

**The RQ2 story now (owner, 2026-10-13; `docs/intent/paper-focus.md` 16):**
1. **Calibration depends on the wording.** The second question improves every decision and base model a lot (Jev
   ECE 0.224 → 0.075, Kev-8B 0.243 → 0.113), with hardly any change in ranking. So the calibration of decision models
   is not robust to the question. Caveat, stated in the text: the untuned base models move as much (Qwen3-8B
   0.383 → 0.231), so this is not special to decision fine-tuning; reranker fine-tunes have no question to word.
2. **Calibration buys little in practice.**
   - Fixed cutoff p ≥ 0.5: the reranker fine-tune keeps the better set on every foundation model, with either
     question (it ranks better). In the ECIR version without numbers; the arXiv version shows them (`paper/tables/calibration-full.tex`).
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

## 3. The paper as submitted to ECIR (tag `ecir2027-submission`)

- **Title**: *Do System One Models Make Better Rerankers?* RQs in `paper/rq-macros.tex` only.
- **Wording rules** (owner): plain, short sentences; "foundation model" or "base model", not "backbone" (the PDF has
  none left); no dash asides.
- **Sections** (`paper/sections/`):
  - 01 introduction: hype, triplet design, two answers, footnote with `\TODO{repository link}`.
  - 02 related work: three short paragraphs (compressed for space).
  - 03 setup: triplets (incl. Ours-9B recipe), collections, decision models + the second question (one sentence),
    metrics (nDCG@10 with Holm-corrected t-tests, ECE, fixed cutoff F1, count error, short sanity checks). Removed
    for space: definitions of regret and post-hoc scaling (restore for arXiv).
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
  - `calibration-full.tex`, `cutoff.tex`, `posthoc.tex`, `context.tex`: generated, not in the ECIR paper; they go
    into the arXiv version.

## 4. Open items

The steps are in `docs/roadmap.md`, "Order of work". In short:

- [ ] 27B reranker fine-tune: checkpoint selection on msmarco-dev, scoring, calibration and post-hoc rows, tables
      (`docs/notes/reranker-27b-plan.md`, sections 4 and 6).
- [ ] Restore what was cut for ECIR (section 3; `docs/intent/paper-focus.md` 18).
- [ ] De-anonymise: authors, affiliations, the disclosure of interests (`paper/main.tex`), the repository link in the
      introduction's footnote.
- [ ] Merge Kanishka's Overleaf edits into `paper/` (by hand).
- [ ] Merge `dev-michael` into `master` after `pytest` (merge `master` in first; CLAUDE.md).

## 5. Rules that still apply (CLAUDE.md)

- Numbers in the paper come only from generated tables; numbers quoted in the text must be in those tables.
- Never edit `results/results.csv` by hand. Literature numbers are labelled as such.
- No computation on the login node; `slurm/exec.sh` / `slurm/submit.sh`.
- Commit on `dev-michael` (or a feature branch) with the attribution lines from the session's system reminder.

## 6. Read next

- `docs/intent/paper-focus.md` (the owner's decisions).
- `docs/experiment-log.md`, entries of 2026-10-12 and 2026-10-13 (the second question, all numbers).
- `paper/sections/05_calibration.tex` and `paper/tables/calibration.tex`.

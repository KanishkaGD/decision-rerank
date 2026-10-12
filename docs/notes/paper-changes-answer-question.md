# Paper changes for RQ2 after the runs with Kanishka's question

For Kanishka, who revises the paper on Overleaf. Written 2026-10-12 against the draft in this repository (commit
446cfcb, `paper/`). The Overleaf copy may have moved on; the sentences quoted below are the ones to look for.

## What happened

RQ2 was rerun on TREC DL 2019–23 with Kanishka's question ("Does document … answer the query?", phrasing `answer`)
instead of S1Rank's. Decision models were run in joint mode. The base model used the same question (system
`qwen3.8-27b-prompted-answer`). Reranker fine-tunes ask no question, so their numbers do not change.

The cluster runs reproduce Kanishka's local numbers within .002 ECE and 0.1 passages.

**Consequence: four claims in the draft are no longer true.** With the new question, Jev, Kev-27B and Clef are
better calibrated than our 9B reranker fine-tune. RQ2 stays negative, but the argument has to change, from "reranker
fine-tunes are better calibrated" to:

1. **Same foundation model, no advantage.**
   - 4B: Kev-4B vs Prism-4B ties (ECE .201 vs .218, Brier .199 vs .172, count error 11.6 vs 11.5), and Prism-4B ranks
     10 points better.
   - 9B: Clef-Flash vs our 9B fine-tune ties (ECE .112 vs .111, Brier .158 vs .157); ours counts better (8.1 vs 7.3)
     and ranks better.
2. **Not from decision fine-tuning.** The untuned Qwen3.8-27B gains as much from the better question (ECE .166 →
   .117) and counts best of all systems (5.8).
3. **Fragile.** Decision-model calibration depends on the wording of the question (Jev .224 → .075).

## Numbers

Means over DL19–23 on the judged pairs, relevant = grade ≥ 2. ECE uses 15 bins. Count error = mean absolute error
of the summed probabilities against the true number of relevant passages per query, in passages. Always guessing
the median count gives 8.6.

| Foundation model | System | ECE, S1Rank's question | ECE, new question | Brier, new question | Count error, new question | nDCG@10 (DL), new question |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | Kev-4B (joint) | .251 | .201 | .199 | 11.6 | 51.9 |
| Qwen3.5-4B | Prism-4B | .218 | (.218) | (.172) | 11.5 | 62.0 |
| Qwen3.5-9B | Clef-Flash (joint) | .140 | .112 | .158 | 8.1 | 56.6 |
| Qwen3.5-9B | Kev-9B (joint) | .237 | *running* | *running* | *running* | *running* |
| Qwen3.5-9B | Prism-9B | .242 | (.242) | (.183) | 12.3 | 62.4 |
| Qwen3.5-9B | Ours-9B | .111 | (.111) | (.157) | 7.3 | 62.0 |
| Qwen3.8-27B | base, zero-shot | .166 | .117 | .144 | 5.8 | 59.6 |
| Qwen3.8-27B | Kev-27B (joint) | .156 | .076 | .130 | 6.0 | 63.3 |
| Qwen3.8-27B | Clef (joint) | .147 | .091 | .139 | 6.7 | 60.6 |
| other | Jev (joint) | .224 | .075 | .127 | 6.3 | 62.2 |
| other | Qwen3-Reranker-8B | .257 | (.257) | (.238) | 12.9 | 60.7 |

Values in brackets are reranker fine-tunes, which ask no question; they repeat the S1Rank column.
- **Kev-9B joint:** jobs 4451209–4451213 are still running. The number goes into `results/results.csv`; run
  `cdm cutoff` for its count error.
- **Ranking:** almost unchanged by the question (Kev-27B DL mean 63.3 vs 63.5).
- **Sources:**
  - ECE, Brier and nDCG come from the rerank rows of `results/results.csv` with `phrasing = answer` (system
    `qwen3.8-27b-prompted-answer` for the base model).
  - Count errors come from the `job = cutoff` rows of the same file.
  - The repo's generated tables do not show the new question yet.

## Text changes

### Abstract (`main.tex`)

Replace the last sentence:
> Decision models are not better calibrated either: with a fixed probability threshold, reranker fine-tunes select
> better passages, and our own reranker fine-tune gives the most reliable probabilities of all systems we tested.

with:
> Decision models are not better calibrated either: on the same foundation model they are no better calibrated than
> reranker fine-tunes, their calibration depends on how the question is worded, and the untuned model profits just
> as much from a better question.

### Introduction (`sections/01_introduction.tex`), finding (2)

Replace:
> (2) Decision models are not better calibrated than reranker fine-tunes of the same model. Our own reranker
> fine-tune of Qwen3.5-9B ranks as well as the best published one and is the best calibrated system we tested.

with:
> (2) Decision models are not better calibrated than reranker fine-tunes of the same model. How well they are
> calibrated depends on how the question is worded, and the untuned model profits from a better question as much as
> they do. Our own reranker fine-tune of Qwen3.5-9B ranks as well as the best published one and is as well
> calibrated as the best decision model of the same size.

### Sect. 5 (`sections/05_calibration.tex`), first paragraph

Replace:
> Only at 9B is Clef-Flash better than Prism (ECE 0.140 vs. 0.242), and there Ours-9B has the lowest ECE of the
> study (0.111, below every decision model including the 27B ones and Jev) and a Brier score as low as the best
> decision model (0.157). Good calibration thus comes from a loss that treats the score as a probability, not from
> decision fine-tuning.

with:
> Only at 9B is Clef-Flash better than Prism (ECE 0.140 vs. 0.242), and there Ours-9B is better calibrated than
> both (0.111, Brier 0.157). A loss that treats the score as a probability thus gives a calibrated reranker without
> any decision fine-tuning.

### Sect. 5, new paragraph after the first

> **The question matters.** S1Rank's question also counts passages that only partly address the query, while we
> score relevance from grade 2 (the passage answers the query). Asking whether the passage *answers* the query
> improves the decision models' calibration a lot (Table X: Jev 0.224 → 0.075, Kev-27B 0.156 → 0.076) and leaves
> their ranking almost unchanged. With this question the 27B decision models and Jev are the best calibrated systems
> of the study. But the untuned Qwen3.8-27B improves as well (0.166 → 0.117). On the same foundation model, decision
> fine-tuning still brings no advantage: Kev-4B is not better than Prism-4B (ECE 0.201 vs. 0.218, Brier 0.199 vs.
> 0.172), and Clef-Flash is not better than Ours-9B (0.112 vs. 0.111, Brier 0.158 vs. 0.157). The calibration of
> decision models is thus a property of model size and of the question's wording, not of decision fine-tuning.

Add the Kev-9B joint numbers once its runs are in. Bold or underline them only if they change the reading.

### Sect. 5, "Uses of calibration"

Replace:
> Summing a query's probabilities estimates its number of relevant passages about equally well for decision and
> reranker fine-tunes, and best for Ours-9B.

with:
> Summing a query's probabilities estimates its number of relevant passages about equally well for decision and
> reranker fine-tunes of the same model (with the new question: Kev-4B 11.6 vs. Prism-4B 11.5 passages off, Clef-Flash
> 8.1 vs. Ours-9B 7.3); the untuned Qwen3.8-27B counts best of all (5.8), so counting well comes with model size.

Also, last sentence of the paragraph, replace:
> A well calibrated model is better, but it need not be a decision model, as Ours-9B shows.

with:
> A well calibrated model helps, but it need not be a decision model.

### Conclusion (`sections/06_conclusion.tex`)

Replace:
> Their calibration is no advantage either: reranker fine-tunes are as well calibrated, select better passages at a
> fixed threshold, and can be trained to be better calibrated than any decision model.

with:
> Their calibration is no advantage either: on the same foundation model reranker fine-tunes match it, it depends
> on how the question is worded, and a reranker can be trained to be calibrated.

Add to the limitations, if there is a sentence for them:
> The second question was fixed before any run but not chosen on a development set, and it was only evaluated on
> TREC DL.

### Setup (`sections/03_setup.tex`)

Where the question is introduced (S1Rank's question for all System One models), add:
> For RQ2 we additionally ask whether the passage answers the query ("Does document … answer the query?"), matching
> relevance grade 2; base models get the same question in their chat prompt.

## Table change -- done in the repo

`paper/tables/calibration.tex` (generated, commit after this note) now has the column **ECE$'$** and a shortened
caption; copy the whole file into Overleaf. With tighter column spacing both panels still fit side by side and the
body still ends on page 6 (built here). Kev-9B's ECE$'$ shows "--" until its runs are done; the file is regenerated
then. What was planned:

Add one column, **ECE with the new question**, to the calibration table (Table 3, `tables/calibration.tex`). Show it
for the rows that have it: Kev-4B, Clef-Flash, Kev-9B (when done), the base Qwen3.8-27B, Kev-27B, Clef and Jev.
Reranker rows repeat their value in brackets, or get "--" with a caption note "rerankers ask no question". The
caption needs one more sentence: "ECE′: with the question whether the passage answers the query (Sect. 5)."

**Caveat on mode:** Table 3 shows each decision model in its better-calibrated mode. For Kev-4B that is isolated
(0.216), but the new-question runs are joint, so their S1Rank reference is joint (.251).

If space is too tight for a column: no new table; put the numbers only in the new paragraph above.

**Page budget:** the body currently ends on page 6 and the references start on page 7. The new paragraph adds about
8–9 lines. It may need trimming elsewhere, e.g. the cutoff sentences in "Uses of calibration".

## Not to forget

- The code and results are pushed on `dev-michael`: phrasing `answer` in `src/cdm/phrasings.py`, the base-model
  templates in `src/cdm/rerankers/prompted.py`, and the log entry of 2026-10-12 in `docs/experiment-log.md`.
- Kev-9B joint with the new question is still running (jobs 4451209–4451213).

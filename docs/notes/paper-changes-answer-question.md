# Paper changes for RQ2 after the runs with Kanishka's question (updated 2026-10-13)

For Kanishka, who revises the paper on Overleaf. This replaces the version of 2026-10-12: the complete runs
**change the RQ2 reading**, and the replacement texts of that version are no longer correct.

## What happened

RQ2 was run on TREC DL 2019–23 with Kanishka's question ("Does document … answer the query?", phrasing `answer`)
for every decision model and base model of the calibration table, in the same mode as the table row. Reranker
fine-tunes ask no question, so their numbers do not change. The runs reproduce Kanishka's local numbers within .002
ECE.

## Numbers (means over DL19–23, judged pairs, relevant = grade ≥ 2)

ECE and Brier: S1Rank's question / the new question. Count = mean absolute error of the summed probabilities
against the number of relevant passages, in passages, with the new question (median guess: 8.6). Rerankers:
question-independent.

| Foundation model | System (mode) | ECE S1Rank → new | Brier S1Rank → new | Count, new | Same-backbone reranker: ECE / Brier / count |
| --- | --- | --- | --- | ---: | --- |
| Qwen3-0.6B | base | .750 → .750 | .750 → .750 | 36.5 | |
| Qwen3-0.6B | Kev-0.6B (i) | .579 → .503 | .543 → .463 | 24.7 | Qwen3-Reranker-0.6B .511 / .481 / 24.8 |
| Qwen3-8B | base | .383 → .231 | .380 → .233 | 8.7 | |
| Qwen3-8B | Kev-8B (j) | .243 → **.113** | .248 → **.177** | **7.6** | Qwen3-Reranker-8B .257 / .238 / 12.9 |
| Qwen3.5-0.8B | base | .716 → .617 | .694 → .566 | 29.9 | |
| Qwen3.5-0.8B | Kev-0.8B (i) | .357 → .336 | .309 → .297 | 16.9 | Prism-0.8B **.243 / .193 / 12.7** |
| Qwen3.5-4B | base | .296 → .181 | .269 → .195 | 9.8 | |
| Qwen3.5-4B | Kev-4B (i) | .216 → **.132** | .207 → .175 | **8.4** | Prism-4B .218 / **.172** / 11.5 |
| Qwen3.5-9B | base | .283 → .165 | .265 → .181 | 8.7 | |
| Qwen3.5-9B | Kev-9B (j) | .237 → .160 | .209 → .177 | 9.5 | Prism-9B .242 / .183 / 12.3 |
| Qwen3.5-9B | Clef-Flash (j) | .140 → .112 | .165 → .158 | 8.1 | **Ours-9B .111 / .157 / 7.3** |
| Qwen3.8-27B | base | .166 → .117 | .180 → .144 | **5.8** | |
| Qwen3.8-27B | Kev-27B (j) | .156 → .076 | .159 → .130 | 6.0 | (none) |
| Qwen3.8-27B | Clef (j) | .147 → .091 | .157 → .139 | 6.7 | (none) |
| other | Jev (j) | .224 → .075 | .187 → .127 | 6.3 | |

Ranking with the new question is slightly lower everywhere, so Table 1 (S1Rank's question) shows the decision
models at their best. DL mean nDCG@10, S1Rank → new: Kev-27B 63.5 → 63.3, Jev 62.7 → 62.2, Clef 61.1 → 60.6,
base 27B 59.9 → 59.6, Clef-Flash 57.0 → 56.6, Kev-4B joint 53.4 → 51.9, Kev-9B joint 54.0 → 51.9.

## What this means for RQ2 -- the reading changes

**With the new question, the published reranker fine-tunes lose the calibration comparison:**
- **8B:** Kev-8B is better than Qwen3-Reranker-8B on all three measures (ECE .113 vs .257, Brier .177 vs .238, count
  7.6 vs 12.9).
- **4B:** Kev-4B has a clearly lower ECE than Prism-4B (.132 vs .218) and counts better (8.4 vs 11.5); Brier ties
  (.175 vs .172).
- **9B:** both decision fine-tunes beat Prism-9B.
- **Small sizes:** only at 0.6B (tie) and 0.8B (Prism better) do the reranker fine-tunes hold.

The draft's sentence "decision models are not better calibrated than reranker fine-tunes of the same model" is
therefore **false with the new question**. It is only true with S1Rank's question.

**What still holds, under both questions:**
1. **Ranking (RQ1):** reranker fine-tunes rank 4–29 points better.
2. **Our own reranker fine-tune** is as well calibrated as the best decision fine-tune of the same model (Ours-9B vs
   Clef-Flash: ECE .111 vs .112, Brier .157 vs .158, count 7.3 vs 8.1). It also ranks much better (62.0 vs 56.6
   DL mean). The calibration advantage is thus a property of the training loss, not of decision fine-tuning, and a
   reranker can have both.
3. **The question matters, for decision and base models alike.** Every base model improves with the new question
   (e.g. Qwen3-8B .383 → .231, Qwen3.5-9B .283 → .165). The base Qwen3.8-27B counts best of all (5.8). Decision
   fine-tuning adds calibration on top of that from 4B upwards (Kev-8B .113 vs its base .231).

**Proposed RQ2 answer (an honest, less negative version):**
> Decision models are better calibrated than the published reranker fine-tunes of the same model, but only when the
> question matches the relevance definition being scored. That advantage is not specific to decision fine-tuning: a
> reranker fine-tune trained with a probability loss (ours) is equally well calibrated and ranks much better.

**Overall answer to "are decision models of use to IR?":** as rerankers, no. Their calibration is a real but
replaceable property: it needs the right question, and a reranker fine-tune can have it too. The owner should decide
whether the headline "negative on both counts" still stands; it is now negative on RQ1 and "not specific" on RQ2.

## Text changes (to adapt to the decision above)

Quoted sentences are from the repo draft (commit 446cfcb); the Overleaf text may differ.

- **Abstract, last sentence** (currently: "Decision models are not better calibrated either: … our own reranker
  fine-tune gives the most reliable probabilities of all systems we tested.") Proposal:
  > Decision models are better calibrated than published reranker fine-tunes only if the question matches the
  > relevance being scored, and a reranker fine-tune trained with a probability loss, as ours is, matches their
  > calibration while ranking better.
- **Introduction, finding (2)** (currently: "Decision models are not better calibrated than reranker fine-tunes of the
  same model. … is the best calibrated system we tested.") Proposal:
  > (2) Decision models can be better calibrated than reranker fine-tunes of the same model, but only with a question
  > that matches the relevance definition, and not better than our own reranker fine-tune of Qwen3.5-9B, which is
  > as well calibrated as the best decision model of its size and ranks as well as the best published reranker.
- **RQ2 wording** ("Are decision models at least better calibrated than reranker fine-tunes of the same foundation
  model?") can stay; the answer becomes "yes, with the right question, but not specifically".
- **Sect. 5:**
  - First paragraph: drop "Ours-9B has the lowest ECE of the study (…below every decision model including the 27B
    ones and Jev)"; instead: "Ours-9B is as well calibrated as Clef-Flash and better than Prism-9B."
  - New paragraph "The question matters", with the table above in short form (Table 3 now has the column ECE′).
  - In "Uses of calibration", replace "best for Ours-9B" with the count numbers above. With the new question Kev-8B
    and Kev-4B count better than their published reranker fine-tunes; the base 27B counts best of all.
- **Conclusion** (currently: "Their calibration is no advantage either: reranker fine-tunes are as well calibrated,
  select better passages at a fixed threshold, and can be trained to be better calibrated than any decision model.")
  Proposal:
  > Their calibration is real but not unique: it depends on asking the right question, and a reranker fine-tune
  > trained with a probability loss reaches it while ranking better.
- **Setup:** one sentence on the second question. Add one limitation sentence: it was fixed before any run, but not
  chosen on a development set, and it was only evaluated on TREC DL.

## Table 3 (`paper/tables/calibration.tex`, generated in the repo)

- The column **ECE′** is complete for every decision model and base model. Reranker cells show their own ECE in
  brackets.
- The caption explains ECE′.
- Copy the whole file into Overleaf.
- **Suggested in the chat but not yet done:** group headers ("S1Rank's question" over ECE/Brier, "answers the query"
  over ECE′).

**Page budget:** with the column filled, the conclusion's last two lines spill onto page 7 (the table needs tighter
spacing, 2pt). The text changes above will need some trimming anyway.

# System One Models Are Not Better Rerankers — status 2026-10-14

> **The paper in one sentence:** on the same backbone, decision fine-tuning never makes a better reranker than plain
> reranker fine-tuning, and the calibration it is advertised for comes from how the question is worded, not from the
> model. **Where we are:** every experiment is done, including our 27B reranker fine-tune (all ten collections,
> post-hoc scaling, significance tests). The draft is in the two-column ACM layout on branch `feature/arxiv-acm`
> (9 sections, teaser figure, 4 generated tables). Left before arXiv: Kanishka's surname and the emails, the
> repository link, acknowledgements, Kanishka's Overleaf edits, a final read. Target: arXiv, end of the week of
> 2026-10-12.

## The narrative: three punchlines

Fixed in `docs/notes/narrative.md` (owner, 2026-10-14). Abstract, introduction and conclusion say exactly these, in
this order and without hedging chains; everything else is a result in its section.

| # | Punchline | Evidence (paper's question, Tables 1, 3, 4) | Firmness |
|---|---|---|---|
| 1 | **Reranker fine-tuning beats decision fine-tuning.** On no backbone does decision fine-tuning make the better reranker. | 0.6B–9B: reranker fine-tune ahead by 5.2–28.8 points nDCG@10 (Avg of ten), 5.4–23.9 on the zero-shot subset; significant on 48 of 60 backbone–collection pairs, never significantly worse. 27B: tie, Ours-27B 59.6 vs Kev-27B 60.1 (ZS 54.9 vs 55.0), significant on one collection only (DBPedia, ours better). | Supported |
| 2 | **Up to 9B, decision fine-tuning adds nothing over the base model.** | Better decision mode vs base asked the same question: −2.5 to +0.7 points; reranker fine-tuning adds 5.9–29.5. At 27B both add about the same: Kev-27B +3.7 (significant on 8 collections), Ours-27B +3.2; Clef adds nothing (54.8 vs 56.4). | Supported |
| 3 | **The calibration advantage comes from the question, not from the model.** | With "Does the passage answer the query?": decision fine-tunes beat the published rerankers from 4B (Kev-8B ECE .113 vs Qwen3-Reranker-8B .257). With S1Rank's question the same models are at best as well calibrated as the best reranker fine-tune of their backbone (Kev-8B .243 vs .257, Kev-27B .156 vs Ours-27B .110), while nDCG@10 moves by only −2.5 to +2.6. Base models move as much (Qwen3-8B .383 → .231). Up to 9B, a reranker fine-tune keeps the better set at p ≥ 0.5 (e.g. Prism-4B F1 51.0 vs Kev-4B 42.6). | Supported (two wordings only; the question was fixed before the runs, not chosen on dev data) |

Closing line of abstract, introduction and conclusion: **"If you need a reranker, fine-tune a reranker."**

Title: **System One Models Are Not Better Rerankers** (no subtitle). It states punchline 1, which holds on all six
backbones (worse up to 9B, a tie at 27B — "not better" in both cases).

## Research questions

- **RQ1:** Do decision models rerank better than their base models or than reranker fine-tunes of the same backbone?
  → No (punchlines 1 and 2).
- **RQ2:** Are their relevance probabilities better calibrated than either, and is that calibration of any use?
  → Only with a question that matches the labels, and up to 9B it does not buy better passage selection
  (punchline 3).

## Results

All numbers from `paper/tables/*.tex` (generated from `results/results.csv`), the paper's question (`answer`).

### RQ1 — triplets (nDCG@10 × 100; Avg of ten collections, ZS = TREC-COVID, Touché, DBPedia)

| Backbone | Base | Best decision fine-tune | Reranker fine-tune(s) | Reranker − decision |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 26.6 | Kev isolated 27.3 | Qwen3-Reranker 56.1 | +28.8 |
| Qwen3-8B | 51.1 | Kev isolated 51.7 | Qwen3-Reranker 59.4 | +7.7 |
| Qwen3.5-0.8B | 33.9 | Kev isolated 34.2 | Prism 57.1 | +22.9 |
| Qwen3.5-4B | 52.4 | Kev isolated 49.9 | Prism 58.7 | +8.8 |
| Qwen3.5-9B | 53.1 | Clef-Flash isolated 53.8 | Prism 59.0 · *Ours* 59.4 | +5.2 · +5.6 |
| Qwen3.8-27B | 56.4 | Kev joint 60.1 | *Ours* 59.6 | −0.5 (n.s. except DBPedia, ours better) |

Ours-9B matches Prism-9B (59.4 vs 59.0; ZS 56.1 vs 53.7) with a public recipe and no decision data, so the gap is not
about the undisclosed recipes of the published rerankers.

### RQ1 — joint vs isolated

Joint scoring (all 100 candidates in one state) helps only the large models: Kev-27B +1.1 (60.1 vs 59.0), Clef +0.9,
Jev +2.0 (59.1 vs 57.1; joint better on 10/10 collections). It hurts the small Kevs badly: Kev-0.6B −12.2 (worse on
all ten, significant), Kev-8B −13.0, Kev-0.8B −6.3. Reading: joint states (≈13k tokens on DL, 40k+ on BEIR) exceed
the state lengths the small Kevs were trained on (≤ 7,552 tokens) but not Kev-27B's (≤ 32,768). Window splits do not
explain it (Kev: 0 splits except Touché).

### RQ1 — context (Table 2, System One models in joint mode)

Jev 59.1 ≈ reranker fine-tunes of 8–9B (59.0–59.4) ≈ Set-Encoder 59.0; Kev-27B 60.1 and Ours-27B 59.6 match the
closed Jev; Laya 43.5 ≈ BM25 43.2. RankZephyr 57.3, monoT5-3B 57.5, bge-reranker-v2-m3 56.6 (BEIR cells of these
three partly literature, marked † in the paper).

### RQ2 — calibration and its uses (Table 3, judged pairs of DL19–23)

- **ECE (paper's question):** Jev .075, Kev-27B .076, Clef .091 are the best calibrated. Ours-9B .111 = Clef-Flash
  .112 (Brier .157 vs .158). At 27B Kev-27B .076 beats Ours-27B .110. Small base models say "yes" to everything
  (.750, .617).
- **Fixed cutoff p ≥ 0.5:** up to 9B a reranker fine-tune keeps the better set on every backbone, although it is the
  worse calibrated model on most of them. Decision models' regret is small (0.9–3.1 points), Prism-9B 4.0, Ours-9B
  6.4. At 27B Kev-27B keeps the better set (57.2 vs 48.3; ours underconfident, 7.9 passages kept).
- **Count test:** most decision models from 4B beat the median guess (8.6) and the published rerankers; the untuned
  Qwen3.8-27B counts best (5.8); Ours-9B beats Clef-Flash (7.3 vs 8.1).
- **Post-hoc scaling (MS MARCO dev):** temperature scaling helps only the worst calibrated rerankers
  (Qwen3-Reranker-0.6B .511 → .326, Set-Encoder .495 → .250). Platt scaling collapses F1@.5 to ≤ 8.1 for every
  reranker: MS MARCO dev's sparse labels (≈1 relevant per query) set the wrong base rate for DL.

### Ablation — the wording (Table 4, S1Rank's question vs ours)

| Model | nDCG@10 Avg S1Rank → ours | ECE S1Rank → ours | Count S1Rank → ours |
|---|---|---|---|
| Jev joint | 60.5 → 59.1 | .224 → .075 | 11.3 → 6.3 |
| Kev-27B joint | 61.1 → 60.1 | .156 → .076 | 8.5 → 6.0 |
| Kev-8B joint | 38.8 → 38.7 | .243 → .113 | 12.8 → 7.6 |
| Qwen3-8B (base) | 53.3 → 51.1 | .383 → .231 | 17.5 → 8.7 |

Ranking moves by at most a few points; calibration by a factor of two to three. With S1Rank's question no decision
model beats the median count guess by more than a fraction of a passage (best: Clef 8.4 vs 8.6).

### Cost and efficiency (not in the paper)

Jev API cost for the whole grid: $3.16 (joint), $6.23 (isolated). On one A100 (DL19, p50 s/query), Kev is about
4–5× slower than the reranker fine-tune of the same backbone (Kev-9B joint 16.7 s vs Prism-9B 4.3 s; Kev-4B joint
10.1 s vs Prism-4B 2.6 s; Kev-8B joint 13.7 s vs Qwen3-Reranker-8B 2.9 s), but Clef-Flash joint is faster than
Prism-9B (2.1 s). So the slowness is Kev's inference path, not decision models as such: not a punchline.

### Sanity checks

- BM25, monoT5-3B, RankZephyr reproduce published numbers within 0.2 points — pass.
- Jev joint with S1Rank's question reproduces S1Rank within 0.5 points on four of five collections (1.5 below on
  TREC-COVID) — pass.
- 27B checkpoint chosen on msmarco-dev by a rule fixed in advance (final, MRR@10 .417) — pass.

## Decisions so far (docs/intent/paper-focus.md)

- 2026-10-09: negative headline; triplets on one backbone; joint vs isolated inside RQ1.
- 2026-10-10: train our own reranker fine-tunes (9B validation passed, 27B trained).
- 2026-10-13: ECIR short paper submitted (tag `ecir2027-submission`); Kanishka's question (`answer`) for all
  results (decision 20).
- 2026-10-14: two-column ACM layout; S1Rank's question only in the ablation (21); title *System One Models Are Not
  Better Rerankers*, three authors (Dinzinger, Kanishka, Granitzer), the three-punchline narrative (22).

## TODOs

- Kanishka's surname and email, Michael Granitzer's email (`paper/main.tex`).
- Repository link (code release branch `feature/release-20261012`) in the introduction's footnote.
- Acknowledgements / compute / disclosure (`paper/main.tex`).
- Merge Kanishka's Overleaf edits; final read; `make check`.
- Merge `feature/arxiv-acm` into `dev-michael` and `master` (commit or discard the uncommitted `results.csv` in the
  main checkout first; the branch has the same rows committed).

## Open questions for the meeting

1. Title: *System One Models Are Not Better Rerankers* — happy with the claim at 27B being a tie?
2. Efficiency stays out of the paper (Kev is slow, Clef is not)? Or a small appendix table?
3. Terminology: the ECIR text uses "backbone"; decision 17 asked for "foundation model". Keep "backbone"?
4. Do we state the 27B calibration result (Kev-27B better calibrated than ours) in the abstract, or only in Sect. 6
   as now?
5. Should a third wording (chosen on msmarco-dev) be run to make punchline 3 firmer than two questions?

## Risks and limitations

- Punchline 3 rests on two wordings; the paper's question was fixed before the runs but not chosen on dev data.
- We compare released models: the decision fine-tunes are not RLCD, and training data differ across fine-tunes.
- Our reranker fine-tunes were trained once, with one recipe, no hyperparameter search.
- Calibration only on TREC DL (judged non-relevant passages), one relevance grade, one threshold.
- The models are a month old; newer releases may change the picture.

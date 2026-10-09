# Do System One Models Make Better Rerankers? Status 2026-10-09 (evening)

> On one backbone at a time, we compare three training objectives: none (the base model asked S1Rank's question
> zero-shot), decision fine-tuning (System One models: Kev, Clef) and reranker fine-tuning (Qwen3-Reranker,
> Prism). **Headline: a negative result.** Reranker fine-tuning beats decision fine-tuning on every backbone where
> both exist (0.6B–9B), by 4–29 nDCG@10 points on average and also on the collections without a training split.
> Decision fine-tuning moves the base model by only −3.6 to +3.7 points. **RQ2** asks whether the one thing decision
> models have and reranker fine-tunes lack, calibrated probabilities, is still of use. **New since the morning:**
> Part A (one fixed cutoff, p ≥ 0.5) and Part C (eight trained rerankers calibrated after the fact on msmarco-dev)
> are done. Neither turns RQ2 into a clear win for decision models, and Part C exposes a problem with the fitting
> data that needs a decision. Triplet grid: 325 of 350
> cells, 3 still running. **Deadline: Nov 17, 2026.**

## Research questions and headline

- **RQ1:** On the same backbone, does decision fine-tuning make a better reranker than reranker fine-tuning or no
  fine-tuning?
- **RQ2:** Where reranker fine-tuning ranks better, are the calibrated probabilities of decision models still of use?

Working title: *Do System One Models Make Better Rerankers? Decision vs. Reranker Fine-Tuning on One Backbone*.

**Why this framing.** System One models are a hype topic and well marketed. Open decision fine-tunes (Kev-27B 61.1)
matched the closed Jev (60.5) within weeks. That points to a training paradigm (RLCD), not a technological
breakthrough, and the fair comparison is against the established alternative, reranker fine-tuning, on the same
backbone. Open vs. closed is now motivation, not a claim.

| Claim | Evidence | Status |
| --- | --- | --- |
| Reranker fine-tuning beats decision fine-tuning wherever both exist | better decision mode vs. reranker fine-tune, Avg: 0.6B 27.0 vs 56.1, Qwen3-8B 52.7 vs 59.4, Qwen3.5-0.8B 35.9 vs 57.1, 4B 51.4 vs 58.7, 9B 54.7 (Clef-Flash) vs 59.0 | supported, 5/5 backbones; paired tests pending; 3 cells running |
| …also without in-domain training data | zero-shot subset (COVID, Touché, DBPedia), same order: 30.9 vs 54.9, 49.1 vs 57.8, 40.7 vs 53.8, 48.4 vs 54.2, 48.8 vs 53.7 | supported; matters because the reranker fine-tunes saw MS MARCO and Kev did not |
| Decision fine-tuning adds little over the base model | better decision mode minus base, Avg: +3.2, −0.6, −1.7, −1.3, −3.6 (Kev-9B) / +1.0 (Clef-Flash), +3.7 at 27B; reranker fine-tune minus base: +5.3 to +32.3 | supported |
| At 27B the decision fine-tune is the best system | Kev-27B joint 61.1, ahead of Qwen3-Reranker-8B 59.4 and Prism-9B 59.0 | true, but there is no 27B reranker fine-tune to compare with; stated as a limitation |
| Joint scoring needs a model trained on long states | joint − isolated: Jev +1.0, Kev-27B +1.4; Kev-4B −1.5, Clef-Flash −1.9; Kev-0.6B joint AUC 0.49 | supported for Kev (trained states ≤ 7.5k tokens vs ~13k joint DL states); tests pending |
| Decision models are better calibrated out of the box | mean ECE DL19-23: Kev-27B joint 0.156, Clef joint 0.147 vs Qwen3-Reranker-8B 0.257, monoT5-3B 0.312 | supported, but Prism-4B (0.218) and bge (0.231) are close to Jev (0.224) |
| A calibrated model allows one cutoff across collections | regret of p ≥ 0.5 vs the oracle cutoff: Kev-27B joint 2.9, Clef joint 3.4, but also Prism 2.7–4.0 and bge 3.8; Qwen3-Reranker 9.0–14.9, monoT5-3B 9.9, Set-Encoder 21.1 | partly: a property of some training recipes, not of decision models alone |
| Even then the reranker fine-tune keeps the better set | best F1 at p ≥ 0.5 per backbone is the reranker fine-tune's (e.g. Prism-4B 51.0 vs Kev-4B 41.0) | supported; extends the negative result to RQ2 |
| Post-hoc calibration does not close the gap | temperature (fit on msmarco-dev) lowers ECE but not below 0.20; Platt collapses the cutoff (F1@0.5 ≈ 6) | partly: rests on the sparse msmarco-dev labels |

## Results

### RQ1: triplets (nDCG@10 × 100, BM25 top-100)

Avg = mean over all ten datasets; ZS = mean over TREC-COVID, Touché and DBPedia. Bold = best per backbone. – = cell
still running.

| Backbone | Base (zero-shot) | Decision FT: Kev joint / isolated | Reranker FT | Avg: base / best decision / reranker | ZS: base / best decision / reranker |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 23.8 | 13.2 / 27.0 | Qwen3-Reranker-0.6B | 23.8 / 27.0 / **56.1** | 25.4 / 30.9 / **54.9** |
| Qwen3-8B | 53.3 | – / 52.7 | Qwen3-Reranker-8B | 53.3 / 52.7 / **59.4** | 47.1 / 49.1 / **57.8** |
| Qwen3.5-0.8B | 37.6 | 25.3 / 35.9 | Prism-0.8B | 37.6 / 35.9 / **57.1** | 37.6 / 40.7 / **53.8** |
| Qwen3.5-4B | 52.7 | 49.8 / 51.4 | Prism-4B | 52.7 / 51.4 / **58.7** | 47.1 / 48.4 / **54.2** |
| Qwen3.5-9B | 53.7 | – / 50.1 (Clef-Flash 52.8 / 54.7) | Prism-9B | 53.7 / 54.7 / **59.0** | 48.2 / 48.8 / **53.7** |
| Qwen3.8-27B | 57.4 | 61.1 / 59.7 (Clef 56.1 / 56.2) | none exists | 57.4 / **61.1** / – | 52.8 / **57.8** / – |

**What it shows:**
- The reranker fine-tune wins on every backbone where it exists, on average and on the zero-shot subset.
- Decision fine-tuning stays within a few points of the zero-shot base model at every size; the reranker fine-tune
  adds 5–32 points.
- Robustness check: Qwen3-8B-Base with a completion prompt reaches 41.4, against 53.3 for the post-trained release.

### RQ1: joint vs. isolated (nDCG@10 points; positive = joint better)

| Model | DL19 | DL20 | DL21 | DL22 | DL23 | COVID | NFC | SciF | Touché | DBP | Avg | joint better |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Jev | +0.7 | +0.6 | +1.8 | +1.7 | +2.1 | −0.1 | −0.1 | +1.3 | −0.4 | +2.0 | +1.0 | 7/10 |
| Clef-Flash | −0.1 | −2.7 | −2.0 | −0.7 | −3.3 | +0.5 | −2.3 | −3.3 | −0.7 | −3.9 | −1.9 | 1/10 |
| Clef | +1.8 | +1.1 | +1.6 | +5.2 | +3.3 | −0.6 | −1.5 | −9.0 | −2.4 | −0.9 | −0.1 | 5/10 |
| Kev-4B | +2.9 | −0.4 | +0.2 | −2.2 | +2.9 | −7.4 | −4.8 | −8.3 | +0.7 | +1.0 | −1.5 | 5/10 |
| Kev-27B | +0.4 | +2.3 | −0.2 | +0.0 | +2.5 | +0.6 | +0.3 | +1.6 | +6.6 | −0.4 | +1.4 | 8/10 |

Kev-0.6B and Kev-0.8B lose most in joint mode (Avg 13.2 vs 27.0 and 25.3 vs 35.9). Window splits do not explain the
losses (Kev-4B never splits on COVID/SciFact). State length does: the small Kev models were trained on states of at
most 7,552 tokens and Kev-27B on up to 32,768. A joint DL state is about 13k tokens, a BEIR state 40k or more.

### RQ1: context (other systems on the same candidates)

\* = published number, not recomputed. System One models in joint mode (Laya isolated).

| System | DL19 | DL20 | DL21 | DL22 | DL23 | COVID | NFC | SciF | Touché | DBP | Avg | $/1k q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 50.6 | 48.0 | 44.6 | 26.9 | 26.2 | 59.5 | 32.2 | 67.9 | 44.2 | 31.8 | 43.2 | – |
| RankGPT-4 | 75.6\* | 70.6\* | – | – | – | 85.5\* | 38.5\* | 75.0\* | 38.6\* | 47.1\* | – | – |
| RankZephyr-7B | 74.3 | 70.0 | 67.9 | 50.5 | 44.1 | 80.7\* | 36.6\* | 75.1\* | 31.1\* | 43.2\* | 57.3 | – |
| bge-reranker-v2-m3 | 72.2\* | 67.0\* | 66.1 | 51.5 | 45.1 | 74.8\* | 33.8\* | 73.5\* | 39.9\* | 41.9\* | 56.6 | – |
| monoT5-3B | 71.8 | 68.9 | 66.7 | 49.5 | 45.0 | 80.7\* | 39.0\* | 76.6\* | 32.4\* | 44.5\* | 57.5 | – |
| Set-Encoder-large | 75.0\* | 72.2\* | 69.9 | 53.4 | 47.7 | 81.2 | 36.8 | 73.8 | 33.2 | 47.0 | 59.0 | – |
| Jev | 73.3 | 71.4 | 72.2 | 50.0 | 46.8 | 85.0 | 38.2 | 81.0 | 39.9 | 46.8 | 60.5 | 1.18 |
| Laya | 60.1 | 54.5 | 53.7 | 35.3 | 33.1 | 74.3 | 28.8 | 43.7 | 24.0 | 32.7 | 44.0 | – |

The open Kev-27B (61.1) matches the closed Jev (60.5), and both are ahead of every trained and listwise reranker on
average. That is the motivation for the controlled comparison above.

### RQ2: is calibration still of use?

**ECE** (15 bins, judged pairs of DL19-23, mean; lower is better): decision models Clef-Flash joint 0.140, Clef joint
0.147, Kev-27B joint 0.156, Jev joint 0.224, Laya 0.131; trained rerankers Prism-4B 0.218, bge 0.231, Prism-0.8B
0.243, Qwen3-Reranker-8B 0.257, monoT5-3B 0.312, Set-Encoder 0.495, Qwen3-Reranker-0.6B 0.511. (bge and Set-Encoder on
DL19/20 from our efficiency runs; their published cells have no probabilities.)

**Part A: one fixed cutoff** (keep p ≥ 0.5; means over DL19-23, × 100; regret = F1 at the per-collection oracle
cutoff − F1 at 0.5; unjudged passages count as non-relevant):

| System | F1 at 0.5 | Regret | Oracle cutoff range | Kept at 0.5 | Top-10 F1 |
| --- | ---: | ---: | --- | ---: | ---: |
| Kev-27B joint | 55.1 | 2.9 | 0.53–0.82 | 22.7 | 45.6 |
| Clef joint | 52.2 | 3.4 | 0.47–0.76 | 23.4 | 43.1 |
| Prism-4B | 51.0 | 2.9 | 0.49–0.65 | 23.1 | 44.2 |
| Prism-9B | 50.7 | 4.0 | 0.54–0.62 | 26.0 | 44.3 |
| Jev joint | 49.9 | 6.2 | 0.61–0.81 | 29.1 | 45.5 |
| bge-reranker-v2-m3 | 48.2 | 3.8 | 0.71–0.81 | 29.3 | 42.2 |
| Qwen3-Reranker-8B | 46.3 | 9.0 | 0.72–0.95 | 32.2 | 43.5 |
| monoT5-3B | 42.0 | 9.9 | 0.85–0.99 | 35.6 | 42.7 |
| Kev-4B isolated | 41.0 | 4.6 | 0.59–0.88 | 30.5 | 36.9 |
| Kev-8B isolated | 40.1 | 6.4 | 0.87–0.93 | 36.5 | 37.7 |
| Set-Encoder-large | 31.6 | 21.1 | 0.98–0.99 | 52.2 | 44.7 |

**Part C: calibrated after the fact** (temperature T and Platt P scaling of each reranker's logit, fitted on the
first 1,000 msmarco-dev queries × 100 candidates; N = native; DL means; BEIR = ECE with unjudged as non-relevant):

| Reranker | ECE DL N / T / P | F1 at 0.5 N / P | Regret N / P | ECE BEIR N / P |
| --- | --- | --- | --- | --- |
| Qwen3-Reranker-0.6B | 0.511 / 0.326 / 0.226 | 31.8 / 6.3 | 14.9 / 40.6 | 0.169 / 0.152 |
| Qwen3-Reranker-8B | 0.257 / 0.237 / 0.227 | 46.3 / 7.4 | 9.0 / 47.5 | 0.095 / 0.152 |
| Prism-0.8B | 0.243 / 0.229 / 0.237 | 46.9 / 5.7 | 2.7 / 43.7 | 0.247 / 0.147 |
| Prism-4B | 0.218 / 0.202 / 0.238 | 51.0 / 5.7 | 2.9 / 47.9 | 0.240 / 0.149 |
| monoT5-3B | 0.312 / 0.257 / 0.232 | 42.0 / 5.8 | 9.9 / 46.0 | – (BEIR from literature) |
| bge-reranker-v2-m3 | 0.231 / 0.226 / 0.236 | 48.2 / 5.7 | 3.8 / 45.9 | – (BEIR from literature) |
| Set-Encoder-large | 0.495 / 0.250* / 0.236 | 31.6 / 5.7 | 21.1 / 47.2 | 0.369 / 0.148 |
| Prism-9B | 0.242 / 0.233 / 0.238 | 50.7 / 5.7 | 4.0 / 48.4 | 0.242 / 0.150 |

\* degenerate: the temperature collapses to a ≈ 0, so every probability is 0.5.

**What it shows:**
- Part A: a usable fixed cutoff comes with some training recipes (decision fine-tuning, Prism, bge), not with
  decision models alone, and the reranker fine-tune still has the better set on every backbone.
- Part C: temperature scaling lowers ECE but leaves every reranker above the strong decision models (0.14–0.16) and
  cannot change the set at 0.5. Platt scaling learns the base rate of the fit data: only 683 of the 100,000 dev
  candidates are judged relevant (0.7%), so it pushes all probabilities down, and on densely judged DL almost nothing
  is kept at 0.5.
- That supports "decision models are calibrated without labelled fitting data, rerankers are not", but it rests on
  the fitting data. A fair test needs either a prior correction or a densely judged non-test set (open question 1).

### Cost and efficiency (DL19, one A100-SXM4-80GB, p50 per query)

| System | s / query | J / query |
| --- | ---: | ---: |
| Set-Encoder-large | 0.24 | 83 |
| monoT5-3B | 0.68 | 258 |
| Qwen3-Reranker-8B | 2.85 | 1139 |
| Kev-8B joint / isolated | 13.67 / 14.58 | 5421 / 5836 |
| Prism-9B | 4.28 | 1711 |
| Kev-9B joint | 16.72 | 5513 |
| Kev-27B joint / isolated | 14.34 / 23.13 | 3453 / 4811 |
| RankZephyr-7B | 22.30 | 4364 |

On the same backbone, Kev-8B is about 5× slower than Qwen3-Reranker-8B. This partly comes from Kev's exact-dtype
path and needs a check before it goes into the paper. Jev joint costs $1.18 per 1,000 queries (API). Still running
on the A100s: Kev-9B isolated and Qwen3.8-27B prompted.

### Sanity checks

- BM25 = Pyserini 2CR on all ten datasets ✔; monoT5-3B DL19/20 71.8 / 68.9 vs 71.83 / 68.89 ✔; RankZephyr DL19/20
  74.3 / 70.0 vs 74.2 / 70.2 ✔.
- Jev joint vs S1Rank: COVID 85.0 vs 86.5 ⚠ (others within 0.5).
- causal-conv1d kernel changes Qwen3.5/3.8 scores through near-ties; most grid cells of those models predate it ⚠.

## Decisions so far

| Date | Decision |
| --- | --- |
| 2026-10-06 | arXiv preprint first, then a WSDM 2027 short paper |
| 2026-10-07 | One setup: BM25 top-100 everywhere; published numbers stand in for runs; S1Rank's terms (joint/isolated) |
| 2026-10-08 | RankZephyr run by us (DL19-23); efficiency = API cost plus one fixed-GPU run |
| 2026-10-09 | Triplets on six backbones; base row = post-trained release zero-shot; Qwen3.7-Flash out of the paper |
| 2026-10-09 | **New narrative:** negative result as headline (RQ1: decision vs. reranker fine-tuning, joint vs. isolated folded in); RQ2 = use of calibration, a discussion backed by Parts A and C; claim scoped to 0.6B-9B; question title |

## TODOs

**Runs**
- [ ] Last triplet cells: Kev-8B joint NFCorpus/SciFact, Kev-9B joint SciFact (running).
- [x] RQ2 Part A: one fixed cutoff (`cdm cutoff`, `paper/tables/cutoff.tex`).
- [x] RQ2 Part C, first fit: eight trained rerankers on 1,000 msmarco-dev queries, `cdm posthoc`,
  `paper/tables/posthoc.tex`.
- [ ] RQ2 Part C, second fit once the fit data is decided (prior correction or another dev set; open question 1).
- [ ] Decide on rerunning Qwen3.5/3.8 cells with the causal-conv1d kernel.

**Analysis**
- [ ] Paired tests: reranker fine-tune vs. the better decision mode per backbone; joint vs. isolated (Holm).
- [ ] Joint gain against state length per model.

**Paper**
- [ ] Write it: every section is `\TODO`/`\PENDING` (structure now: introduction, related work, setup, decision vs.
  reranker fine-tuning, is calibration still of use?, conclusion). Move to ACM sigconf.
- [ ] `\FLAG`s: citation for RLCD; release dates of Jev, Clef, Kev ("within weeks").

## Open questions for the meeting

1. **Part C fit data.** msmarco-dev has 0.7% relevant candidates (sparse judgments), so Platt scaling pushes every
   probability down and the fixed cutoff collapses on DL. Do we (a) report this as the finding ("rerankers need
   densely judged labels to be calibrated, decision models don't"), (b) add a prior-corrected Platt fit, or (c) find
   a densely judged non-test collection to fit on?
2. **RQ2's message.** Part A says a fixed cutoff works for Prism and bge too, and the reranker fine-tune keeps the
   better set. Is RQ2 then "calibration is a property of training recipes, not of decision models", and is set F1
   at p ≥ 0.5 the right utility (vs. precision at the cutoff or a cost per passage)?
3. **RLCD source and the "within weeks" claim.** What do we cite, and what are the release dates?
4. **Kernel change.** Do we rerun the Qwen3.5/3.8 cells (Kev, Clef, Prism, base models) or state it?
5. **Training-data confound.** The reranker fine-tunes saw MS MARCO. Is the zero-shot subset enough to answer this,
   or do we need more?
6. **Writing split and timeline** up to the arXiv post and Nov 17.

## Risks and limitations

- **No 27B reranker fine-tune:** at the one size where decision fine-tuning leads, the comparison is missing.
- **Different training data** between the fine-tunes (MS MARCO vs. none); the zero-shot subset has only three
  collections.
- **RQ2 is weaker than hoped:** some reranker fine-tunes (Prism, bge) allow a fixed cutoff as well, and Part C's
  answer to "just calibrate the reranker" depends on the fit data.
- **Mixed code versions** for the Qwen3.5/3.8 cells (kernel).
- Literature numbers (RankGPT-4, RankZephyr BEIR, part of bge/monoT5/Set-Encoder) have no per-query runs; benchmark
  contamination; Jev is closed and vendor-versioned.
- Calibration is measured on DL19-23 only, and energy is GPU-board only.

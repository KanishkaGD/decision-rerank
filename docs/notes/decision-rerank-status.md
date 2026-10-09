# Do System One Models Make Better Rerankers? Status 2026-10-09 (new narrative)

> On one backbone at a time, we compare three training objectives: none (the base model asked S1Rank's question
> zero-shot), decision fine-tuning (System One models: Kev, Clef) and reranker fine-tuning (Qwen3-Reranker,
> Prism). **Headline: a negative result.** Reranker fine-tuning beats decision fine-tuning on every backbone where
> both exist (0.6B–9B), by 4–29 nDCG@10 points on average and also on the collections without a training split.
> Decision fine-tuning moves the base model by only −3.6 to +3.7 points. **RQ2** asks whether the one thing decision
> models have and reranker fine-tunes lack, calibrated probabilities, is still of use; we argue it in a discussion,
> backed by a cutoff analysis and post-hoc calibration of the reranker fine-tunes. **Status:** narrative switched on
> 2026-10-09; the triplet grid has 325 of 350 cells; the RQ2 analyses are not run yet. WSDM 2027 short paper, arXiv
> first. **Deadline: Nov 17, 2026.**

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
| Decision models are better calibrated out of the box | mean ECE DL19-23: Kev-27B joint 0.156, Clef joint 0.147 vs Qwen3-Reranker-8B 0.257, monoT5-3B 0.312 | supported, but **not yet shown to be useful**: RQ2's cutoff and post-hoc-scaling analyses are open |

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

ECE, 15 bins, judged pairs of DL19-23 (lower is better):

| System | DL19 | DL20 | DL21 | DL22 | DL23 | mean |
|---|---:|---:|---:|---:|---:|---:|
| monoT5-3B | 0.185 | 0.248 | 0.359 | 0.347 | 0.421 | 0.312 |
| Qwen3-Reranker-8B | 0.123 | 0.188 | 0.309 | 0.298 | 0.365 | 0.257 |
| Set-Encoder-large | – | – | 0.474 | 0.547 | 0.580 | 0.534 |
| Jev joint / isolated | 0.157 / 0.192 | 0.227 / 0.266 | 0.244 / 0.290 | 0.225 / 0.253 | 0.265 / 0.301 | 0.224 / 0.260 |
| Clef joint / isolated | 0.095 / 0.174 | 0.155 / 0.238 | 0.162 / 0.271 | 0.151 / 0.232 | 0.172 / 0.281 | 0.147 / 0.240 |
| Kev-4B joint / isolated | 0.174 / 0.137 | 0.259 / 0.214 | 0.243 / 0.234 | 0.297 / 0.244 | 0.283 / 0.249 | 0.251 / 0.216 |
| Kev-27B joint / isolated | 0.098 / 0.203 | 0.145 / 0.257 | 0.186 / 0.309 | 0.163 / 0.252 | 0.189 / 0.290 | 0.156 / 0.262 |
| Laya | 0.066 | 0.102 | 0.134 | 0.173 | 0.178 | 0.131 |

**Planned evidence** (`docs/notes/calibration-proposal.md`; none run yet):
- **One cutoff (Part A):** keep every passage with p ≥ 0.5. Report set F1 and the regret against the per-dataset
  oracle cutoff for the decision fine-tunes and for the reranker fine-tunes' native probabilities, with fixed top-k
  as the baseline. Uses stored outputs only.
- **Calibrated after the fact (Part C):** temperature and Platt scaling of the reranker fine-tunes, fitted on a
  msmarco-dev subset. This needs new runs on msmarco-dev. It answers the reviewer's "just calibrate the reranker".
- **Discussion:** cutoffs for RAG, QPP (Kanishka's results), "no relevant passage" detection. Can one model rank like
  a reranker fine-tune and stay calibrated like a decision model? That is roadmap Phase 2.

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
- [ ] RQ2 Part C: reranker fine-tunes (Qwen3-Reranker-0.6B/8B, Prism-0.8B/4B/9B, maybe monoT5-3B) on a msmarco-dev
  subset. Needs `cdm first-stage msmarco-dev` and an analysis module for the scaling.
- [ ] Decide on rerunning Qwen3.5/3.8 cells with the causal-conv1d kernel.

**Analysis**
- [ ] Paired tests: reranker fine-tune vs. the better decision mode per backbone; joint vs. isolated (Holm).
- [ ] RQ2 Part A from stored outputs: cutoff F1, regret, oracle cutoff per dataset.
- [ ] Joint gain against state length per model.
- [ ] Commit `results/results.csv` and regenerate `paper/tables/triplets.tex` and `context.tex`.

**Paper**
- [ ] Write it: every section is `\TODO`/`\PENDING` (structure now: introduction, related work, setup, decision vs.
  reranker fine-tuning, is calibration still of use?, conclusion). Move to ACM sigconf.
- [ ] `\FLAG`s: citation for RLCD; release dates of Jev, Clef, Kev ("within weeks").

## Open questions for the meeting

1. **RLCD source and the "within weeks" claim.** What do we cite, and what are the release dates?
2. **RQ2 utility.** Is set F1 at p ≥ 0.5 the right measure, or precision at the cutoff, or a cost per passage? Which
   msmarco-dev subset do we fit the scaling on?
3. **Kernel change.** Do we rerun the Qwen3.5/3.8 cells (Kev, Clef, Prism, base models) or state it?
4. **Training-data confound.** The reranker fine-tunes saw MS MARCO. Is the zero-shot subset enough to answer this,
   or do we need more?
5. **Writing split and timeline** up to the arXiv post and Nov 17.

## Risks and limitations

- **No 27B reranker fine-tune:** at the one size where decision fine-tuning leads, the comparison is missing.
- **Different training data** between the fine-tunes (MS MARCO vs. none); the zero-shot subset has only three
  collections.
- **RQ2 is unproven so far:** without Part C, "just calibrate the reranker" is unanswered.
- **Mixed code versions** for the Qwen3.5/3.8 cells (kernel).
- Literature numbers (RankGPT-4, RankZephyr BEIR, part of bge/monoT5/Set-Encoder) have no per-query runs; benchmark
  contamination; Jev is closed and vendor-versioned.
- Calibration is measured on DL19-23 only, and energy is GPU-board only.

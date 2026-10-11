# Do System One Models Make Better Rerankers? Status 2026-10-11

> On one backbone at a time, we compare three training objectives: none (the base model asked S1Rank's question
> zero-shot), decision fine-tuning (System One models: Kev, Clef) and reranker fine-tuning (Qwen3-Reranker, Prism).
> **Headline: a negative result.** Reranker fine-tuning beats decision fine-tuning on every backbone where both
> exist (0.6B–9B), on average and on the collections without a training split. **RQ2** asks whether calibrated
> probabilities, the one thing decision models offer, are still of use.
>
> **New since 2026-10-09:**
> - The triplet grid is complete (350 cells).
> - Every Qwen3.5/3.8 cell was rerun on the container with the causal-conv1d kernel. No result moved by more than
>   0.6 points.
> - The tables and the RQ2 analyses were recomputed.
> - **Our own 27B reranker fine-tune is now planned**, to fill the one missing cell. The 9B validation run has
>   finished training (8.3 h on 2 H200s), and checkpoint selection on msmarco-dev is running.
>
> **Deadline: Nov 17, 2026.**

## Research questions and headline

- **RQ1:** On the same backbone, does decision fine-tuning make a better reranker than reranker fine-tuning or no
  fine-tuning?
- **RQ2:** Where reranker fine-tuning ranks better, are the calibrated probabilities of decision models still of use?

Working title: *Do System One Models Make Better Rerankers? Decision vs. Reranker Fine-Tuning on One Backbone*.

**Why this framing.** System One models are a hype topic and well marketed. Open decision fine-tunes (Kev-27B 61.1)
matched the closed Jev (60.5) within weeks. That points to a training paradigm (RLCD), not a technological
breakthrough. The fair comparison is therefore against the established alternative, reranker fine-tuning, on the
same backbone. Open vs. closed is now motivation, not a claim.

| Claim | Evidence (nDCG@10 × 100) | Status |
| --- | --- | --- |
| Reranker fine-tuning beats decision fine-tuning wherever both exist | Avg, better decision mode vs. reranker fine-tune: 0.6B 27.0 vs 56.1; Qwen3-8B 52.7 vs 59.4; Qwen3.5-0.8B 36.0 vs 57.1; 4B 51.3 vs 58.7; 9B 54.7 (Clef-Flash) vs 59.0 | supported on 5/5 backbones; paired tests pending |
| …also on collections without training data | zero-shot subset (COVID, Touché, DBPedia), same order: 30.9 vs 54.9, 49.1 vs 57.8, 40.7 vs 53.8, 48.4 vs 54.2, 49.0 vs 53.7 | supported; matters because the reranker fine-tunes saw MS MARCO and Kev did not |
| Decision fine-tuning adds little over the base model | Avg, better decision mode minus base: +3.2, −0.6, −1.6, −1.4, −3.5 (Kev-9B) / +1.0 (Clef-Flash), +3.7 (Kev-27B) / −1.0 (Clef). Reranker fine-tune minus base: +5.3 to +32.3 | supported |
| At 27B the decision fine-tune is the best system | Kev-27B joint 61.1, ahead of Qwen3-Reranker-8B 59.4 and Prism-9B 59.0 | true, but untested against a reranker fine-tune of the same size: **our 27B reranker is in preparation** |
| Joint scoring needs a model trained on long states | joint − isolated, Avg: Kev-27B +1.3, Jev +1.0, Kev-9B +0.1, Kev-4B −1.5, Clef-Flash −2.0, Kev-0.8B −10.7, Kev-0.6B −13.7, **Kev-8B −13.9** | supported for the small Kev models (trained states ≤ 7.5k tokens vs ~13k-token joint DL states); Kev-8B's collapse still needs its training length; tests pending |
| Strong decision models are well calibrated out of the box | mean ECE DL19-23: Clef-Flash joint 0.140, Clef joint 0.147, Kev-27B joint 0.156 vs Qwen3-Reranker-8B 0.257, monoT5-3B 0.312 | partly: small Kev models are badly calibrated (Kev-0.6B 0.58, Kev-0.8B 0.36), and the base Qwen3.8-27B is close (0.166) |
| A calibrated model allows one cutoff for every collection | regret at p ≥ 0.5: Kev-27B joint 2.8, Clef joint 3.1, but also Prism 2.8–4.0, bge 3.8, Qwen3.8-27B base 2.4; Qwen3-Reranker 9.0–14.9, monoT5-3B 9.9, Set-Encoder 21.1 | partly: a property of some training recipes, not of decision models alone |
| Even then the reranker fine-tune keeps the better set | best F1 at p ≥ 0.5 per backbone (0.6B–9B) is the reranker fine-tune's, e.g. Prism-4B 51.0 vs Kev-4B 41.0, Prism-9B 50.7 vs Clef-Flash 49.2 | supported |
| Calibrating the rerankers after the fact does not close the gap | temperature (fitted on msmarco-dev) lowers ECE but not below 0.20; Platt scaling collapses the cutoff (F1 at 0.5 ≈ 6) | partly: rests on the sparse msmarco-dev labels (open question 2) |

## Results

### RQ1: triplets (nDCG@10 × 100, BM25 top-100)

Avg = mean over all ten datasets; ZS = mean over TREC-COVID, Touché and DBPedia. Bold = best per backbone.

| Backbone | Base (zero-shot) | Decision fine-tune, joint / isolated | Reranker fine-tune | Avg: base / best decision / reranker | ZS: base / best decision / reranker |
| --- | ---: | --- | --- | ---: | ---: |
| Qwen3-0.6B | 23.8 | Kev 13.2 / 27.0 | Qwen3-Reranker-0.6B | 23.8 / 27.0 / **56.1** | 25.4 / 30.9 / **54.9** |
| Qwen3-8B | 53.3 | Kev 38.8 / 52.7 | Qwen3-Reranker-8B | 53.3 / 52.7 / **59.4** | 47.1 / 49.1 / **57.8** |
| Qwen3.5-0.8B | 37.6 | Kev 25.3 / 36.0 | Prism-0.8B | 37.6 / 36.0 / **57.1** | 37.7 / 40.7 / **53.8** |
| Qwen3.5-4B | 52.7 | Kev 49.8 / 51.3 | Prism-4B | 52.7 / 51.3 / **58.7** | 47.1 / 48.4 / **54.2** |
| Qwen3.5-9B | 53.7 | Kev 50.2 / 50.1; Clef-Flash 52.8 / 54.7 | Prism-9B (ours: validation running) | 53.7 / 54.7 / **59.0** | 48.2 / 49.0 / **53.7** |
| Qwen3.8-27B | 57.4 | Kev 61.1 / 59.8; Clef 56.1 / 56.4 | ours: planned | 57.4 / **61.1** / – | 52.7 / **57.9** / – |

**What it shows:**
- The reranker fine-tune wins on every backbone where it exists.
- Decision fine-tuning stays within about ±4 points of the zero-shot base model.
- Robustness check: Qwen3-8B-Base with a completion prompt scores 41.4, against 53.3 for the post-trained release.

### RQ1: joint vs. isolated (joint minus isolated, nDCG@10 points)

| Model | DL19 | DL20 | DL21 | DL22 | DL23 | COVID | NFC | SciF | Touché | DBP | Avg | joint better |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Jev | +0.7 | +0.6 | +1.8 | +1.7 | +2.1 | −0.1 | −0.1 | +1.3 | −0.4 | +2.0 | +1.0 | 7/10 |
| Clef-Flash | −0.0 | −2.6 | −2.0 | −1.3 | −3.4 | −0.2 | −2.2 | −3.3 | −0.8 | −3.8 | −2.0 | 0/10 |
| Clef | +2.0 | +0.8 | +1.9 | +5.2 | +3.0 | −0.6 | −1.5 | −8.9 | −3.1 | −1.3 | −0.3 | 5/10 |
| Kev-0.6B | −15.0 | −15.4 | −20.7 | −10.1 | −7.8 | −24.2 | −5.0 | −15.0 | −8.5 | −15.4 | −13.7 | 0/10 |
| Kev-0.8B | −7.1 | −7.6 | −9.7 | −5.9 | −5.6 | −26.6 | −2.3 | −15.8 | −8.6 | −17.5 | −10.7 | 0/10 |
| Kev-4B | +2.9 | −0.5 | +0.4 | −2.1 | +3.0 | −7.3 | −4.8 | −8.3 | +0.7 | +1.0 | −1.5 | 5/10 |
| Kev-8B | −7.8 | −4.4 | −1.2 | −1.1 | −1.9 | −32.6 | −15.4 | −54.7 | −16.8 | −3.3 | −13.9 | 0/10 |
| Kev-9B | −2.7 | +1.9 | +2.8 | +0.3 | +3.8 | +0.5 | −1.3 | −11.6 | +2.8 | +4.4 | +0.1 | 7/10 |
| Kev-27B | +0.5 | +2.3 | −0.1 | −0.2 | +2.5 | +0.5 | +0.4 | +1.2 | +6.7 | −0.5 | +1.3 | 7/10 |

**Reading:**
- The losses are largest where joint states are longest: COVID, NFCorpus, SciFact and Touché, with passages of about
  400–860 tokens, i.e. 40k+ tokens per joint request.
- The small Kev models were trained on states of at most 7,552 tokens; Kev-27B on up to 32,768.
- On DL19, Kev-0.6B joint does not discriminate at all: the AUC of relevant vs. judged non-relevant passages is 0.49,
  and its scores rise with a passage's position in the state.
- Window splits don't explain the losses: the Kev models never split, except on Touché.

### RQ1: context (other systems on the same candidates)

\* = published number. System One models are in joint mode (Laya isolated).

| System | DL19 | DL20 | DL21 | DL22 | DL23 | COVID | NFC | SciF | Touché | DBP | Avg | $/1k q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 50.6 | 48.0 | 44.6 | 26.9 | 26.2 | 59.5 | 32.2 | 67.9 | 44.2 | 31.8 | 43.2 | – |
| RankGPT-4 | 75.6\* | 70.6\* | – | – | – | 85.5\* | 38.5\* | 75.0\* | 38.6\* | 47.1\* | – | – |
| RankZephyr-7B | 74.3 | 70.0 | 67.9 | 50.5 | 44.1 | 80.7\* | 36.6\* | 75.1\* | 31.1\* | 43.2\* | 57.3 | – |
| bge-reranker-v2-m3 | 72.2\* | 67.0\* | 66.1 | 51.5 | 45.1 | 74.8\* | 33.8\* | 73.5\* | 39.9\* | 41.9\* | 56.6 | – |
| monoT5-3B | 71.8 | 68.9 | 66.7 | 49.5 | 45.0 | 80.7\* | 39.0\* | 76.6\* | 32.4\* | 44.5\* | 57.5 | – |
| Set-Encoder-large | 75.0\* | 72.2\* | 69.9 | 53.4 | 47.7 | 81.2 | 36.8 | 73.8 | 33.2 | 47.0 | 59.0 | – |
| Jev | 73.3 | 71.4 | 72.2 | 50.0 | 46.8 | 85.0 | 38.2 | 81.0 | 39.9 | 46.8 | 60.5 | 1.18 |
| Clef | 71.4 | 70.1 | 68.6 | 49.3 | 45.9 | 82.9 | 36.3 | 62.9 | 30.8 | 42.8 | 56.1 | – |
| Kev-27B | 73.9 | 73.3 | 69.9 | 51.1 | 49.3 | 85.9 | 38.7 | 81.1 | 40.8 | 46.9 | 61.1 | – |
| Laya | 60.1 | 54.5 | 53.7 | 35.3 | 33.1 | 74.3 | 28.8 | 43.7 | 24.0 | 32.7 | 44.0 | – |

The open Kev-27B (61.1) matches the closed Jev (60.5). Both lead every trained and listwise reranker on average,
which is the motivation for the controlled comparison.

### RQ2: is calibration still of use?

**ECE** (15 bins, judged pairs of DL19–23, mean; lower is better):

| Backbone | Base | Decision fine-tune (better mode) | Reranker fine-tune |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 0.750 | Kev 0.579 (isolated) | 0.511 |
| Qwen3-8B | 0.383 | Kev 0.243 (joint) | 0.257 |
| Qwen3.5-0.8B | 0.716 | Kev 0.357 (isolated) | 0.243 |
| Qwen3.5-4B | 0.296 | Kev 0.216 (isolated) | 0.218 |
| Qwen3.5-9B | 0.283 | Clef-Flash 0.140 (joint) | 0.242 |
| Qwen3.8-27B | 0.166 | Clef 0.147 (joint) / Kev 0.156 (joint) | – |

bge and Set-Encoder have probabilities only from our own runs (DL21–23 and the DL19/20 efficiency runs); their
published cells have none. Decision fine-tuning improves calibration over the base model at every size. Against the reranker fine-tune it is
better only from 9B upward; below that the two are about equal, or the reranker is better.

**Part A, one fixed cutoff** (keep p ≥ 0.5; means over DL19–23, × 100; regret = F1 at the per-collection oracle
cutoff − F1 at 0.5):

| System | F1 at 0.5 | Regret | Oracle cutoffs | Kept at 0.5 |
| --- | ---: | ---: | --- | ---: |
| Kev-27B joint | 55.1 | 2.8 | 0.59–0.77 | 22.8 |
| Clef joint | 52.5 | 3.1 | 0.46–0.74 | 23.3 |
| Qwen3.8-27B base | 52.2 | 2.4 | 0.56–0.89 | 24.2 |
| Prism-4B | 51.0 | 2.8 | 0.49–0.65 | 23.1 |
| Prism-9B | 50.7 | 4.0 | 0.54–0.62 | 26.0 |
| Jev joint | 49.9 | 6.2 | 0.61–0.81 | 29.1 |
| Clef-Flash isolated | 49.2 | 3.3 | 0.56–0.81 | 25.9 |
| bge-reranker-v2-m3 | 48.2 | 3.8 | 0.71–0.81 | 29.3 |
| Qwen3-Reranker-8B | 46.3 | 9.0 | 0.72–0.95 | 32.2 |
| monoT5-3B | 42.0 | 9.9 | 0.85–0.99 | 35.6 |
| Kev-4B isolated | 41.0 | 4.6 | 0.59–0.88 | 30.5 |
| Set-Encoder-large | 31.6 | 21.1 | 0.98–0.99 | 52.2 |

**Part C, calibrated after the fact.** Each reranker's logit is rescaled with temperature (T) or Platt (P) scaling,
fitted on the first 1,000 msmarco-dev queries; N = native; DL means.

| Reranker | ECE DL: N / T / P | F1 at 0.5: N / P | Regret: N / P | ECE BEIR: N / P |
| --- | --- | --- | --- | --- |
| Qwen3-Reranker-0.6B | 0.511 / 0.326 / 0.226 | 31.8 / 6.3 | 14.9 / 40.6 | 0.169 / 0.152 |
| Qwen3-Reranker-8B | 0.257 / 0.237 / 0.227 | 46.3 / 7.4 | 9.0 / 47.5 | 0.095 / 0.152 |
| Prism-0.8B | 0.243 / 0.229 / 0.237 | 46.9 / 5.7 | 2.8 / 43.7 | 0.247 / 0.147 |
| Prism-4B | 0.218 / 0.202 / 0.238 | 51.0 / 5.7 | 2.8 / 47.9 | 0.241 / 0.149 |
| Prism-9B | 0.242 / 0.233 / 0.238 | 50.7 / 5.7 | 4.0 / 48.3 | 0.242 / 0.150 |
| monoT5-3B | 0.312 / 0.257 / 0.232 | 42.0 / 5.8 | 9.9 / 46.0 | – (BEIR from the literature) |
| bge-reranker-v2-m3 | 0.231 / 0.226 / 0.236 | 48.2 / 5.7 | 3.8 / 45.9 | – (BEIR from the literature) |
| Set-Encoder-large | 0.495 / 0.250\* / 0.236 | 31.6 / 5.7 | 21.1 / 47.2 | 0.369 / 0.148 |

\* Degenerate: the temperature collapses to a ≈ 0, so every probability is 0.5.

**What it shows:**
- Temperature scaling lowers ECE, but every reranker stays above the strong decision models (0.14–0.16).
- Platt scaling learns the fit data's base rate: only 683 of the 100,000 msmarco-dev candidates (0.7%) are judged
  relevant. It pushes all probabilities down, so on densely judged DL almost nothing is kept at 0.5.
- That supports "decision models are calibrated without labelled fitting data, rerankers are not", but the evidence
  rests on the fitting data. A fair test needs a prior correction or a densely judged non-test set (open question 2).

### Cost and efficiency (DL19, one A100-SXM4-80GB, median per query, causal-conv1d kernel)

| Backbone | Base (s / J) | Decision fine-tune (s / J) | Reranker fine-tune (s / J) |
| --- | --- | --- | --- |
| Qwen3-0.6B | 0.33 / 122 | Kev isolated 2.84 / 534 | 0.41 / 159 |
| Qwen3-8B | 2.26 / 911 | Kev isolated 14.58 / 5,836 | 2.85 / 1,139 |
| Qwen3.5-4B | 1.37 / 553 | Kev isolated 12.22 / 3,881 | 2.62 / 1,049 |
| Qwen3.5-9B | 2.30 / 924 | Kev isolated 19.24 / 6,950; Clef-Flash 2.11 / 800 (joint) | 4.28 / 1,711 |
| Qwen3.8-27B | 4.25 / 1,712 | Kev joint 14.34 / 3,453; Clef joint 6.43 / 2,562 | – |

- Context: Set-Encoder 0.24 s / 83 J, monoT5-3B 0.68 s / 258 J, RankZephyr 22.3 s / 4,364 J. Jev joint costs $1.18
  per 1,000 queries. RankGPT-4 costs $0.641 per query (literature).
- Kev is 4.5–7× slower than the reranker fine-tune of the same backbone. This also holds on Qwen3, where the kernel
  plays no role, so it comes from Kev's own exact-dtype inference path. Clef, on the same backbones, is fast. Check
  before the efficiency table goes into the paper.

### Our reranker fine-tune (the missing 27B cell)

- **Recipe** (`docs/notes/reranker-27b-plan.md`): LoRA r 32 on the checkpoint Kev-27B and Clef start from;
  Qwen3-Reranker prompt; ranking loss plus yes/no loss; 100k MS MARCO queries with dense and BM25 negatives
  (197 queries overlapping test or dev topics removed); checkpoints chosen on msmarco-dev only.
- **9B validation (Qwen3.5-9B vs Prism-9B)**: trained, 1,562 steps in 8.3 h on 2 H200s (job 4411777). Checkpoint
  selection on 200 msmarco-dev queries is running (4427678–4427685), followed by all ten datasets.
- **Criterion, fixed in advance:** Avg ≥ 57.0 (within 2 points of Prism-9B) and above Clef-Flash isolated (54.7).
- **27B run, only if the criterion is met:** 4 × 80 GB, about 30 h. Probe: 61 GiB per GPU, ~4 s per training group.

### Sanity checks

- BM25 = Pyserini 2CR on all ten datasets ✔
- monoT5-3B on DL19/20: 71.8 / 68.9 vs 71.83 / 68.89 ✔
- RankZephyr on DL19/20: 74.3 / 70.0 vs 74.2 / 70.2 ✔
- Jev joint vs S1Rank: COVID 85.0 vs 86.5 ⚠ (the others are within 0.5)
- Kernel rerun: every Qwen3.5/3.8 cell now comes from one implementation ✔ (changes ≤ 0.6 points)

## Decisions so far

| Date | Decision |
| --- | --- |
| 2026-10-06 | arXiv preprint first, then a WSDM 2027 short paper |
| 2026-10-07 | One setup: BM25 top-100 everywhere; published numbers stand in for runs; S1Rank's terms (joint/isolated) |
| 2026-10-08 | RankZephyr run by us (DL19–23); efficiency = API cost plus one fixed-GPU run |
| 2026-10-09 | Triplets on six backbones; the base row is the post-trained release, zero-shot; Qwen3.7-Flash dropped |
| 2026-10-09 | Narrative: a negative result as the headline (RQ1); RQ2 = the use of calibration, argued with Parts A and C; a question as the title |
| 2026-10-10 | Every Qwen3.5/3.8 cell rerun on the causal-conv1d image |
| 2026-10-10 | **Our own 27B reranker fine-tune**, after a 9B validation with a criterion fixed in advance |

## TODOs

**Runs**
- [ ] 9B validation: checkpoint selection (running), then all ten datasets, then the criterion.
- [ ] If it passes: the 27B fine-tune (~30 h on 4 × 80 GB), its checkpoint selection and scoring.
- [ ] RQ2 Part C, second fit once the fit data is decided (open question 2).
- [x] Triplet grid complete; kernel rerun; tables and RQ2 rows regenerated.

**Analysis**
- [ ] Paired tests: reranker fine-tune vs. the better decision mode per backbone; joint vs. isolated (Holm).
- [ ] Joint gain against state length per model (Kev-8B's training length is still missing).
- [ ] Kev's inference speed (4.5–7× slower than the reranker fine-tune of the same backbone).

**Paper and repository**
- [ ] Write it: every section is still `\TODO` or `\PENDING`. Move to ACM sigconf.
- [ ] `\FLAG`s: a citation for RLCD; the release dates of Jev, Clef and Kev.
- [ ] Merge `dev-michael` into `master` (70 commits ahead).

## Open questions for the meeting

1. **If our 27B reranker beats Kev-27B,** the negative result holds at every size. If it doesn't, 27B becomes the
   one size where decision fine-tuning leads. Which framing do we prepare for each case?
2. **Part C fit data.** msmarco-dev has only 0.7% relevant candidates, so Platt scaling pushes every probability
   down. Options: report this as the finding, add a prior-corrected fit, or fit on a densely judged non-test
   collection.
3. **RQ2's message.** A fixed cutoff works for Prism, bge and the base Qwen3.8-27B too, and below 9B the reranker
   fine-tune is as well calibrated as Kev. Is RQ2 then "calibration comes with some training recipes and with
   scale, not with decision fine-tuning as such"?
4. **Training-data confound.** Our reranker fine-tune also trains on MS MARCO. Is the three-collection zero-shot
   subset enough?
5. **RLCD source and the "within weeks" claim.** What do we cite, and what are the release dates?
6. **Writing split and timeline** up to the arXiv post and Nov 17.

## Risks and limitations

- **The 27B comparison is still missing.** It depends on the validation run passing and on GPU availability
  (4 × 80 GB for about 30 h).
- **Our recipe is not Qwen's or Prism's:** it trains on MS MARCO only, and the validation against Prism-9B is what
  makes the 27B cell interpretable.
- **RQ2 is weaker than hoped:** reranker fine-tunes (Prism, bge) and the 27B base allow a fixed cutoff too, and Part
  C's answer to "just calibrate the reranker" depends on the fit data.
- Literature numbers (RankGPT-4, RankZephyr BEIR, part of bge, monoT5 and Set-Encoder) have no per-query runs.
- Benchmark contamination is possible; Jev is closed and versioned by its vendor.
- Calibration is measured on DL19–23 only, and energy is GPU-board only.

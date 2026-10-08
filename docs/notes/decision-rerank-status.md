# Calibrated System One Models as Rerankers: status 2026-10-08

> We test whether System One decision models, the closed Jev and the open Clef, Kev and Laya, rerank BM25 top-100
> candidates as well as strong trained rerankers, whether scoring all candidates in one request (joint) beats
> scoring each alone (isolated), and whether their probabilities are calibrated. **Headline: open matches closed.**
> The open Kev-27B matches the closed Jev, and both beat every trained reranker on average. **Status:** the grid is
> complete: 146 cells computed, 14 taken from the literature, on 10 datasets. RankZephyr (run by us) and a
> fixed-GPU latency/energy run are in the queue (33 jobs). The target is a WSDM 2027 short paper (4 pages, ACM
> sigconf), with an arXiv preprint first. **Deadline: Nov 17, 2026.**

## Research questions and headline

- **RQ1:** Do open System One models rerank BM25 candidates as well as the closed Jev and strong trained rerankers?
- **RQ2:** Does joint pointwise scoring, with all candidates in one state, beat isolated pointwise scoring of each
  query–passage pair, and does that hold beyond Jev?
- **RQ3:** Are the relevance probabilities of System One models calibrated, and does joint scoring change that?

Working title: *Calibrated System One Models as Rerankers* (kept open on purpose).

| Claim | Evidence | Status |
| --- | --- | --- |
| An open model matches the closed one | Kev-27B joint averages 61.1, Jev joint 60.5 over 10 datasets | supported; no significance test yet |
| Strong System One models beat trained rerankers | best trained rerankers: Qwen3-Reranker-8B 59.4, Set-Encoder 59.0, monoT5-3B 57.5 | supported on average; Touché and NFCorpus are exceptions |
| They are on par with listwise LLM rerankers | Kev-27B vs RankGPT-4 (literature): DL20 73.2 vs 70.6, COVID 86.0 vs 85.5, SciFact 81.3 vs 75.0, DL19 73.9 vs 75.6 | partly: literature numbers, the 2023 GPT-4, and no DL21-23 cells yet |
| Size matters more than open vs. closed | small open models fall behind: Clef-Flash 52.8, Kev-4B 49.8, Laya 44.0 | supported |
| Joint scoring helps, but only strong models | Jev +1.0 (joint better on 7/10 datasets), Kev-27B +1.4 (8/10); Clef-Flash −1.9 (1/10), Kev-4B −1.5 | supported descriptively; tests pending |
| Joint scoring improves calibration | mean ECE on DL19-23, joint vs isolated: Kev-27B 0.156 vs 0.262, Clef 0.147 vs 0.240, Jev 0.224 vs 0.260 | supported, except Kev-4B |
| System One probabilities are better calibrated than trained rerankers' | mean ECE: Kev-27B joint 0.156, Clef joint 0.147 vs monoT5-3B 0.312, Qwen3-8B 0.257 | supported on DL19-23 |
| Orders of magnitude cheaper than RankGPT-4 | Jev joint $1.18 per 1k queries (measured) vs RankGPT-4 $641 (Jin et al. 2025) | supported; the RankGPT-4 cost is from the literature |

## Results

### RQ1: effectiveness (nDCG@10 × 100, BM25 top-100)

System One models are shown in joint mode (Laya isolated). \* = published number, not recomputed. Bold = best
reranker per column. $/1k q = mean API cost per 1,000 queries.

| System | DL19 | DL20 | DL21 | DL22 | DL23 | COVID | NFC | SciF | Touché | DBP | Avg | $/1k q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 50.6 | 48.0 | 44.6 | 26.9 | 26.2 | 59.5 | 32.2 | 67.9 | 44.2 | 31.8 | 43.2 | – |
| RankGPT-4 | **75.6\*** | 70.6\* | – | – | – | 85.5\* | 38.5\* | 75.0\* | 38.6\* | **47.1\*** | – | – |
| RankZephyr-7B | running | running | running | running | running | 80.7\* | 36.6\* | 75.1\* | 31.1\* | 43.2\* | – | – |
| bge-reranker-v2-m3 | 72.2\* | 67.0\* | 66.1 | 51.5 | 45.1 | 74.8\* | 33.8\* | 73.5\* | 39.9\* | 41.9\* | 56.6 | – |
| monoT5-3B | 71.8 | 68.9 | 66.7 | 49.5 | 45.0 | 80.7\* | **39.0\*** | 76.6\* | 32.4\* | 44.5\* | 57.5 | – |
| Qwen3-Reranker-8B | 73.2 | 66.8 | 67.0 | 51.2 | 45.2 | 84.0 | 37.3 | 79.9 | **43.1** | 46.2 | 59.4 | – |
| Set-Encoder-large | 75.0\* | 72.2\* | 69.9 | **53.4** | 47.7 | 81.2 | 36.8 | 73.8 | 33.2 | 47.0 | 59.0 | – |
| Qwen3.7-Flash | 68.8 | 66.8 | 65.4 | 44.8 | 42.4 | 76.8 | 34.4 | 68.5 | 26.6 | 35.9 | 53.0 | 0.82 |
| Jev | 73.3 | 71.4 | **72.2** | 50.0 | 46.8 | 85.0 | 38.2 | 81.0 | 39.9 | 46.8 | 60.5 | 1.18 |
| Clef-Flash | 68.9 | 67.4 | 64.3 | 44.3 | 40.7 | 82.4 | 35.0 | 65.2 | 21.2 | 38.7 | 52.8 | – |
| Clef | 71.5 | 69.8 | 68.3 | 49.4 | 46.0 | 82.8 | 36.2 | 62.6 | 31.2 | 43.0 | 56.1 | – |
| Kev-4B | 65.5 | 62.9 | 62.4 | 37.7 | 38.3 | 75.3 | 29.3 | 62.4 | 23.7 | 40.7 | 49.8 | – |
| **Kev-27B** | 73.9 | **73.2** | 70.0 | 51.3 | **49.2** | **86.0** | 38.6 | **81.3** | 40.4 | 47.0 | **61.1** | – |
| Laya | 60.1 | 54.5 | 53.7 | 35.3 | 33.1 | 74.3 | 28.8 | 43.7 | 24.0 | 32.7 | 44.0 | – |

**What it shows:**
- Kev-27B and Jev are the two best systems on average.
- Every reranker falls below BM25 on Touché (44.2), a known effect.
- Qwen3.7-Flash, a plain chat LLM at Jev's price, trails Jev by 7.5 points on average. That makes it a good control
  for "the gain comes from the decision model, not from any LLM".

### RQ2: joint minus isolated (nDCG@10 points; positive = joint better)

| Model | DL19 | DL20 | DL21 | DL22 | DL23 | COVID | NFC | SciF | Touché | DBP | Avg | joint better |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Jev | +0.7 | +0.6 | +1.8 | +1.7 | +2.1 | −0.1 | −0.1 | +1.3 | −0.4 | +2.0 | +1.0 | 7/10 |
| Clef-Flash | −0.1 | −2.7 | −2.0 | −0.7 | −3.3 | +0.5 | −2.3 | −3.3 | −0.7 | −3.9 | −1.9 | 1/10 |
| Clef | +1.8 | +1.1 | +1.6 | +5.2 | +3.3 | −0.6 | −1.5 | −9.0 | −2.4 | −0.9 | −0.1 | 5/10 |
| Kev-4B | +2.9 | −0.4 | +0.2 | −2.2 | +2.9 | −7.4 | −4.8 | −8.3 | +0.7 | +1.0 | −1.5 | 5/10 |
| Kev-27B | +0.4 | +2.3 | −0.2 | +0.0 | +2.5 | +0.6 | +0.3 | +1.6 | +6.6 | −0.4 | +1.4 | 8/10 |

A window split happens when a joint request is too long for the model and gets halved. Splits per query in joint
mode:

| Model | DL19 | DL20 | DL21 | DL22 | DL23 | COVID | NFC | SciF | Touché | DBP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Jev | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.9 | 0.6 | 0.9 | 2.7 | 0.0 |
| Clef-Flash, Clef | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 3.0 | 2.0 | 3.1 | 7.1 | 1.0 |
| Kev-4B, Kev-27B | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.7 | 0.0 |

**What it shows:**
- Joint scoring helps the strong models (Jev, Kev-27B).
- Clef gains on the short DL passages and loses on the long BEIR ones (SciFact −9.0).
- **The splits do not explain the losses.** Kev-4B never splits on COVID or SciFact, yet loses 7–8 points there. Clef
  splits on every DL query and still gains on DL. The better explanation is the length of the joint state:
  100 long passages, about 40k tokens, are too much for the smaller models. This needs a decision (see open
  questions).

### RQ3: calibration (ECE, 15 bins, judged pairs of DL19-23; lower is better)

| System | DL19 | DL20 | DL21 | DL22 | DL23 | mean |
|---|---:|---:|---:|---:|---:|---:|
| monoT5-3B | 0.185 | 0.248 | 0.359 | 0.347 | 0.421 | 0.312 |
| Qwen3-Reranker-8B | 0.123 | 0.188 | 0.309 | 0.298 | 0.365 | 0.257 |
| Set-Encoder-large | – | – | 0.474 | 0.547 | 0.580 | 0.534 |
| bge-reranker-v2-m3 | – | – | 0.291 | 0.226 | 0.269 | 0.262 |
| Qwen3.7-Flash | 0.132 | 0.181 | 0.185 | 0.117 | 0.192 | 0.161 |
| Jev joint / isolated | 0.157 / 0.192 | 0.227 / 0.266 | 0.244 / 0.290 | 0.225 / 0.253 | 0.265 / 0.301 | 0.224 / 0.260 |
| Clef-Flash joint / isolated | 0.066 / 0.116 | 0.157 / 0.154 | 0.156 / 0.214 | 0.154 / 0.171 | 0.167 / 0.210 | 0.140 / 0.173 |
| Clef joint / isolated | 0.095 / 0.174 | 0.155 / 0.238 | 0.162 / 0.271 | 0.151 / 0.232 | 0.172 / 0.281 | 0.147 / 0.240 |
| Kev-4B joint / isolated | 0.174 / 0.137 | 0.259 / 0.214 | 0.243 / 0.234 | 0.297 / 0.244 | 0.283 / 0.249 | 0.251 / 0.216 |
| Kev-27B joint / isolated | 0.098 / 0.203 | 0.145 / 0.257 | 0.186 / 0.309 | 0.163 / 0.252 | 0.189 / 0.290 | 0.156 / 0.262 |
| Laya | 0.066 | 0.102 | 0.134 | 0.173 | 0.178 | 0.131 |

**What it shows:**
- Joint scoring lowers ECE for four of the five models (not for Kev-4B).
- The trained rerankers' scores are poorly calibrated, worst on DL21-23. The Set-Encoder's sigmoid was never meant
  as a probability.
- Laya is the best calibrated and the weakest ranker: calibration and ranking quality are separate properties.

### Cost and efficiency

**API cost (measured):**
- Per 1,000 queries: Jev joint $1.18, Jev isolated $2.29, Qwen3.7-Flash $0.82.
- The whole 10-dataset grid cost $1.62 (Jev joint), $3.16 (Jev isolated) and $1.13 (Qwen3.7-Flash).

**From the literature:**
- RankGPT-4 (GPT-4-0613) costs $0.641 per query and takes 82 s, with 19,890 input and 732 output tokens (Jin et al.
  2025).
- RankZephyr takes 2.6–3.1 s per query on one RTX 4090 with vLLM (Chen, Pradeep and Lin 2025).

**Efficiency run (queued):**
- All 14 local systems run on DL19/20 on one A100-SXM-80GB, with GPU-board energy from NVML.
- The energy measurement works: on an L40S, bge-m3 used 123 J per query at 0.33 s.
- The RankZephyr trial took 21 s per query on an L40S. That is slower than the published number because we use
  transformers rather than vLLM, the same as for every other local model.

### Sanity checks

- **BM25 vs Pyserini's reference scores (2CR):** match on all ten datasets. ✔
- **monoT5-3B on DL19/20:** 71.8 / 68.9, against the published 71.83 / 68.89. ✔
- **Jev joint vs S1Rank:** same model, question and candidates. Within 0.5 points on DL19, DL20, NFCorpus and
  SciFact; **COVID is 1.5 points lower** (85.0 vs 86.5). ⚠ Worth a look.
- **RankZephyr:** to be checked against the published 74.2 / 70.2 when the DL19/20 runs finish.

## Decisions so far

| Date | Decision |
| --- | --- |
| 2026-10-06 | arXiv preprint first, then a WSDM 2027 short paper; arXiv before Nov 17 |
| 2026-10-07 | One setup: BM25 top-100 everywhere; published numbers stand in for runs; S1Rank's terms (joint/isolated, both pointwise) |
| 2026-10-08 | Headline: open matches closed; RQ1 reworded to lead with it; title stays open |
| 2026-10-08 | Efficiency: API cost plus one fixed-GPU latency/energy run |
| 2026-10-08 | RankZephyr run by us (DL19-23); RankGPT-4 from the literature only |
| 2026-10-08 | Main table in joint mode only; isolated runs go to the RQ2 table |
| 2026-10-08 | Kanishka's QPP results get one paragraph; RQ2's explanation goes into the main text |

## TODOs

**Runs (Phase 1, roadmap step 4)**
- [ ] RankZephyr on DL19-23 (running), then the check against 74.2 / 70.2.
- [ ] Efficiency run, 28 jobs (queued), then a latency/energy table.

**Analysis (step 5)**
- [ ] Paired t-tests for joint vs. isolated, Holm-corrected; RQ2 table generated into `paper/tables/`.
- [ ] Calibration table and one reliability diagram.
- [ ] Efficiency table: latency, energy and cost per query, with the literature numbers marked.
- [ ] RQ2 explanation: joint gain against state length in tokens per model.
- [ ] Look into Jev's COVID gap to S1Rank.

**Paper (step 6)**
- [ ] Move to ACM sigconf, 4 pages.
- [ ] Write the abstract, introduction (motivation, contributions), related work (three paragraphs), setup table, the
  RQ1–RQ3 sections and the conclusion. All are still `\TODO` or `\PENDING`.
- [ ] QPP paragraph (Kanishka).
- [ ] Fill the model facts (parameters, context) from the model cards.
- [ ] Verify the bib entries flagged "to verify" (FIRST venue, Qwen3-Embedding authors).
- [ ] Check for System One reranking papers newer than October 2026.

**Repository**
- [ ] Merge `dev-michael` into `master` (41 commits ahead). How the two codebases are combined is still open.

## Open questions for the meeting

1. **Combining the codebases.** `master` holds Kanishka's `decisionrerank`, `dev-michael` holds `cdm`. Which one
   produces the paper's numbers, and how do we merge?
2. **QPP paragraph.** Kanishka's QPP results use the official 1000-candidate DL19/20 pools. Should they be rerun on
   our BM25 top-100 outputs (stored probabilities per query and passage are available), and who does it?
3. **RQ2 explanation.** The data points to state length, not window splits. Do we frame it as "joint scoring needs
   enough model capacity for long states" and add a length analysis?
4. **What "matches" means.** Kev-27B vs Jev is 61.1 vs 60.5. Do we claim parity with a paired test (or an
   equivalence margin), or only "comparable"?
5. **Writing split and timeline** up to the arXiv post and Nov 17.

## Risks and limitations

- **Literature numbers** (RankGPT-4, RankZephyr BEIR, part of bge/monoT5) have no per-query runs, so we cannot test
  against them. RankGPT-4 is the 2023 GPT-4.
- **Benchmark contamination:** the training data of Jev, Clef, Kev and the LLMs is unknown.
- **Jev is closed and versioned by its vendor.** We pinned jev-1.13.0; a later version may differ.
- **Energy** is GPU-board only. RankZephyr's latency is comparable within our setup, not to vLLM numbers.
- **Calibration** is measured only on DL19-23, the datasets with judged non-relevant passages.
- **15 NFCorpus queries** have no BM25 hit and count with nDCG 0 for every system.
- **4 pages are tight** for three RQs plus efficiency; the RQ2 length analysis and the QPP paragraph compete for
  space.

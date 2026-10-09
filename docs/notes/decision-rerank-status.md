# Calibrated System One Models as Rerankers: status 2026-10-09

> We test whether System One decision models, the closed Jev and the open Clef, Kev and Laya, rerank BM25 top-100
> candidates as well as strong trained rerankers, whether scoring all candidates in one request (joint) beats
> scoring each alone (isolated), and whether their probabilities are calibrated. **Headline: open matches closed.**
> The open Kev-27B (61.1 average nDCG@10) matches the closed Jev (60.5), and both beat every trained reranker and
> our own RankZephyr run (57.3) on average. **Since 2026-10-08:** RankZephyr reproduces its published DL19/20
> numbers; the fixed-GPU efficiency run is done (a few reruns queued); the new **triplet** comparison (base model /
> decision fine-tune / reranker fine-tune on one backbone) has 325 of 350 cells. RQ3 is open again: the owner is
> rethinking it. Target: WSDM 2027 short paper (4 pages, ACM sigconf), arXiv first. **Deadline: Nov 17, 2026.**

## Research questions and headline

- **RQ1:** Do open System One models rerank BM25 candidates as well as the closed Jev and strong trained rerankers?
- **RQ2:** Does joint pointwise scoring, with all candidates in one state, beat isolated pointwise scoring of each
  query–passage pair, and does that hold beyond Jev?
- **RQ3:** Are the relevance probabilities of System One models calibrated, and does joint scoring change that?
  *(Wording under review, decision 12.)*

Working title: *Calibrated System One Models as Rerankers* (kept open on purpose).

| Claim | Evidence | Status |
| --- | --- | --- |
| An open model matches the closed one | Kev-27B joint 61.1, Jev joint 60.5 average over 10 datasets | supported; no significance test yet |
| Strong System One models beat trained rerankers | Qwen3-Reranker-8B 59.4, Set-Encoder 59.0, monoT5-3B 57.5, bge 56.6 | supported on average; NFCorpus and Touché are exceptions; several baseline cells are literature numbers |
| They beat a listwise LLM reranker | Kev-27B vs RankZephyr-7B (our run on DL19-23): 61.1 vs 57.3 average; DL23 49.2 vs 44.1 | supported; RankZephyr's BEIR cells are literature numbers. Against RankGPT-4 (literature only): mixed, DL19 73.9 vs 75.6 |
| Size matters more than open vs. closed | Clef-Flash 52.8, Kev-4B 49.8, Laya 44.0 | supported |
| Decision fine-tuning helps only at scale | at 27B Kev joint 61.1 vs the base model asked zero-shot 57.4; at 0.6B–9B Kev is at or below its base model, and the reranker fine-tune wins clearly at every size where one exists | new, triplet grid nearly complete; **touches the headline** (see open questions) |
| Joint scoring helps strong models only | Jev +1.0 (7/10), Kev-27B +1.4 (8/10); Clef-Flash −1.9, Kev-4B −1.5 | supported descriptively; tests pending |
| Joint losses come from state length | Kev-0.6B joint AUC 0.49/0.47 on DL19/20 vs 0.64 isolated; small Kev trained on states ≤ 7.5k tokens, a joint DL state is ~13k, BEIR ≥ 40k | partly: mechanism shown for Kev; no length-controlled run yet |
| Joint scoring improves calibration | mean ECE joint vs isolated: Kev-27B 0.156 vs 0.262, Clef 0.147 vs 0.240, Jev 0.224 vs 0.260 | supported except Kev-4B; RQ3 framing open |
| Cheap compared with LLM rerankers | Jev joint $1.18 per 1k queries (measured) vs RankGPT-4 $641 (literature); Kev-27B joint 14.3 s vs RankZephyr 22.3 s per DL19 query, same A100 | supported; Kev-27B is 5x slower than Qwen3-Reranker-8B |

## Results

### RQ1: effectiveness (nDCG@10 × 100, BM25 top-100)

System One models in joint mode (Laya isolated). \* = published number, not recomputed. Bold = best per column.
$/1k q = API cost per 1,000 queries. Qwen3.7-Flash is dropped from the paper (decision 11).

| System | DL19 | DL20 | DL21 | DL22 | DL23 | COVID | NFC | SciF | Touché | DBP | Avg | $/1k q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 50.6 | 48.0 | 44.6 | 26.9 | 26.2 | 59.5 | 32.2 | 67.9 | 44.2 | 31.8 | 43.2 | – |
| RankGPT-4 | **75.6\*** | 70.6\* | – | – | – | 85.5\* | 38.5\* | 75.0\* | 38.6\* | **47.1\*** | – | – |
| RankZephyr-7B | 74.3 | 70.0 | 67.9 | 50.5 | 44.1 | 80.7\* | 36.6\* | 75.1\* | 31.1\* | 43.2\* | 57.3 | – |
| bge-reranker-v2-m3 | 72.2\* | 67.0\* | 66.1 | 51.5 | 45.1 | 74.8\* | 33.8\* | 73.5\* | 39.9\* | 41.9\* | 56.6 | – |
| monoT5-3B | 71.8 | 68.9 | 66.7 | 49.5 | 45.0 | 80.7\* | **39.0\*** | 76.6\* | 32.4\* | 44.5\* | 57.5 | – |
| Qwen3-Reranker-8B | 73.2 | 66.8 | 67.0 | 51.2 | 45.2 | 84.0 | 37.3 | 79.9 | **43.1** | 46.2 | 59.4 | – |
| Set-Encoder-large | 75.0\* | 72.2\* | 69.9 | **53.4** | 47.7 | 81.2 | 36.8 | 73.8 | 33.2 | 47.0 | 59.0 | – |
| Jev | 73.3 | 71.4 | **72.2** | 50.0 | 46.8 | 85.0 | 38.2 | 81.0 | 39.9 | 46.8 | 60.5 | 1.18 |
| Clef-Flash | 68.9 | 67.4 | 64.3 | 44.3 | 40.7 | 82.4 | 35.0 | 65.2 | 21.2 | 38.7 | 52.8 | – |
| Clef | 71.5 | 69.8 | 68.3 | 49.4 | 46.0 | 82.8 | 36.2 | 62.6 | 31.2 | 43.0 | 56.1 | – |
| Kev-4B | 65.5 | 62.9 | 62.4 | 37.7 | 38.3 | 75.3 | 29.3 | 62.4 | 23.7 | 40.7 | 49.8 | – |
| **Kev-27B** | 73.9 | **73.2** | 70.0 | 51.3 | **49.2** | **86.0** | 38.6 | **81.3** | 40.4 | 47.0 | **61.1** | – |
| Laya | 60.1 | 54.5 | 53.7 | 35.3 | 33.1 | 74.3 | 28.8 | 43.7 | 24.0 | 32.7 | 44.0 | – |

**What it shows:**
- Kev-27B and Jev are the two best systems on average; Kev-27B is best on DL20, DL23, COVID and SciFact.
- RankZephyr, now run by us on DL19-23, lands at 57.3: below both, about level with monoT5-3B.
- Every reranker falls below BM25 on Touché (44.2), a known effect.

### Triplets: one backbone, three training objectives (nDCG@10 × 100)

Base = the post-trained release asked S1Rank's question zero-shot (P(yes) of the first answer token). ZS avg = the
BEIR sets without training splits (COVID, Touché, DBPedia). – = cell still running.

| Backbone | Base (zero-shot) | Decision FT: Kev joint / isolated | Reranker FT | ZS avg: base / Kev best / reranker |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 23.8 | 13.2 / 27.0 | 56.1 (Qwen3-Reranker) | 25.4 / 30.9 / 54.9 |
| Qwen3-8B | 53.3 | – / 52.7 | 59.4 (Qwen3-Reranker) | 47.1 / 49.1 / 57.8 |
| Qwen3.5-0.8B | 37.6 | 25.3 / 35.9 | 57.1 (Prism) | 37.6 / 40.7 / 53.8 |
| Qwen3.5-4B | 52.7 | 49.8 / 51.4 | 58.7 (Prism) | 47.1 / 48.4 / 54.2 |
| Qwen3.5-9B | 53.7 | – / 50.1 (Clef-Flash 52.8 / 54.7) | 59.0 (Prism) | 48.2 / 46.4 (Clef-Flash 48.8) / 53.7 |
| Qwen3.8-27B | 57.4 | 61.1 / 59.7 (Clef 56.1 / 56.2) | none exists | 52.8 / 57.8 / – |

**What it shows:**
- Reranker fine-tuning beats both the base model and the decision fine-tune at every size where it exists, also on
  the zero-shot subset. Its training data includes MS MARCO (in-domain for DL19-23); Kev's does not.
- Decision fine-tuning gains over the base model only at 27B (+3.7). Below that, Kev is at or under its base model,
  and joint mode is far worse for the small models (Kev-0.6B joint 13.2, below BM25).
- Robustness check (Option B): the pretrained Qwen3-8B-Base with a completion prompt gets 41.4, against 53.3 for the
  post-trained release.

### RQ2: joint minus isolated (nDCG@10 points; positive = joint better)

| Model | DL19 | DL20 | DL21 | DL22 | DL23 | COVID | NFC | SciF | Touché | DBP | Avg | joint better |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Jev | +0.7 | +0.6 | +1.8 | +1.7 | +2.1 | −0.1 | −0.1 | +1.3 | −0.4 | +2.0 | +1.0 | 7/10 |
| Clef-Flash | −0.1 | −2.7 | −2.0 | −0.7 | −3.3 | +0.5 | −2.3 | −3.3 | −0.7 | −3.9 | −1.9 | 1/10 |
| Clef | +1.8 | +1.1 | +1.6 | +5.2 | +3.3 | −0.6 | −1.5 | −9.0 | −2.4 | −0.9 | −0.1 | 5/10 |
| Kev-4B | +2.9 | −0.4 | +0.2 | −2.2 | +2.9 | −7.4 | −4.8 | −8.3 | +0.7 | +1.0 | −1.5 | 5/10 |
| Kev-27B | +0.4 | +2.3 | −0.2 | +0.0 | +2.5 | +0.6 | +0.3 | +1.6 | +6.6 | −0.4 | +1.4 | 8/10 |

Window splits per query in joint mode (a request too long for the model is halved): Jev 0 on DL, 0.6–2.7 on BEIR;
Clef and Clef-Flash 1.0 on DL, 2.0–7.1 on BEIR; Kev 0 everywhere except Touché (0.7).

**What it shows:**
- Joint scoring helps the strong models (Jev, Kev-27B) and hurts the small ones.
- The splits do not explain it (Kev-4B never splits on COVID/SciFact yet loses 7–8 points). State length does
  (decision 8). New evidence from the Kev cards: Kev-0.8B/4B were trained on states of at most 7,552 tokens,
  Kev-27B on up to 32,768; a joint DL state is about 13k tokens, a BEIR state 40k or more. Kev-0.6B joint does not
  discriminate at all (AUC 0.49/0.47 on DL19/20, almost every passage at p ≈ 0.83, p rising with position).
  Per-model AUC joint vs isolated: Kev-0.8B 0.63 vs 0.69, Kev-4B 0.81 vs 0.82, Kev-27B 0.92 vs 0.92/0.89.

### RQ3: calibration (ECE, 15 bins, judged pairs of DL19-23; lower is better)

| System | DL19 | DL20 | DL21 | DL22 | DL23 | mean |
|---|---:|---:|---:|---:|---:|---:|
| monoT5-3B | 0.185 | 0.248 | 0.359 | 0.347 | 0.421 | 0.312 |
| Qwen3-Reranker-8B | 0.123 | 0.188 | 0.309 | 0.298 | 0.365 | 0.257 |
| Set-Encoder-large | – | – | 0.474 | 0.547 | 0.580 | 0.534 |
| bge-reranker-v2-m3 | – | – | 0.291 | 0.226 | 0.269 | 0.262 |
| Jev joint / isolated | 0.157 / 0.192 | 0.227 / 0.266 | 0.244 / 0.290 | 0.225 / 0.253 | 0.265 / 0.301 | 0.224 / 0.260 |
| Clef-Flash joint / isolated | 0.066 / 0.116 | 0.157 / 0.154 | 0.156 / 0.214 | 0.154 / 0.171 | 0.167 / 0.210 | 0.140 / 0.173 |
| Clef joint / isolated | 0.095 / 0.174 | 0.155 / 0.238 | 0.162 / 0.271 | 0.151 / 0.232 | 0.172 / 0.281 | 0.147 / 0.240 |
| Kev-4B joint / isolated | 0.174 / 0.137 | 0.259 / 0.214 | 0.243 / 0.234 | 0.297 / 0.244 | 0.283 / 0.249 | 0.251 / 0.216 |
| Kev-27B joint / isolated | 0.098 / 0.203 | 0.145 / 0.257 | 0.186 / 0.309 | 0.163 / 0.252 | 0.189 / 0.290 | 0.156 / 0.262 |
| Laya | 0.066 | 0.102 | 0.134 | 0.173 | 0.178 | 0.131 |

**What it shows:**
- Joint scoring lowers ECE for four of five models (not Kev-4B); the trained rerankers are worst on DL21-23.
- Laya is the best calibrated and the weakest ranker: calibration and ranking quality are separate properties.
- **RQ3 is open** (decision 12). ECE alone invites the reviewer question "why does a reranker care?". The proposals
  (uses of the probability value: thresholding, QPP, adaptive depth) are in `docs/notes/rq3-calibration-proposal.md`.

### Cost and efficiency

**API cost (measured, whole 10-dataset grid):** Jev joint $1.62, Jev isolated $3.16 ($1.18 per 1k queries joint).

**Fixed-GPU run (DL19, one A100-SXM4-80GB, p50 per query, GPU-board energy from NVML):**

| System | s / query | J / query |
| --- | ---: | ---: |
| Set-Encoder-large | 0.24 | 83 |
| Laya | 0.34 | 128 |
| bge-reranker-v2-m3 | 0.57 | 199 |
| monoT5-3B | 0.68 | 258 |
| Clef-Flash joint / isolated | 2.11 / 2.78 | 800 / 990 |
| Qwen3-Reranker-8B | 2.85 | 1139 |
| Prism-9B | 4.28 | 1711 |
| Clef joint / isolated | 6.43 / 8.00 | 2562 / 3115 |
| Kev-4B joint / isolated | 10.08 / 12.22 | 3054 / 3881 |
| Kev-27B joint / isolated | 14.34 / 23.13 | 3453 / 4811 |
| RankZephyr-7B | 22.30 | 4364 |

DL20 gives the same picture. Joint is faster than isolated for every System One model. Kev is slow for its size
(Kev-4B slower than Clef at 27B), likely its exact-dtype path; worth a look before the table goes in the paper.
Still queued on the A100s: Kev-8B/9B (joint and isolated) and Qwen3.8-27B prompted.

**From the literature:** RankGPT-4 $0.641 and 82 s per query (Jin et al. 2025); RankZephyr 2.6–3.1 s per query with
vLLM on an RTX 4090 (Chen, Pradeep and Lin 2025). Our RankZephyr uses transformers, like every other local model.

### Sanity checks

- **BM25 vs Pyserini 2CR:** match on all ten datasets. ✔
- **monoT5-3B DL19/20:** 71.8 / 68.9 vs published 71.83 / 68.89. ✔
- **RankZephyr DL19/20:** 74.3 / 70.0 vs published 74.2 / 70.2. ✔ (6–15% of window answers miss identifiers,
  filled in the old order as rank_llm does.)
- **Jev joint vs S1Rank:** within 0.5 points on DL19, DL20, NFCorpus, SciFact; **COVID 1.5 lower** (85.0 vs 86.5). ⚠
  Still unexplained.
- **causal-conv1d kernel:** changes Qwen3.5/3.8 scores in the last digits; on 3 DL19 queries nDCG@10 moved
  0.4945 → 0.4650 through near-ties (ECE/Brier unchanged). ⚠ Grid cells of Kev, Clef, Prism and the prompted
  Qwen3.5/3.8 models were mostly computed without it.

## Decisions so far

| Date | Decision |
| --- | --- |
| 2026-10-06 | arXiv preprint first, then a WSDM 2027 short paper; arXiv before Nov 17 |
| 2026-10-07 | One setup: BM25 top-100 everywhere; published numbers stand in for runs; S1Rank's terms (joint/isolated) |
| 2026-10-08 | Headline: open matches closed; RQ1 reworded to lead with it; title stays open |
| 2026-10-08 | Efficiency: API cost plus one fixed-GPU latency/energy run |
| 2026-10-08 | RankZephyr run by us (DL19-23); RankGPT-4 from the literature only |
| 2026-10-08 | Main table in joint mode only; isolated runs go to the RQ2 table |
| 2026-10-08 | Kanishka's QPP results get one paragraph; RQ2 explained by state length in the main text |
| 2026-10-09 | Triplets on six backbones (base / decision FT / reranker FT); base row = post-trained release, zero-shot; Option B check on Qwen3-8B |
| 2026-10-09 | Qwen3.7-Flash dropped from the paper (runs stay in results.csv) |
| 2026-10-09 | RQ3 open; proposals in `docs/notes/rq3-calibration-proposal.md` |

## TODOs

**Runs**
- [x] RankZephyr on DL19-23; check against 74.2 / 70.2 passes.
- [x] Efficiency run on DL19/20 (A100).
- [ ] Last triplet cells: Kev-8B joint on NFCorpus/SciFact, Kev-9B joint on SciFact (running).
- [ ] Efficiency reruns: Kev-8B/9B joint and isolated, Qwen3.8-27B prompted (queued).
- [ ] Decide on, then possibly rerun, the Qwen3.5/3.8 grid cells with the causal-conv1d kernel.
- [ ] Optional control for RQ2: joint requests cut to each model's trained state length.

**Analysis (step 5)**
- [ ] Paired t-tests for joint vs isolated (Holm); Kev-27B vs Jev parity test.
- [ ] Generated tables in `paper/tables/`: main, triplets, RQ2, calibration, efficiency.
- [ ] RQ2: joint gain against state length in tokens per model.
- [ ] RQ3 design (owner), then its table/plot.
- [ ] Look into Jev's COVID gap and Kev's slow inference path.

**Paper (step 6)**
- [ ] Move to ACM sigconf, 4 pages.
- [ ] All sections still `\TODO`/`\PENDING`: abstract, introduction, related work (3 paragraphs), setup table,
  RQ1–RQ3 sections (`04_reranking`, `05_joint_scoring`, `06_calibration`), conclusion.
- [ ] QPP paragraph (Kanishka); model facts from the cards; bib entries flagged "to verify".
- [ ] Related-work check for System One reranking papers newer than Oct 2026.

**Repository**
- [ ] Merge `dev-michael` into `master` (51 commits ahead).

## Open questions for the meeting

1. **Triplets vs the headline.** Below 27B the reranker fine-tune beats Kev clearly, and Kev barely beats its own
   base model. Does the paper say "decision models need scale" openly, and how does that sit with "open matches
   closed"? Where do the triplets go in 4 pages?
2. **The kernel change.** Rerun all Qwen3.5/3.8 grid cells with causal-conv1d (consistent code for every number) or
   keep them and state it? Rerunning costs GPU hours, mostly Kev-27B and Clef.
3. **RQ3.** Which of the proposals (thresholding, QPP, adaptive depth, or ECE only) do we take into the short paper?
4. **What "matches" means.** Kev-27B vs Jev is 61.1 vs 60.5. Paired test, equivalence margin, or "comparable"?
5. **QPP paragraph.** Rerun Kanishka's QPP on our BM25 top-100 probabilities, and who does it?
6. **Codebases, writing split and timeline** up to the arXiv post and Nov 17.

## Risks and limitations

- **Triplets weaken a simple story:** decision fine-tuning gains over the base model only at 27B, and the reranker
  fine-tunes have seen MS MARCO while Kev has not.
- **Mixed code versions:** most Qwen3.5/3.8 grid cells predate the causal-conv1d kernel, which moves nDCG through
  near-ties.
- **Literature numbers** (RankGPT-4, RankZephyr BEIR, part of bge/monoT5/Set-Encoder) have no per-query runs, so no
  tests against them. RankGPT-4 is the 2023 GPT-4.
- **Benchmark contamination:** training data of Jev, Clef, Kev and the base LLMs is unknown.
- **Jev is closed and versioned by its vendor** (pinned jev-1.13.0).
- **Energy** is GPU-board only; RankZephyr's latency is comparable within our setup, not to vLLM numbers.
- **Calibration** only on DL19-23; **15 NFCorpus queries** have no BM25 hit and count with nDCG 0.
- **4 pages** for three RQs, triplets and efficiency: something has to move to the arXiv version.

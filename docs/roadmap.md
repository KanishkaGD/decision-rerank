# Roadmap

The plan of record: what the paper claims, how it is tested, and the order of the work. Decisions behind it are in
`docs/intent/`, what happened along the way in `docs/experiment-log.md`. Keep this file current -- it describes the
plan as it is, not how it got here (the `straighten-branch` skill does that sweep).

**Working title:** *Calibrated System One Models as Rerankers* (deliberately open, `docs/intent/paper-focus.md`).
**Venue:** arXiv preprint, then a WSDM 2027 short paper (4 pages + references, ACM sigconf, anonymised), deadline
**Nov 17, 2026** (`docs/intent/venue.md`).

## Contribution

S1Rank showed that Jev, a closed System One model, reranks BM25 candidates as well as strong cross-encoders when it
scores all candidates in one request. We test whether that holds for System One models in general -- closed (Jev) and
open (Clef, Kev, Laya) -- against strong trained rerankers on ten datasets, whether the joint mode is what makes the
difference, and whether their probabilities are calibrated.

**Headline** (`docs/intent/paper-focus.md`): open matches closed -- an open System One model (Kev-27B) matches Jev
and beats every trained reranker on average; size matters more than open vs. closed. RQ2 and RQ3 support it.

## Research questions (paper)

| RQ | Question | Evidence |
| --- | --- | --- |
| RQ1 | Do open System One models rerank BM25 top-100 as well as the closed Jev and strong trained rerankers? | nDCG@10 on all ten datasets |
| RQ2 | Does joint pointwise scoring (all candidates in one state) beat isolated pointwise scoring (one candidate per request), and does that hold beyond Jev? | joint vs. isolated for Jev, Clef-Flash, Clef, Kev-4B, Kev-27B; Set-Encoder as the trained counterpart; paired tests |
| RQ3 | Are their relevance probabilities calibrated, and does joint scoring change that? | ECE and Brier on DL19-DL23 against the trained rerankers' probabilities |

Terminology (S1Rank's): every System One run is **pointwise** -- one Noul probability per candidate, ranked by it.
**Joint** = all candidates of a query in one state; **isolated** = one candidate per request. "Listwise" is reserved
for rerankers that output a ranking (RankGPT, RankZephyr).

## Setup

**Candidates.** BM25 top-100 of Pyserini's default BM25 (k1 0.9, b 0.4; BEIR flat indexes) for every dataset -- the
candidate set S1Rank, RankGPT, RankZephyr and Abdallah et al. (2025) rerank, so their published numbers belong in the
same table. For DL21-23 the official top-100 files are exactly this set. Our BM25 must reproduce Pyserini's 2CR
scores (`results/literature.csv`, role `reference`).

| Dataset | Queries | Corpus | Calibration (judged non-relevant passages) |
| --- | ---: | --- | :---: |
| DL19, DL20 | 43, 54 | MS MARCO v1 passage | yes |
| DL21, DL22, DL23 | 53, 76, 82 | MS MARCO v2 passage (official top-100 files) | yes |
| TREC-COVID, NFCorpus, SciFact, Touché, DBPedia | 50, 323, 300, 49, 400 | BEIR | no |

**Systems.** Computed by us unless a published number on the same candidates exists (`results/literature.csv`,
role `baseline`; `from_literature` in `slurm/config.sh` makes `slurm/submit.sh` skip those cells).

| System | Kind | DL19/20 | DL21-23 | BEIR |
| --- | --- | --- | --- | --- |
| RankGPT-4 | listwise LLM reranker | literature | -- | literature |
| RankZephyr-7B | listwise LLM reranker (run by us) | computed (check against literature) | computed | literature |
| bge-reranker-v2-m3 | cross-encoder | literature | computed | literature |
| monoT5-3B | cross-encoder (seq2seq) | computed (check against literature; calibration) | computed | literature |
| Qwen3-Reranker-8B | pointwise LLM reranker | computed | computed | computed |
| Set-Encoder-large | cross-encoder, scores candidates jointly | literature | computed | computed |
| Jev, Clef-Flash, Clef, Kev-4B, Kev-27B | System One, joint and isolated | computed | computed | computed |
| Laya | System One, isolated only (context too short) | computed | computed | computed |

**Triplets** (`docs/intent/paper-focus.md`, decided 2026-10-09): one backbone, three training objectives -- none
(the post-trained release asked S1Rank's question zero-shot, `*-prompted`), decision fine-tuning (Kev, Clef; joint
and isolated) and reranker fine-tuning (Qwen3-Reranker, Prism). All computed on all ten datasets.

| Backbone | Base model (zero-shot) | Decision fine-tune | Reranker fine-tune |
| --- | --- | --- | --- |
| Qwen3-0.6B | `qwen3-0.6b-prompted` | Kev-0.6B (on -Base) | Qwen3-Reranker-0.6B (on -Base) |
| Qwen3-8B | `qwen3-8b-prompted`; `qwen3-8b-base-prompted` (-Base, completion prompt: robustness check) | Kev-8B (on -Base) | Qwen3-Reranker-8B (on -Base) |
| Qwen3.5-0.8B | `qwen3.5-0.8b-prompted` | Kev-0.8B (on -Base) | Prism-0.8B (checkpoint not stated) |
| Qwen3.5-4B | `qwen3.5-4b-prompted` | Kev-4B (on -Base) | Prism-4B |
| Qwen3.5-9B | `qwen3.5-9b-prompted` | Kev-9B (on -Base), Clef-Flash | Prism-9B |
| Qwen3.8-27B | `qwen3.8-27b-prompted` | Kev-27B, Clef (on the post-trained release) | -- (none exists) |

Training data differs between the fine-tunes: Qwen3-Reranker and Prism are trained on MS MARCO among much else
(in-domain for DL19-23); Kev's public sources contain no MS MARCO or BEIR. The paper reports the BEIR collections
without training splits (TREC-COVID, Touché, DBPedia) as the zero-shot subset next to the averages.

**Question.** S1Rank's Noul question (phrasing `s1rank`), fixed for all System One models; no phrasing search.

**Metrics.** nDCG@10 everywhere; ECE (15 bins) and Brier on judged pairs of DL19-23, relevant from grade 2; API
cost. Latency and energy from one efficiency run: every local model on DL19/20 on one A100-SXM-80GB, energy from
NVML per run; API models get cost and wall-clock time only.

**Main table.** One row per System One model (joint; Laya isolated) plus the baselines; the isolated rows are in the
RQ2 table (joint minus isolated, with tests).

**Significance.** Paired t-test over queries for joint vs. isolated per model and dataset (from the stored
per-query metrics; no extra runs), Holm-corrected across datasets. Literature numbers are not tested (no per-query
runs); the paper says so.

**Sanity checks.** BM25 = Pyserini 2CR on every dataset; monoT5-3B on DL19/20 within about one point of the published
71.83 / 68.89; our `jev-joint` against S1Rank's 73.7 / 71.7 (same model, question and candidates).

## Order of work

### Phase 1 -- the paper (until Nov 17, 2026)

1. **Candidate sets** for all ten datasets; check BM25 against 2CR. *DL21-23 done (exact match).*
2. **Sanity checks**: monoT5-3B on DL19/20, `jev-joint` against S1Rank.
3. **Grid runs** -- the cells of `slurm/config.sh` (`experiment-status` skill, section `next`). *Done 2026-10-08
   (146 computed, 14 from the literature).*
4. **RankZephyr and the efficiency run** (`docs/intent/paper-focus.md`): RankZephyr-7B on DL19-23 with rank_llm's
   prompt (`src/cdm/rerankers/listwise.py`; DL19/20 against the published 74.2 / 70.2); GPU energy from NVML in
   results.csv (`src/cdm/energy.py`); all local models on DL19/20 on one A100-SXM-80GB.
5. **Analysis**: main table (systems x datasets, plus cost) generated from `results/results.csv` and
   `results/literature.csv` into `paper/tables/`; joint-vs-isolated tests; calibration table and reliability plot.
6. **Related-work check** for System One reranking papers newer than Oct 2026; move the paper to ACM sigconf; write;
   post on arXiv; submit to WSDM.

### Phase 2 -- after the submission (arXiv extension, follow-up paper)

7. **Probabilities as a control signal**: query performance prediction (Kanishka's QPP results on DL19/20 are the
   starting point), "no relevant document" detection, adaptive rerank depth. Uses the Phase 1 outputs; little new
   compute.
8. **Fine-tuning** an open System One model on MS MARCO (`msmarco-dev` for selection), with and without a
   calibration term in the loss; effectiveness and calibration before and after. Model to be chosen from the Phase 1
   results (Laya, Kev-4B).
9. **BRIGHT**: where fast System One judges fail on reasoning-intensive queries, against an LLM reranker, with an
   error analysis of 20 failure cases.

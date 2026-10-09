# Paper notes

Working notes for the paper: where every number from outside our runs comes from, what the results say so far, and
what is still open. The plan of record stays `docs/roadmap.md`; numbers in the paper itself come only from the
generated tables (`paper/tables/`, `python -m cdm.paper_tables`). Numbers quoted below are a snapshot of
`results/results.csv` on **2026-10-08** (grid complete: 146 cells computed, 14 from the literature) -- regenerate,
don't copy.

## 1. Source register

Every number we use but did not compute. Machine-readable copy: `results/literature.csv` (`role` = `baseline`: in the
main table; `reference`: sanity checks; `efficiency`: cost/latency). BibTeX keys refer to `paper/sources.bib`.

| What | Value(s) | Source | BibTeX key | Used for |
| --- | --- | --- | --- | --- |
| RankGPT-4, RankZephyr-7B, monoT5-3B, bge-reranker-v2-m3: nDCG@10 on BM25 top-100, DL19/20 + 5 BEIR | Tables 2-3 | Abdallah et al. 2025, *How Good are LLM-based Rerankers?* (EMNLP Findings) | `Abdallah2025HowGood` | main table (dagger) |
| Set-Encoder-large DL19/20 nDCG@10 | 75.0 / 72.2 (authors' bf16 run) | webis/set-encoder-large model card | `SetEncoderCard`, `Schlatt2025SetEncoder` | main table (dagger) -- **caption currently credits only Abdallah et al.: fix** |
| BM25 reference scores, all 10 datasets | e.g. DL19 50.58, DL20 47.96, NFCorpus 32.18 | Pyserini 2CR (bm25-flat) | `Pyserini2CR`, `Lin2021Pyserini` | sanity check: our BM25 matches on every dataset |
| S1Rank's Jev (joint), bge-m3, monoT5-3B, BM25 | Jev DL19 73.7, DL20 71.7, COVID 86.5, NFC 38.3, SciFact 80.5 | S1Rank README (read 2026-10-07) | `S1Rank` | sanity check of our Jev run |
| RankGPT (GPT-4-0613) cost per query on TREC DL | 19,890 in / 732 out tokens, 82 s, $0.641 | Jin et al. 2025, RankFlow (WWW Companion '25), Table 9 | `Jin2025RankFlow` | efficiency comparison; validates our token model |
| RankZephyr latency and tokens per query, DL19-23 | 15.7k-18.2k in / 711 out tokens; 2.6-3.1 s on one RTX 4090 | Chen, Pradeep, Lin 2025 (SIGIR), Table 5 | `Chen2025FIRSTRepro` (preprint arXiv:2411.05508) | efficiency comparison |
| FIRST (single-token listwise reranking, ~40-50% faster than RankZephyr) | -- | Reddy et al. 2024 | `Reddy2024FIRST` (venue to verify) | related work |
| Cross-encoders up to 173x faster than LLM rerankers at similar effectiveness | -- | Schlatt et al. 2025, Rank-DistiLLM (ECIR) | `Schlatt2025RankDistiLLM` | related work / motivation |
| Jev price | $0.042 per M input tokens, output free | TypeSafe docs (2026-10-06); same on OpenRouter (2026-10-08) | `TypeSafeModels`, `OpenRouterJev` | cost column |
| LLM list prices (OpenRouter) | see section 4 | OpenRouter models API (2026-10-08) | `OpenRouterModels` | RankGPT cost estimate, Qwen3.7-Flash price |
| Model checkpoints | pinned revisions | Hugging Face / GitHub | `Clef`, `Kev`, `Laya`, `Nogueira2020monoT5`, `Chen2024BGEM3`, `Zhang2025Qwen3Embedding` | setup table |

Disagreements between sources (pick one source per table and say so):

- monoT5-3B NFCorpus: 39.0 (Abdallah et al.) vs 37.0 (S1Rank); bge-m3 TREC-COVID 74.8 vs 79.0. Our own DL19/20 monoT5
  runs match Abdallah et al. within 0.1 (71.8 / 68.9 vs 71.83 / 68.89), so the main table uses Abdallah et al.
- Set-Encoder-large: model card 75.0 / 72.2 on DL19/20 (bf16); our own runs on other datasets are fp32.

Not citable as is (found, not verified): a RankZephyr "~110x slower than monoELECTRA" claim attributed to a
Rank-DistiLLM-type paper -- not on the abstract page; check the full text before using it.

## 2. Sanity checks (all pass)

- **BM25 = Pyserini 2CR** on all ten datasets (DL21-23 official top-100 files: exact match).
- **monoT5-3B** DL19/20: 71.8 / 68.9 vs published 71.83 / 68.89.
- **Jev joint vs S1Rank** (same model jev-1.13.0, question, candidates): DL19 73.3 vs 73.7, DL20 71.4 vs 71.7,
  COVID 85.0 vs 86.5, NFCorpus 38.2 vs 38.3, SciFact 81.0 vs 80.5. COVID is 1.5 points lower -- the only gap above one
  point; possibly passage text/title handling. Check before writing "reproduces".

## 3. Findings so far (snapshot 2026-10-09)

nDCG@10 x 100. Avg = mean over all ten datasets; ZS = mean over TREC-COVID, Touché and DBPedia (no training split in
any fine-tune's data). Triplet grid 325 of 350 cells; Kev-8B joint (NFCorpus, SciFact) and Kev-9B joint (SciFact)
still running.

**RQ1, triplets (the headline).** Reranker fine-tune vs. the better decision mode vs. base model, Avg / ZS:

| Backbone | Base | Decision fine-tune (better mode) | Reranker fine-tune |
| --- | --- | --- | --- |
| Qwen3-0.6B | 23.8 / 25.4 | Kev-0.6B isolated 27.0 / 30.9 | Qwen3-Reranker-0.6B **56.1 / 54.9** |
| Qwen3-8B | 53.3 / 47.1 | Kev-8B isolated 52.7 / 49.1 | Qwen3-Reranker-8B **59.4 / 57.8** |
| Qwen3.5-0.8B | 37.6 / 37.6 | Kev-0.8B isolated 35.9 / 40.7 | Prism-0.8B **57.1 / 53.8** |
| Qwen3.5-4B | 52.7 / 47.1 | Kev-4B isolated 51.4 / 48.4 | Prism-4B **58.7 / 54.2** |
| Qwen3.5-9B | 53.7 / 48.2 | Clef-Flash isolated 54.7 / 48.8; Kev-9B isolated 50.1 / 43.8 | Prism-9B **59.0 / 53.7** |
| Qwen3.8-27B | 57.4 / 52.8 | Kev-27B joint **61.1 / 57.8** | none exists |

- Reranker fine-tuning wins on every backbone where it exists, by 4-29 points on average and also on the zero-shot
  subset. Its training data includes MS MARCO (in-domain for DL19-23); Kev's does not -- a caveat for DL, not for ZS.
- Decision fine-tuning stays within a few points of the base model at every size (better mode minus base, Avg:
  0.6B +3.2, 8B -0.6, 0.8B -1.7, 4B -1.3, 9B -3.6 Kev / +1.0 Clef-Flash, 27B +3.7); reranker fine-tuning adds 5-32
  points over the base model.
- Robustness: Qwen3-8B-Base with a completion prompt reaches 41.4, the post-trained release 53.3 -- the base row uses
  the post-trained release.

**RQ1, joint vs. isolated (no significance tests yet).** Joint helps Jev (+1.0, 7/10) and Kev-27B (+1.4, 8/10); it
hurts Clef-Flash (-1.9) and Kev-4B (-1.5), and Clef gains on DL but loses on BEIR (SciFact -9.0). Window splits do
not explain it (Kev-4B never splits on COVID/SciFact yet loses 7-8 points). The state length does: Kev-0.8B/4B were
trained on states of at most 7,552 tokens, Kev-27B on up to 32,768 (KevCards); a joint DL state is about 13k tokens,
a BEIR state 40k or more. Kev-0.6B joint does not discriminate at all (AUC 0.49 / 0.47 on DL19/20, p rising with the
position in the state).

**RQ1, context.** Kev-27B joint 61.1 and Jev joint 60.5 lead on average, ahead of Qwen3-Reranker-8B 59.4,
Set-Encoder-large 59.0, monoT5-3B 57.5, RankZephyr-7B 57.3 (our DL runs, BEIR from the literature) and bge 56.6.
Open decision fine-tunes match the closed Jev; the 27B decision fine-tune edges reranker fine-tunes a third its
size. Laya 44.0 is barely above BM25 (43.2). Every reranker is below BM25 on Touché (known effect).

**RQ2 (ECE on DL19-23, 15 bins, unjudged excluded).**
- Joint scoring lowers ECE for Jev, Clef, Clef-Flash and Kev-27B (0.156 vs 0.262 mean); Kev-4B is the exception.
- Reranker fine-tunes' native probabilities are worse calibrated on DL21-23: Qwen3-Reranker-8B 0.30-0.37,
  monoT5-3B 0.36-0.42, Set-Encoder 0.47-0.58 (sigmoid of a logit, never meant as a probability).
- Laya is the best calibrated (mean 0.131) and the weakest ranker: calibration and ranking quality are separate
  properties.
- Not run yet: the cutoff analysis (p >= 0.5) and post-hoc scaling of the reranker fine-tunes on msmarco-dev
  (`docs/notes/calibration-proposal.md`, Parts A and C). Without Part C, "just calibrate the reranker" is unanswered.

**Cost (measured).** API cost per 1,000 queries: Jev joint $1.18, Jev isolated $2.29. Whole grid: Jev joint $1.62,
Jev isolated $3.16. RankGPT-4 as published: $641 per 1,000 queries (Jin et al.)
-- about 500x Jev joint.

**Latency and energy (A100-SXM4-80GB, DL19, p50 per query).** Set-Encoder 0.24 s / 83 J, monoT5-3B 0.68 s / 258 J,
Qwen3-Reranker-8B 2.85 s / 1,139 J, Prism-9B 4.28 s / 1,711 J, Kev-27B joint 14.34 s / 3,453 J, Kev-27B isolated
23.13 s / 4,811 J, RankZephyr 22.30 s / 4,364 J. Joint is faster than isolated for every decision model. Kev is slow
for its size (Kev-4B joint 10.08 s, slower than Clef at 27B): its exact-dtype path, to check before the table.

## 4. Cost estimate: a current RankGPT on our candidates (not run)

Token model: sliding window 20, step 10, 9 calls per query on 100 candidates, passages cut at 300 words. Input per
query = 1,800 + 180 x passage tokens (capped at 390); output 732 per query (Jin et al.). Passage lengths from our Jev
token counts: DL 92-114, DBPedia 116, COVID 384, NFCorpus 409, SciFact 415, Touché 858 tokens. Check: DL estimate 21.9k
input tokens per query vs 19.9k measured by Jin et al. Totals: DL21-23 3.9M in / 0.15M out (211 queries); all ten
datasets 67M in / 1.05M out (1,430 queries; NFCorpus and SciFact are 45M of it).

| Model (OpenRouter list price, $/M in/out) | DL21-23 | all 10 |
| --- | ---: | ---: |
| gpt-4, the original RankGPT-4 (30 / 60) | $127 | $2,075 |
| gpt-4o (2.50 / 10) | $11 | $178 |
| gpt-6-sol, claude-sonnet-5.5 (2 / 10) | $9 | $145 |
| gemini-3.8-flash (0.75 / 3.75) | $3.50 | $54 |
| gpt-6-luna, claude-haiku-5.5 (0.10 / 0.50) | $0.50 | $7 |

Batch prices halve these. Reasoning models need reasoning off, or their reasoning tokens (billed as output) multiply
the cost.

## 5. Model facts for the setup table

| System | Parameters | Access | Context (ours) | Probability |
| --- | --- | --- | --- | --- |
| Jev 1.13.0 | undisclosed | TypeSafe API ($0.042/M in) | 32k | model's Noul |
| Clef / Clef-Flash | 27B / 9B | open weights | -- | model's Noul |
| Kev-27B / Kev-4B | 27B / 4B (LoRA on Qwen bases) | open weights | -- | model's Noul |
| Laya | small (check the card) | open weights | ~320 tokens of state | model's Noul (with its temperatures) |
| Qwen3-Reranker-8B | 8B | open weights | -- | softmax(yes, no) |
| monoT5-3B | 3B | open weights | 512 | softmax(true, false) |
| bge-reranker-v2-m3 | 568M | open weights | -- | sigmoid(logit) |
| Set-Encoder-large | ~335M (ELECTRA-large) | open weights | -- | sigmoid(logit) |
| Qwen3.7-Flash | undisclosed | OpenRouter ($0.03/$0.13 per M) | -- | P(yes) from top-5 log-probs |
| RankGPT-4 | undisclosed (GPT-4) | literature only | 8k | none (ranking only) |
| RankZephyr | 7B | literature only | 4k | none (ranking only) |

Fill the blanks from the model cards before the setup table is written.

## 6. Limitations and caveats to state

- Literature numbers are not recomputed and have no per-query runs: no significance tests against them.
- RankGPT-4 is the 2023 GPT-4; no listwise baseline on DL21-23 (RankZephyr reports DL21/22 only on SPLADE++).
- Benchmark contamination: training data of Jev, Clef, Kev and the LLMs is unknown.
- Jev is closed and versioned by its vendor (pinned jev-1.13.0; the OpenRouter route serves a dated snapshot).
- 15 NFCorpus queries have no BM25 hit; they count with nDCG 0 for every system (as in BEIR).
- Calibration only on DL19-23 (judged non-relevant passages); unjudged excluded there, counted as non-relevant
  in the `_unjudged0` variants.
- Latency of the grid runs comes from mixed hardware; the comparable numbers come from the efficiency run (variant
  `a100`, DL19/20, one A100-SXM-80GB, GPU-board energy from NVML -- not CPU or cooling). RankZephyr runs with
  transformers, not vLLM: its latency is comparable within our setup, not to rank_llm's published speed.

## 7. Open questions for the owner

Answered 2026-10-09 (`docs/intent/paper-focus.md`): the headline is the negative result (reranker fine-tuning beats
decision fine-tuning, 0.6B-9B); joint vs. isolated is part of RQ1; calibration is RQ2, argued in a discussion with
the cutoff analysis and post-hoc scaling on msmarco-dev; no 27B reranker fine-tune (stated as a limitation); a
question as the title.

Still open:

1. Citation for RLCD and the release dates of Jev, Clef and Kev (the "matched within weeks" argument).
2. Rerun the Qwen3.5/3.8 grid cells with the causal-conv1d kernel, or state that most were computed without it?
3. RQ2 metrics: set F1 at p >= 0.5 as the utility, or precision at the cutoff / a per-passage cost? Which msmarco-dev
   subset (size, judged pairs only) to fit the scaling on?
4. Kanishka's QPP results: named in the RQ2 discussion only, or rerun on our BM25 top-100 outputs?

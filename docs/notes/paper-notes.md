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

## 3. Findings so far (snapshot 2026-10-08)

nDCG@10 x 100; Avg over all ten datasets.

| System | Avg | Notes |
| --- | ---: | --- |
| Kev-27B joint | **61.1** | best average; best on DL20, DL23, COVID, SciFact |
| Jev joint | 60.5 | best on DL21; closed |
| Kev-27B isolated / Jev isolated | 59.7 / 59.5 | |
| Qwen3-Reranker-8B | 59.4 | best trained reranker; best on Touché (43.1, still below BM25 44.2) |
| Set-Encoder-large | 59.0 | best on DL22 (53.4); DL19/20 from the model card |
| monoT5-3B / bge-m3 | 57.5 / 56.6 | mostly literature cells |
| Clef joint / isolated | 56.1 / 56.2 | |
| Clef-Flash isolated / joint | 54.7 / 52.8 | |
| Qwen3.7-Flash (LLM judge, Jev's price) | 53.0 | the "plain LLM at the same price" control |
| Kev-4B isolated / joint | 51.4 / 49.8 | |
| Laya | 44.0 | barely above BM25 (43.2) |
| RankGPT-4 / RankZephyr-7B (literature) | -- | no DL21-23 on BM25 candidates; RankGPT-4 best on DL19 (75.6) |

**RQ1.** The two strongest System One models (Kev-27B, open; Jev, closed) beat every trained reranker on average and
match RankGPT-4 on BEIR and DL20, at about 2 points below it on DL19. Smaller open models (Clef-Flash, Kev-4B, Laya)
fall behind the trained rerankers. Size matters more than the open/closed split. Touché: every reranker is below BM25
(known effect).

**RQ2 (joint minus isolated, no significance tests yet).** Not uniform:
- Jev: joint better on 7 of 10, equal on COVID (+1.0 average).
- Kev-27B: joint better on average (+1.4), large on Touché (+6.6) and DL20/DL23 (+2.2/+2.5).
- Clef: joint better on all five DL sets (+1.1 to +5.2), isolated better on BEIR (SciFact -9.1, Touché -2.4,
  NFCorpus -1.5); averages equal (56.1 vs 56.2).
- Clef-Flash: isolated better on DL20-23 and SciFact (-3.2); average -1.9.
- Kev-4B: isolated better on average (-1.6), large on COVID (-7.4) and SciFact (-8.3).
- Reading: joint scoring helps the strongest models everywhere and the mid-sized ones on short passages (DL), but
  hurts them on long BEIR passages (COVID, NFCorpus, SciFact ~400 tokens, Touché ~860: 100 candidates no longer fit
  one request and get split). "Joint is what makes Jev work" (S1Rank) therefore holds only for strong models -- a
  candidate headline for RQ2. To check: window_splits per dataset as the explanatory variable.

**RQ3 (ECE on DL19-23, 15 bins, unjudged excluded).**
- Joint scoring lowers ECE for Jev (DL19 0.157 vs 0.192), Clef (0.095 vs 0.174), Kev-27B (0.098 vs 0.203) and
  Clef-Flash on most sets; Kev-4B is the exception.
- Trained rerankers are worse calibrated on DL21-23: monoT5-3B 0.36-0.42, Qwen3-Reranker-8B 0.30-0.37, Set-Encoder
  0.47-0.58 (sigmoid of a logit, never meant as a probability), against 0.16-0.19 for Kev-27B joint.
- Laya and Qwen3.7-Flash are among the best calibrated (Laya DL19 0.066) despite weak ranking: calibration and
  ranking quality are separate properties -- worth one sentence.

**Cost (measured).** API cost per 1,000 queries: Jev joint $1.18, Jev isolated $2.29, Qwen3.7-Flash $0.82. Whole grid:
Jev joint $1.62, Jev isolated $3.16, Qwen3.7-Flash $1.13. RankGPT-4 as published: $641 per 1,000 queries (Jin et al.)
-- about 500x Jev joint.

**Latency (measured, not yet comparable).** p50 per query of 100 candidates: Jev joint 0.35 s (API), Jev isolated
1.7 s, Qwen3-Reranker-8B ~3.4 s, Kev-27B joint 8-15 s, Kev-27B isolated ~24 s, monoT5-3B ~0.8 s, Set-Encoder ~0.2 s.
Runs used mixed GPUs (L40, L40S, A100-80GB, H200), so these cannot go into the paper as a comparison. RankZephyr:
2.6-3.1 s on one RTX 4090 (Chen et al.).

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

Asked and answered 2026-10-08 (`docs/intent/paper-focus.md`):

1. Headline: **open matches closed** (Kev-27B vs Jev, both above the trained rerankers).
2. Efficiency: **API cost plus one fixed-GPU latency/energy run** (DL19/20, A100-SXM-80GB, NVML).
3. Listwise baseline: **RankZephyr run by us** (DL19-23; BEIR from the literature); RankGPT-4 literature only.
4. Main table: **joint mode only**; isolated rows in the RQ2 table.

Follow-ups (same day): RQ1 leads with the headline (open vs. the closed Jev); the title stays open, *Calibrated System
One Models as Rerankers*; Kanishka's QPP results get one paragraph; RQ2's explanation (window splits, passage length)
goes into the main text.

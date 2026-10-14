# Paper notes

Reference notes for the paper: where every number from outside our runs comes from, the sanity checks, model facts
and the caveats to state. The plan of record is `docs/roadmap.md`; numbers in the paper come only from the generated
tables (`paper/tables/`, `python -m cdm.paper_tables`).

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
| LLM list prices (OpenRouter) | in git history of this file (cost estimate, not run) | OpenRouter models API (2026-10-08) | `OpenRouterModels` | RankGPT cost estimate, Qwen3.7-Flash price |
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

## 3. Model facts for the setup table

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

## 4. Limitations and caveats to state

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
- RQ2's second question (`answer`) was fixed before its runs but not chosen on development data.

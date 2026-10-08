# Models, candidates and evaluation

The owner's current decisions; `docs/roadmap.md` turns them into the setup.

## One setup, aligned with S1Rank and the literature (2026-10-07)

- **Candidates: BM25 top-100** (Pyserini default) for every dataset, so that published numbers on the same
  candidates can be used instead of recomputing them. For DL21-23 the official top-100 files are exactly this set.
- **Datasets:** DL19-DL23; BEIR TREC-COVID, NFCorpus, SciFact, Touché, DBPedia (the public sets with published
  RankGPT/RankZephyr/monoT5 numbers).
- **Take published numbers where they exist** -- mainly from Abdallah et al. (2025), "How Good are LLM-based
  Rerankers?" -- for RankGPT-4, RankZephyr-7B (not run by us), bge-reranker-v2-m3 and monoT5-3B; compute the rest.
- **System One models:** Jev (closed), Clef-Flash and Clef, Kev in two sizes (4B, 27B), Laya. Every one joint and
  isolated, except Laya (isolated only).
- **Baselines computed:** monoT5-3B, bge-reranker-v2-m3 (where no published number exists), Qwen3-Reranker-8B,
  Set-Encoder-large. Qwen3.7-Flash (an OpenRouter LLM judge at Jev's price) was run but dropped from the paper on
  2026-10-09: the triplets' base models (`docs/intent/paper-focus.md`) are the better-matched control.
- **Terminology:** S1Rank's -- joint pointwise and isolated pointwise; "listwise" only for rerankers that output a
  ranking.

## Access

- **Jev:** TypeSafe API, key in `.env` (`TYPESAFE_AI_API_KEY`). No fixed API budget; runs cost a few dollars, ask
  before anything larger.
- **OpenRouter** (`OPENROUTER_API_KEY` in `.env`): an alternative route for Jev (`JEV_PROVIDER=openrouter`).
- **Open models** (Clef, Kev, Laya, the cross-encoders): Hugging Face, pinned revisions, prefetched to `data/`.

## Calibration

- Unjudged documents count as non-relevant (the BEIR convention), stated in the paper.
- The calibration results (RQ3) come only from the sets with judged non-relevant documents: DL19-DL23.

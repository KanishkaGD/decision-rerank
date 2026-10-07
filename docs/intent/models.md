# System One models

## Jev and Laya access

Decided by the owner on 2026-10-06 (open question 3):

- **Jev: API access exists.** The TypeSafe AI API key is in `.env` as `TYPESAFE_AI_API_KEY`. No API budget was
  given; ask before large runs.
- **Laya: the owner is trying to get access to a Laya API.** Not available yet.
- **Start with Jev**: implement and run Jev first, Laya when its API arrives.

## Laya and CLEF in the setup

Decided by the owner on 2026-10-06:

- **Integrate Laya and Cloudflare's CLEF as well**, alongside Jev. Laya is open weights on Hugging Face
  (`convaiinnovations/laya`, Apache 2.0), so no Laya API is needed; CLEF is `Cloudflare/clef` (27B) and
  `Cloudflare/clef-flash` (9B), Apache 2.0, Jev-compatible request format. CLEF is not in the research plan.
- **One System One reranker** for all three, so every model gets the same state and questions.

## Calibration on sparsely judged sets

Decided by the owner on 2026-10-06:

- Unjudged documents count as non-relevant (the usual BEIR convention), stated in the paper.
- **The main calibration results come only from sets with judged non-relevant documents** (TREC DL, TREC-COVID,
  and others that have them); calibration on sets judged only for relevance is secondary.

## Candidates and systems aligned with Kanishka's study

Decided by the owner on 2026-10-07:

- **Official rerank pools where they exist.** Rerank the track's own pool (TREC DL 2019/2020:
  `msmarco-passagetest{2019,2020}-top1000`, as on Kanishka's branch); build our own BM25 pool only where none exists,
  and avoid that where possible. This may influence which datasets are evaluated. BEIR publishes no pools (its
  convention is Pyserini BM25 flat top-100); TREC DL 2021-2023 have official top-100 pools, but on MS MARCO v2.
- **Kanishka's systems join ours**: monoT5-base, Qwen3-Reranker-0.6B, Kev-0.8B and Qwen3.7-Flash (OpenRouter), next to
  Jev and Laya. Kanishka has one A100; we have H200s through Slurm, so we also run larger sizes (monoT5-3B,
  Qwen3-Reranker-4B/8B, Kev-4B/9B/27B).

## Pointwise vs. listwise (RQ4)

Decided by the owner on 2026-10-07:

- **IR terminology.** "Pointwise" (one query-passage pair per model call) and "listwise" (the whole candidate list
  in view, one score per candidate) replace "isolated" and "joint" in code, system names and paper.
- **RQ4 compares pointwise and listwise for Jev, Clef and Kev** (`<model>-pointwise` / `<model>-listwise`).
- **Set-Encoder joins as a baseline** (webis/set-encoder-base and -large): the listwise counterpart among trained
  cross-encoders.

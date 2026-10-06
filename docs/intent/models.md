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

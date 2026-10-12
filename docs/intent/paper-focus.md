# Paper focus

The owner's current decisions on what the paper claims; `docs/roadmap.md` turns them into the plan. Evidence:
`docs/notes/paper-notes.md`, `docs/notes/decision-rerank-status.md`.

## Narrative (decided 2026-10-09)

1. **Headline: a negative result.** On the same backbone, reranker fine-tuning beats decision fine-tuning (System One
   models such as Kev) at every size where both exist (Qwen3/Qwen3.5, 0.6B-9B), on TREC DL and on the BEIR
   collections without training splits. Decision fine-tuning stays within a few points of the zero-shot base model
   at every size (-3.6 to +3.7 average nDCG@10), while reranker fine-tuning adds 5-32 points.
2. **Why it matters.** System One / decision models are a hype topic and well marketed. That open decision
   fine-tunes (Kev-27B) matched the closed Jev within weeks of its release suggests no technological breakthrough
   behind them, but a new training paradigm (RLCD) with no intuition yet for how well its first authors executed
   it. A controlled comparison on one backbone separates the training objective from the model. The open-vs-closed
   comparison is motivation and context, not a claim of its own.
3. **RQ1: decision vs. reranker fine-tuning** on the triplets (decision 6). Joint vs. isolated scoring is part of
   RQ1, not a separate RQ: the decision fine-tunes are reported in both modes; where joint scoring hurts, it is
   explained by the length of the joint state (tokens in one request) against the length the model was trained on,
   not by window splits.
4. **RQ2: is calibration still of use?** Decision models give calibrated probabilities out of the box; reranker
   fine-tunes do not. RQ2 asks whether that is worth anything once the reranker fine-tune ranks better. It is kept
   as an open question and argued in a discussion, backed by three pieces of evidence (proposals in
   `docs/notes/calibration-proposal.md`):
   - ECE and Brier on the judged pairs of DL19-23 (as computed);
   - **one fixed cutoff** (Part A): keep every passage with p >= 0.5, set precision/recall/F1 and the regret against
     the per-dataset oracle cutoff -- from stored outputs, no new runs;
   - **calibrated after the fact** (Part C): temperature and Platt scaling of the reranker fine-tunes, fitted on
     `msmarco-dev` only, evaluated on DL19-23 and BEIR -- the answer to "just calibrate the reranker" (new runs on
     msmarco-dev).

   Query performance prediction (Kanishka's results) is named in the discussion as a further use; the study stays
   Phase 2.
5. **Our own 27B reranker fine-tune** (revised 2026-10-10; before: "not planned"). No published reranker fine-tune
   of Qwen3.8-27B exists, so we train one with the recipe in `docs/notes/reranker-27b-plan.md` (LoRA on the
   checkpoint Kev-27B and Clef start from; MS MARCO, 100k queries; ranking + yes/no loss; checkpoints chosen on
   msmarco-dev only). First a 9B validation run with the same recipe (`qwen3.5-9b-reranker`) against Prism-9B. The
   27B run (`qwen3.8-27b-reranker`) starts only if the validation succeeds, by the criterion fixed before its results:
   average nDCG@10 over the ten datasets within 2 points of Prism-9B (>= 57.0) and above every 9B decision fine-tune
   (Clef-Flash isolated, 54.7). Otherwise the paper keeps the claim at 0.6B-9B and says that no 27B reranker
   fine-tune exists.
6. **Triplets, full scale line.** On one backbone: the base model (zero-shot), its decision fine-tune and its
   reranker fine-tune -- Qwen3-0.6B, Qwen3-8B, Qwen3.5-0.8B, Qwen3.5-4B, Qwen3.5-9B and Qwen3.8-27B. Table in
   `docs/roadmap.md`, "Triplets".
7. **Base row = the post-trained release** asked S1Rank's question zero-shot in its chat format (thinking off),
   P(yes) from the first answer token. Robustness check for Qwen3-8B: the pretrained -Base with a completion prompt.
8. **Title: a question**, naming the setup and leaving the answer to the abstract. Working title: *Do System One
   Models Make Better Rerankers? Decision vs. Reranker Fine-Tuning on One Backbone*.

## Tables and baselines (decided 2026-10-08, still current)

9. **Main table = the triplets** (decision fine-tunes in both modes). A second, context table holds the other
   systems on the same candidates, System One models in joint mode (Laya isolated): Jev, Clef, the trained rerankers
   and the listwise LLM rerankers.
10. **Listwise baseline: RankZephyr run by us** on DL19-23 with rank_llm's prompt and sliding window, with
    transformers like our other GPU models (`src/cdm/rerankers/listwise.py`); its DL19/20 runs check our setup
    against the published 74.2 / 70.2. BEIR cells stay from the literature (Abdallah et al.). RankGPT-4 stays
    literature-only.
11. **Efficiency: API cost plus one fixed-GPU run.** All local models are timed on DL19/20 on one GPU type
    (A100-SXM-80GB) with an energy reading per run (NVML); the API model (Jev) gets cost and wall-clock time only.
12. **Qwen3.7-Flash is not in the paper** (2026-10-09); its runs stay in `results/results.csv`.

## ECIR short paper (decided 2026-10-12)

13. **Venue: ECIR 2027 short paper**, deadline 2026-10-12 23:59 AoE (2026-10-13 13:59 CEST) (`docs/intent/venue.md`); written with what exists.
14. **RQ2 is a negative result too.** Overall answer to "are the hyped decision models of use to the IR community?":
    no -- not as rerankers (RQ1), and not for their calibration either (RQ2), because reranker fine-tunes are
    partly better calibrated (our Qwen3.5-9B reranker fine-tune: mean ECE 0.111 on DL19-23, below every decision
    model; Prism-4B 0.218 vs Kev-4B 0.216; Qwen3-Reranker-8B 0.257 vs Kev-8B 0.243) and keep the better set at a
    fixed cutoff.
15. **Our 27B reranker fine-tune stays out of the ECIR paper** (supersedes 5 for this paper). The run continues and
    goes into the arXiv version at the end of the week, next to the 9B fine-tune. **Our 9B fine-tune (`qwen3.5-9b-reranker`) is in**: a second reranker fine-tune of Qwen3.5-9B next to
    Prism-9B, trained by us without any decision data (Avg 59.4 vs Prism 59.0, Clef-Flash 54.7).
16. **Tables of the ECIR paper** (decided 2026-10-12, for the 6-page limit): (1) the full-width triplet table with
    every dataset and our 9B fine-tune as `Ours-9B` (`triplets.tex`); (2) one RQ2 table merging ECE (native,
    temperature, Platt), set F1 at 0.5 (native, Platt), regret and set size, one row per triplet system (decision
    fine-tunes in their better-calibrated mode) plus Jev, monoT5, bge and the Set-Encoder (`calibration.tex`);
    (3) a small context table with the DL/BEIR/Avg/ZS means of the systems outside the triplets
    (`context-short.tex`). The full context, cutoff and post-hoc tables stay generated for the arXiv version.
17. **RQ2 wording** (decided 2026-10-12): "Are decision models at least better calibrated than reranker fine-tunes of
    the same backbone?" -- narrower than before: about ECE and the fixed cutoff, not about every use of the
    probabilities.

# Paper focus

Decided by the owner on 2026-10-08, after the grid was complete (evidence: `docs/notes/paper-notes.md`).

1. **Headline: open matches closed.** An open System One model (Kev-27B) matches the closed Jev and beats every
   trained reranker on average; model size matters more than open vs. closed. RQ2 (joint vs. isolated) and RQ3
   (calibration) support this message rather than standing as separate stories.
2. **Efficiency: API cost plus one fixed-GPU run.** The main table keeps the API cost column. In addition, all local
   models are timed on DL19/20 on one GPU type (A100-SXM-80GB) with an energy reading per run (NVML), so latency and
   energy are comparable across systems. API models (Jev, Qwen3.7-Flash) get cost and wall-clock time only.
3. **Listwise baseline: run RankZephyr ourselves.** Open 7B model with rank_llm's prompt and sliding window, run
   with transformers like our other GPU models (`src/cdm/rerankers/listwise.py`; no vLLM, so its latency is
   comparable to theirs within our setup). It fills the DL21-23 cells (no published BM25 numbers), gives latency and
   energy on our hardware, and its DL19/20 runs check our setup against the published 74.2 / 70.2. BEIR cells stay
   from the literature (Abdallah et al.), as the rule "don't recompute what is published on the same candidates"
   requires. RankGPT-4 stays literature-only.
4. **Main table: joint only.** One row per System One model (its joint mode; Laya isolated); the isolated rows go
   into the RQ2 table of joint-minus-isolated differences.

Follow-up decisions, 2026-10-08:

5. **RQ1 wording** leads with the headline: "Do open System One models rerank BM25 candidates as well as the closed
   Jev and strong trained rerankers?" (`paper/rq-macros.tex`).
6. **Title stays open:** *Calibrated System One Models as Rerankers*. A title naming the headline ("Open System One
   Models Rerank as Well as Closed Ones") is too narrow.
7. **Kanishka's QPP results** get one paragraph in the paper (probabilities as a control signal); the full study
   stays Phase 2.
8. **RQ2's explanation goes into the main text:** where joint scoring hurts, related to how often a joint request has
   to be split (window_splits per dataset, passage length).

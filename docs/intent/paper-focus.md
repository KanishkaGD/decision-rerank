# Paper focus

Decided by the owner on 2026-10-08, after the grid was complete (evidence: `docs/notes/paper-notes.md`).

1. **Headline: open matches closed.** An open System One model (Kev-27B) matches the closed Jev and beats every
   trained reranker on average; model size matters more than open vs. closed. RQ2 (joint vs. isolated) and RQ3
   (calibration) support this message rather than standing as separate stories.
2. **Efficiency: API cost plus one fixed-GPU run.** The main table keeps the API cost column. In addition, all local
   models are timed on DL19/20 on one GPU type (A100-SXM-80GB) with an energy reading per run (NVML), so latency and
   energy are comparable across systems. API models (Jev, Qwen3.7-Flash) get cost and wall-clock time only.
3. **Listwise baseline: run RankZephyr ourselves.** Open 7B model through rank_llm/vLLM on our GPUs. It fills the
   DL21-23 cells (no published BM25 numbers), gives latency and energy on our hardware, and its DL19/20 runs check
   our setup against the published 74.2 / 70.2. BEIR cells stay from the literature (Abdallah et al.), as the rule
   "don't recompute what is published on the same candidates" requires. RankGPT-4 stays literature-only.
4. **Main table: joint only.** One row per System One model (its joint mode; Laya isolated); the isolated rows go
   into the RQ2 table of joint-minus-isolated differences.

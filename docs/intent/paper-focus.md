# Paper focus

The owner's current decisions on what the paper claims; `docs/roadmap.md` turns them into the plan. Evidence:
`docs/notes/paper-notes.md` and `docs/experiment-log.md`.

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
   fine-tunes need not. RQ2 asks whether that is worth anything once the reranker fine-tune ranks better (wording
   and answer: 15, 16). Evidence: ECE and Brier on the judged pairs of DL19-23; one fixed cutoff (keep every passage
   with p >= 0.5; set F1 and the regret against the per-dataset oracle cutoff); the count test; the reranker
   fine-tunes calibrated after the fact (temperature and Platt scaling fitted on `msmarco-dev` only). Query
   performance prediction is named in the discussion as a further use, not studied.
5. **Our own reranker fine-tunes** (2026-10-10). No published reranker fine-tune of Qwen3.8-27B exists, so we train
   one with the recipe in `docs/notes/reranker-27b-plan.md` (LoRA on the checkpoint Kev-27B and Clef start from; MS
   MARCO, 100k queries; ranking + yes/no loss; checkpoints chosen on msmarco-dev only). The 9B validation run with
   the same recipe (`qwen3.5-9b-reranker`) passed the criterion fixed before its results (Avg within 2 points of
   Prism-9B and above every 9B decision fine-tune), so the 27B run (`qwen3.8-27b-reranker`) was trained.
6. **Triplets, full scale line.** On one backbone: the base model (zero-shot), its decision fine-tune and its
   reranker fine-tune -- Qwen3-0.6B, Qwen3-8B, Qwen3.5-0.8B, Qwen3.5-4B, Qwen3.5-9B and Qwen3.8-27B. Table in
   `docs/roadmap.md`, "Triplets".
7. **Base row = the post-trained release** asked S1Rank's question zero-shot in its chat format (thinking off),
   P(yes) from the first answer token. Robustness check for Qwen3-8B: the pretrained -Base with a completion prompt.
8. **Title: a question**, leaving the answer to the abstract: *Do System One Models Make Better Rerankers?* (17).

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

## The arXiv paper (decided 2026-10-12/13; ECIR submission 2026-10-13)

13. **Target: the arXiv preprint** (`docs/intent/venue.md`). It starts from the submitted ECIR paper (tag
    `ecir2027-submission`) and adds what was cut for space and our 27B reranker fine-tune.
14. **Our reranker fine-tunes are in**: `qwen3.5-9b-reranker` (*Ours-9B*, a second reranker fine-tune of Qwen3.5-9B
    next to Prism-9B, trained without any decision data; Avg 59.4 vs Prism 59.0, Clef-Flash 54.7) and
    `qwen3.8-27b-reranker` (trained, job 4429626; it fills the empty reranker cell of the 27B triplet).
15. **RQ2 wording**: "Are decision models at least better calibrated than reranker fine-tunes of the same foundation
    model?" -- about ECE, Brier, the fixed cutoff and the count test, not every use of the probabilities.
16. **RQ2 answer** (2026-10-13). Two questions are run on TREC DL 2019-2023: S1Rank's (`s1rank`) and Kanishka's
    ("Does document ... answer the query?", phrasing `answer`; fixed before the runs but not chosen on development
    data -- the owner's override of the CLAUDE.md rule, stated in the paper). The calibration of decision models
    depends on the wording of the question (the base models move as much); it does not make them select better
    passages at a fixed cutoff; the count test shows no benefit specific to decision fine-tuning; our reranker
    fine-tune matches the best decision model of its size. Both answers are mostly negative.
17. **Framing and wording**: title *Do System One Models Make Better Rerankers?*; "foundation model", not "backbone"
    (also in the RQs); plain language without dash asides. Kev and Clef are open alternatives with Jev's kind of
    output, not reproductions of Jev's training (RLCD is unpublished), so "decision fine-tuning" means fine-tuning an
    open model to give Jev's kind of output.
18. **Tables**: rows shaded by training (base grey, decision fine-tune red, reranker fine-tune blue), our
    fine-tunes in italics, Avg/ZS grouped under "Mean", best bold and second underlined. The material cut for ECIR
    comes back: the full calibration table (`calibration-full.tex`: native, temperature and Platt ECE, F1 at the
    fixed cutoff, regret, set size; monoT5, bge and the Set-Encoder), the cutoff and post-hoc tables, the full
    context table, DL20 and DL22 in the triplet table, regret and post-hoc scaling in the setup, a limitations
    paragraph.
19. **Sources**: the paper is written in `paper/` (Michael's draft, decided 2026-10-12); Kanishka revises on Overleaf,
    and his edits are merged into `paper/` by hand.
20. **One question for the whole arXiv paper** (decided 2026-10-13; extends the override in 16 from RQ2 to RQ1).
    Tables 1 and 2 (triplets and context) are recomputed with Kanishka's question (`answer`) on all ten datasets: every
    decision fine-tune in both modes, the base models (`<name>-prompted-answer`), Jev joint and isolated, Laya.
    Reranker fine-tunes and cross-encoders ask no question and keep their runs. S1Rank's question stays in the
    repository and as a comparison (on DL19-23 it gives slightly higher nDCG@10 for most decision models, -0.2 to
    -2.2 points with `answer`); the sanity check of `jev-joint` against S1Rank's published numbers uses S1Rank's
    question. The paper states that the question was fixed before the runs but not chosen on development data.
21. **The arXiv narrative** (owner, 2026-10-14). Every result in the paper uses Kanishka's question (`answer`); S1Rank's
    question appears only in an ablation (Sect. 7, `paper/tables/question-ablation.tex`) that compares the two
    wordings and shows the lack of robustness (ranking hardly moves, calibration and the RQ2 verdict flip). The
    introduction, title, RQ wording and conclusion follow the submitted ECIR paper (`ECIR-submission.pdf`), in a
    slightly more provocative tone; a TikZ teaser figure on page 1 shows the triplet design and the questions. The
    material cut for ECIR is back (all ten collections in Table 1, full context table, post-hoc scaling and the
    fixed cutoff in the calibration table, a discussion and limitations section). Layout: two-column ACM
    (`docs/intent/venue.md`).
22. **Title, authors, one narrative** (owner, 2026-10-14). Title *System One Models Are Not Better Rerankers*, no
    subtitle (replaces 8 and 17's question title). Authors: Michael Dinzinger (University of Passau), Kanishka (IT:U
    Linz), Michael Granitzer (University of Passau and IT:U Linz). The narrative is fixed in
    `docs/notes/narrative.md`: three punchlines (reranker fine-tuning beats decision fine-tuning; up to 9B decision
    fine-tuning adds nothing over the base model; the calibration advantage comes from the question, not from the
    model), no hedging chains in abstract, introduction and conclusion. The teaser figure shows how System One models
    rerank and what we test, without the RQ boxes.

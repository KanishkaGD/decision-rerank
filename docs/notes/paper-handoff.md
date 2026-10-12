# Handoff: writing the ECIR short paper

Written 2026-10-12 at the end of the experiment session, for a **new session that writes the paper**. Read this
file first, then the files in "Read next". Do not redo experiments. The numbers exist; the job is writing, tables
and at most minor analyses.

## 1. Facts of the submission

- **Venue: ECIR 2027 short paper.** Deadline **2026-10-12, 23:59 AoE (2026-10-13, 13:59 CEST)**; abstract registered.
  Springer LNCS: the sources in `paper/` are already LNCS (`llncs.cls`, `splncs04.bst`).
- **Official call** (https://www.ecir2027.co.uk/call-for-short-papers, read 2026-10-12):
  - **Up to 6 pages plus references.** Appendices count toward the 6 pages and go before the references.
  - Springer LNCS templates.
  - **Double-anonymous**: no author names, no repository URL ("anonymised link"), no self-identifying citations.
  - A prior arXiv version is allowed but must not be cited.
  - EasyChair, conference id ecir27.
  - Deadline 12 October 2026, 23:59 AoE (= 2026-10-13, 13:59 CEST). The abstract is registered.
- Build: `cd paper && make` (latexmk), `make check` (checks), `make tables` (regenerates `paper/tables/*.tex` from
  `results/*.csv` via `python -m cdm.paper_tables`; standard library only, fine on the login node).
- The login node may run `make`, `pytest` and `cdm.paper_tables`. **No model runs on the login node**; anything
  heavier goes through `slurm/exec.sh` (CLAUDE.md, "Where things run").

## 2. The story (owner's decisions, `docs/intent/paper-focus.md`, items 1-15)

**One sentence:** On the same backbone, decision fine-tuning (the training behind the hyped System One models, e.g.
Kev and Clef) makes a worse reranker than plain reranker fine-tuning. Its one distinctive feature, calibrated
probabilities, doesn't help either: reranker fine-tunes are calibrated as well or better. So the answer to "are
decision models of use to the IR community?" is no, for both reranking and calibration.

**Title (working, a question):** *Do System One Models Make Better Rerankers? Decision vs. Reranker Fine-Tuning on
One Backbone* (`paper/main.tex`).

**RQs** (`paper/rq-macros.tex`; the wording is only there). RQ2's wording probably needs a small edit for the
negative framing, e.g. "...are the calibrated probabilities of decision models still an advantage?". The owner
decides; propose it.
- **RQ1:** On the same backbone, does decision fine-tuning make a better reranker than reranker fine-tuning or no
  fine-tuning?
- **RQ2:** Where reranker fine-tuning ranks better, are the calibrated probabilities of decision models still of use?

**Motivation (intro):**
- System One / decision models (Jev; open: Clef, Kev, Laya) are hyped and well marketed. Open decision fine-tunes
  matched the closed Jev within weeks (Kev-27B 61.1 vs Jev 60.5 average nDCG@10 in our grid).
- That points to a training paradigm, decision fine-tuning with RLCD, not a breakthrough. The fair test is a
  controlled comparison on one backbone against the established alternative, reranker fine-tuning.
- `\FLAG`s: a citation for RLCD, and the release dates of Jev, Clef and Kev.

**What is IN:**
- Triplets on five backbones: base model zero-shot / decision fine-tune (joint and isolated) / reranker fine-tune
  (Qwen3-0.6B, Qwen3-8B, Qwen3.5-0.8B, 4B, 9B). Qwen3.8-27B appears with base + decision fine-tunes only.
- **Our own Qwen3.5-9B reranker fine-tune** (`qwen3.5-9b-reranker`), as a second reranker fine-tune of Qwen3.5-9B
  next to Prism-9B.
- The context systems (Jev, Laya, Clef, monoT5, bge, Qwen3-Reranker, Set-Encoder, RankZephyr, RankGPT-4 from the
  literature).
- RQ2: ECE, the fixed-cutoff analysis (Part A), and post-hoc scaling (Part C).

**What is OUT:**
- **Our 27B reranker fine-tune** (training job 4429626 is running). It goes into the arXiv version at the end of the
  week, not into the ECIR submission. Do not wait for it.
- Qwen3.7-Flash.
- Query performance prediction, except one sentence as future work.
- The efficiency table is optional, space permitting. If used, note that Kev's inference is 4.5–7× slower than the
  same-size reranker fine-tune, partly because of Kev's own inference code.

## 3. Key numbers (all from `results/results.csv`; in the paper only via `paper/tables/*.tex`)

nDCG@10 × 100, BM25 top-100. Avg = mean over the ten datasets. ZS = TREC-COVID, Touché, DBPedia (no training
splits).

| Backbone | Base | Best decision fine-tune (mode) | Reranker fine-tune | Avg base / decision / reranker | ZS |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | prompted | Kev-0.6B (isolated) | Qwen3-Reranker-0.6B | 23.8 / 27.0 / 56.1 | 25.4 / 30.9 / 54.9 |
| Qwen3-8B | prompted | Kev-8B (isolated) | Qwen3-Reranker-8B | 53.3 / 52.7 / 59.4 | 47.1 / 49.1 / 57.8 |
| Qwen3.5-0.8B | prompted | Kev-0.8B (isolated) | Prism-0.8B | 37.6 / 36.0 / 57.1 | 37.7 / 40.7 / 53.8 |
| Qwen3.5-4B | prompted | Kev-4B (isolated) | Prism-4B | 52.7 / 51.3 / 58.7 | 47.1 / 48.4 / 54.2 |
| Qwen3.5-9B | prompted | Clef-Flash (isolated) | Prism-9B; **ours** | 53.7 / 54.7 / 59.0; **59.4** | 48.2 / 49.0 / 53.7; **56.1** |
| Qwen3.8-27B | prompted | Kev-27B (joint) | -- | 57.4 / 61.1 / -- | 52.7 / 57.9 / -- |

**RQ1, joint vs. isolated:**
- Joint − isolated, Avg: Kev-27B +1.3, Jev +1.0, Kev-9B +0.1, Clef −0.3, Kev-4B −1.5, Clef-Flash −2.0, Kev-0.8B
  −10.7, Kev-0.6B −13.7, Kev-8B −13.9.
- Explanation (owner's decision 8): the length of the joint state against the length the model was trained on. The
  small Kev models were trained on states of ≤ 7,552 tokens (model cards), Kev-27B on ≤ 32,768. Joint DL states are
  about 13k tokens, BEIR states 40k+.
- Kev-0.6B joint on DL19: AUC 0.49 (no discrimination), and scores rise with position in the state.
- Kev-8B's training length is unknown; check its card or drop the claim for it.

**RQ2, calibration** (mean ECE over DL19–23, judged pairs, 15 bins; lower is better):

| Backbone | Base | Decision fine-tune (better mode) | Reranker fine-tune |
| --- | --- | --- | --- |
| Qwen3-0.6B | 0.750 | 0.579 | 0.511 |
| Qwen3-8B | 0.383 | 0.243 | 0.257 |
| Qwen3.5-0.8B | 0.716 | 0.357 | 0.243 |
| Qwen3.5-4B | 0.296 | 0.216 | 0.218 |
| Qwen3.5-9B | 0.283 | 0.140 (Clef-Flash joint) | Prism 0.242; **ours 0.111** |
| Qwen3.8-27B | 0.166 | 0.147 (Clef joint), 0.156 (Kev joint) | -- |

The negative-result reading:
- The reranker fine-tune is as well or better calibrated at 0.6B, 0.8B and 4B, and our 9B fine-tune is the best
  calibrated system overall.
- At a fixed cutoff (p ≥ 0.5; `paper/tables/cutoff.tex`), the reranker fine-tune keeps the better set on every
  backbone (e.g. F1 at 0.5: Prism-4B 51.0 vs Kev-4B 41.0; Prism-9B 50.7 vs Clef-Flash isolated 49.2).
- Low regret of a fixed cutoff is not unique to decision models: Prism 2.8–4.0, bge 3.8, base Qwen3.8-27B 2.4.
- Post-hoc scaling of the trained rerankers (`paper/tables/posthoc.tex`): temperature lowers ECE but stays ≥ 0.20.
  Platt collapses the cutoff because msmarco-dev has only 0.7% relevant candidates. This supports "rerankers need
  labelled data to be recalibrated", but **our 9B fine-tune shows that a reranker can be trained calibrated** (BCE
  on yes/no). That is the stronger argument for the negative RQ2.

**Sanity checks** (setup section):
- BM25 = Pyserini 2CR on all ten datasets.
- monoT5-3B DL19/20: 71.8 / 68.9 vs published 71.83 / 68.89.
- RankZephyr DL19/20: 74.3 / 70.0 vs 74.2 / 70.2.
- Jev joint vs S1Rank: within 0.5, except COVID (85.0 vs 86.5).

## 4. Work list for the writing session (in order)

1. **Check `make` builds** and the page budget: 6 pages plus references, LNCS, anonymous.
2. **Small table and analysis updates** (minutes, no GPU):
   - Add `qwen3.5-9b-reranker` ("ours") as a second reranker fine-tune row in the Qwen3.5-9B group of
     `TRIPLETS` (`src/cdm/paper_tables.py`).
   - Run `cdm cutoff` for it so the cutoff table has its row: `slurm/exec.sh -- cdm cutoff --systems
     qwen3.5-9b-reranker` (CPU, minutes).
   - Then `make tables`.
   - Decide whether the 27B group stays in the triplet table (base + decision only) or moves to the context table.
3. **Paired tests** (optional, if time allows; per-query metrics are stored in `outputs/.../perquery.csv`): reranker
   fine-tune vs. the better decision mode per backbone, and joint vs. isolated, with Holm correction. Without them,
   say "differences of several points on all five backbones" and avoid "significant".
4. **Write**, section files in `paper/sections/` (every one is still `\TODO`/`\PENDING`):
   - `01_introduction.tex`: hype, the research question, the controlled triplet design, two negative findings,
     contributions.
   - `02_related_work.tex`: cross-encoders and LLM rerankers, listwise; System One reranking (S1Rank, jev-rerank,
     blog posts: cite as non-peer-reviewed); calibration in IR.
   - `03_setup.tex`: triplets, candidates, question, readouts, metrics, our 9B recipe in two sentences, sanity checks.
   - `04_reranking.tex`: RQ1, the triplet table, joint vs. isolated and state length, the context table.
   - `05_calibration.tex`: RQ2, the ECE table, cutoff, post-hoc, our calibrated 9B fine-tune.
   - `06_conclusion.tex`: the answers; limitations (no 27B reranker fine-tune; different training data — reranker
     fine-tunes saw MS MARCO, Kev did not, hence the zero-shot subset; literature numbers not recomputed;
     calibration on DL only); the anonymised code link.
   - Abstract in `main.tex`.
5. **Resolve every `\TODO`, `\PENDING`, `\FLAG`** (`grep -rn '\\TODO\|\\PENDING\|\\FLAG' paper/`). Check the
   bibliography entries marked "to verify" in `paper/sources.bib` (FIRST venue, Qwen3-Embedding authors). Cite
   Prism (Zhang 2026) and Kev (model cards) for the triplets.
6. `make check`, page count, anonymity check, PDF to the owner.

## 5. Read next

- `docs/notes/decision-rerank-status.md`: the latest project report (tables per RQ, claims with their status).
- `docs/notes/paper-notes.md`: the source register (where every outside number comes from), limitations.
- `docs/notes/reranker-27b-plan.md`, sections 1-4: our fine-tuning recipe (for the 9B description in the setup).
- `docs/experiment-log.md`: the newest entries (2026-10-09 to 2026-10-11).
- `paper/tables/*.tex`: what the tables already contain.
- `.claude/skills/project-report/scripts/facts.py`: prints every current number from `results/` in one go.

## 6. Rules that still apply (CLAUDE.md)

- Numbers in the paper come only from generated tables (`paper/tables/`), never typed by hand. In the text, quote
  numbers that are in those tables.
- Literature numbers are labelled as such.
- Do not edit `results/results.csv` by hand.
- Commit on `dev-michael` with the attribution lines from the session's system reminder.
- Another session's jobs may still write `results/results.csv` (the 27B run does not write results).

## 7. Suggested first message for the new session

> Read `docs/notes/paper-handoff.md` and follow it: we write the ECIR 2027 short paper today (deadline 2026-10-12
> 23:59 AoE = 2026-10-13 13:59 CEST). Start with work-list items 1–2, then draft the sections in order. Ask me before changing the RQ wording or
> the title.

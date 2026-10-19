# Roadmap

The plan of record: what the paper claims, how it is tested, and the order of the work. Decisions behind it are in
`docs/intent/`, what happened along the way in `docs/experiment-log.md`. Keep this file current -- it describes the
plan as it is, not how it got here (the `straighten-branch` skill does that sweep).

**Title:** *Do System One Models Make Better Rerankers?* (`docs/intent/paper-focus.md`).
**Target:** the arXiv preprint, end of the week of 2026-10-12 (`docs/intent/venue.md`): the submitted ECIR 2027 short
paper (tag `ecir2027-submission`) plus what was cut for space (our 27B reranker fine-tune is not in the paper; owner, 2026-10-19). Handoff for the next
session: `docs/notes/paper-handoff.md`.

## Contribution

System One (decision) models answer a yes/no question with a probability, and S1Rank showed that the closed Jev
reranks BM25 candidates as well as strong cross-encoders. Open decision fine-tunes (Kev, Clef) matched Jev within
weeks, which points to a training paradigm (RLCD) rather than a technological breakthrough. We hold the backbone
fixed and compare three training objectives on it -- none (the base model asked zero-shot), decision fine-tuning and
reranker fine-tuning -- on ten datasets, and ask whether the one property decision models have and reranker
fine-tunes lack, calibrated probabilities, is still of use.

**Headline** (`docs/intent/paper-focus.md`): a negative result -- reranker fine-tuning beats decision fine-tuning at
every size where both exist (0.6B-9B). Decision fine-tuning stays within a few points of the zero-shot base model at
every size (-3.6 to +3.7), reranker fine-tuning adds 5-32 points. At 27B no published reranker fine-tune exists;
ours (`qwen3.8-27b-reranker`) fills that cell.

## Research questions (paper)

| RQ | Question | Evidence |
| --- | --- | --- |
| RQ1 | On the same foundation model, does decision fine-tuning make a better reranker than reranker fine-tuning or no fine-tuning? | nDCG@10 of the triplets on all ten datasets (average and zero-shot subset), decision fine-tunes joint and isolated with paired tests; joint vs. isolated against the state length the model was trained on; the other systems as context |
| RQ2 | Are decision models at least better calibrated than reranker fine-tunes of the same foundation model? | ECE and Brier on DL19-23 with two questions (S1Rank's and `answer`), incl. our reranker fine-tunes; one fixed cutoff (p >= 0.5): set F1 and regret against the oracle cutoff; the count test; reranker fine-tunes calibrated after the fact on msmarco-dev |

Terminology (S1Rank's): every System One run is **pointwise** -- one Noul probability per candidate, ranked by it.
**Joint** = all candidates of a query in one state; **isolated** = one candidate per request. "Listwise" is reserved
for rerankers that output a ranking (RankGPT, RankZephyr).

## Setup

**Candidates.** BM25 top-100 of Pyserini's default BM25 (k1 0.9, b 0.4; BEIR flat indexes) for every dataset -- the
candidate set S1Rank, RankGPT, RankZephyr and Abdallah et al. (2025) rerank, so their published numbers belong in the
same table. For DL21-23 the official top-100 files are exactly this set. Our BM25 must reproduce Pyserini's 2CR
scores (`results/literature.csv`, role `reference`).

| Dataset | Queries | Corpus | Calibration (judged non-relevant passages) |
| --- | ---: | --- | :---: |
| DL19, DL20 | 43, 54 | MS MARCO v1 passage | yes |
| DL21, DL22, DL23 | 53, 76, 82 | MS MARCO v2 passage (official top-100 files) | yes |
| TREC-COVID, NFCorpus, SciFact, Touché, DBPedia | 50, 323, 300, 49, 400 | BEIR | no |

**Systems.** Computed by us unless a published number on the same candidates exists (`results/literature.csv`,
role `baseline`; `from_literature` in `slurm/config.sh` makes `slurm/submit.sh` skip those cells).

| System | Kind | DL19/20 | DL21-23 | BEIR |
| --- | --- | --- | --- | --- |
| RankGPT-4 | listwise LLM reranker | literature | -- | literature |
| RankZephyr-7B | listwise LLM reranker (run by us) | computed (check against literature) | computed | literature |
| bge-reranker-v2-m3 | cross-encoder | literature | computed | literature |
| monoT5-3B | cross-encoder (seq2seq) | computed (check against literature; calibration) | computed | literature |
| Qwen3-Reranker-8B | pointwise LLM reranker | computed | computed | computed |
| Set-Encoder-large | cross-encoder, scores candidates jointly | literature | computed | computed |
| Jev, Clef-Flash, Clef, Kev-4B, Kev-27B | System One, joint and isolated | computed | computed | computed |
| Laya | System One, isolated only (context too short) | computed | computed | computed |

**Triplets** (`docs/intent/paper-focus.md`, decided 2026-10-09): one backbone, three training objectives -- none
(the post-trained release asked S1Rank's question zero-shot, `*-prompted`), decision fine-tuning (Kev, Clef; joint
and isolated) and reranker fine-tuning (Qwen3-Reranker, Prism). All computed on all ten datasets.

| Backbone | Base model (zero-shot) | Decision fine-tune | Reranker fine-tune |
| --- | --- | --- | --- |
| Qwen3-0.6B | `qwen3-0.6b-prompted` | Kev-0.6B (on -Base) | Qwen3-Reranker-0.6B (on -Base) |
| Qwen3-8B | `qwen3-8b-prompted`; `qwen3-8b-base-prompted` (-Base, completion prompt: robustness check) | Kev-8B (on -Base) | Qwen3-Reranker-8B (on -Base) |
| Qwen3.5-0.8B | `qwen3.5-0.8b-prompted` | Kev-0.8B (on -Base) | Prism-0.8B (checkpoint not stated) |
| Qwen3.5-4B | `qwen3.5-4b-prompted` | Kev-4B (on -Base) | Prism-4B |
| Qwen3.5-9B | `qwen3.5-9b-prompted` | Kev-9B (on -Base), Clef-Flash | Prism-9B; ours: `qwen3.5-9b-reranker` |
| Qwen3.8-27B | `qwen3.8-27b-prompted` | Kev-27B, Clef (on the post-trained release) | ours: `qwen3.8-27b-reranker` (none published) |

Training data differs between the fine-tunes: Qwen3-Reranker and Prism are trained on MS MARCO among much else
(in-domain for DL19-23); Kev's public sources contain no MS MARCO or BEIR. The paper reports the BEIR collections
without training splits (TREC-COVID, Touché, DBPedia) as the zero-shot subset next to the averages.

**Question.** Kanishka's question (phrasing `answer`: ``Does document `{doc}` answer the `query`?``) for all
System One and base models in both RQs (`docs/intent/paper-focus.md` 16, 20); fixed before the runs, not chosen on
development data (stated in the paper). S1Rank's question (`s1rank`) stays as the comparison and for the sanity check
against S1Rank's published numbers.

**Metrics.** nDCG@10 everywhere; ECE (15 bins) and Brier on judged pairs of DL19-23, relevant from grade 2; API
cost. Latency and energy from one efficiency run: every local model on DL19/20 on one A100-SXM-80GB, energy from
NVML per run; API models get cost and wall-clock time only.

**RQ2 evidence.** *One fixed cutoff:* per query keep the
candidates with p >= 0.5 (0.3 and 0.7 as a sensitivity check); set precision, recall and F1 against the judgments,
mean set size, share of empty sets; regret = F1 of the per-dataset oracle cutoff minus F1 at 0.5; baselines fixed
top-k (5, 10, 20) and the reranker fine-tunes' native probabilities. DL19-23 only (judged non-relevant passages).
From stored outputs, no new runs. *Calibrated after the fact:* temperature and Platt scaling of each reranker
fine-tune's logit, fitted on the first 1,000 msmarco-dev queries x BM25 top-100 with every candidate labelled
(msmarco-dev judges only relevant passages, so unjudged = non-relevant), evaluated with ECE, Brier and the cutoff F1
on DL19-23 and with ECE on BEIR (unjudged = non-relevant, as a shift check). Systems:
Qwen3-Reranker-0.6B/8B, Prism-0.8B/4B/9B, monoT5-3B, bge-reranker-v2-m3, Set-Encoder-large. *Count test:* per query
the sum of the probabilities over the judged passages against the number of relevant ones (mean absolute error;
baseline: always the median count).

**Main table.** The triplets: per backbone the base model, the decision fine-tunes (joint and isolated) and the
reranker fine-tune; average over ten datasets and over the zero-shot subset. **Context table:** the other systems on
the same candidates, System One models in joint mode (Laya isolated).

**Significance.** Paired t-test over queries, from the stored per-query metrics (no extra runs), Holm-corrected
across datasets: reranker fine-tune vs. the better decision mode per backbone, and joint vs. isolated per decision
fine-tune. Literature numbers are not tested (no per-query
runs); the paper says so.

**Sanity checks.** BM25 = Pyserini 2CR on every dataset; monoT5-3B on DL19/20 within about one point of the published
71.83 / 68.89; our `jev-joint` against S1Rank's 73.7 / 71.7 (same model, question and candidates).

## Order of work

**Done** (details in `docs/experiment-log.md`): candidate sets and sanity checks; the full grid and the triplet runs
(DL19-23 and BEIR); RankZephyr and the efficiency run; the RQ1 and RQ2 analyses (paired tests, cutoff, post-hoc
scaling, count test, the `answer` question); our 9B reranker fine-tune (scored, in the paper) and the training of the
27B one (job 4429626, all checkpoints in `checkpoints/qwen3.8-27b-reranker/`); the ECIR short paper, submitted
2026-10-13 (tag `ecir2027-submission`).

**The arXiv paper** (end of the week of 2026-10-12). Done by 2026-10-15 (`docs/experiment-log.md`): our 27B reranker
fine-tune scored (left out of the paper on 2026-10-19, as in the ECIR version); what was cut for ECIR restored; Tables 1-3 with the `answer` question, S1Rank's
question in the ablation table; the text along `docs/notes/narrative.md` (title *System One Models Are Not Better
Rerankers*, two-column ACM layout, teaser figure); authors and e-mails; Kanishka's edits from the submitted ECIR PDF.
Left:

1. **Acknowledgements / disclosure** in `paper/main.tex`. (The repository link is in: the full repository,
   https://github.com/KanishkaGD/decision-rerank; `release/` and the orphan branch `release_20261012` still hold the
   ECIR state.)
2. **Overleaf edits after the ECIR submission**, if any (needs the Overleaf source); `make check`; final read; post
   to arXiv.
3. **Merge `dev-michael` into `master`** (merge `master` in first: 10 commits by Kanishka; `pytest`).

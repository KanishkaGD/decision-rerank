# RQ2 calibration: proposal and research brief

Status: **partly decided (2026-10-09).** Calibration is now RQ2 of the paper, behind the negative result of RQ1
(reranker fine-tuning beats decision fine-tuning; `docs/intent/paper-focus.md`, decision 4). RQ2 stays an open
question that the paper argues in a discussion, backed by **Part A** (one fixed cutoff) and **Part C** (calibrated
after the fact); Part B (QPP) is named in the discussion only and stays Phase 2. **Part A is implemented**
(`cdm cutoff`, `src/cdm/cutoff.py`, `paper/tables/cutoff.tex`; first results in `docs/experiment-log.md`,
2026-10-09). The rest of this brief -- metrics,
related work, threats -- is still to be settled by the investigation.

## 0. Read first

| File | Why |
| --- | --- |
| `CLAUDE.md` | conventions; where things run (no computation on the login node); experiment rules (never tune on test splits) |
| `docs/roadmap.md` | the plan: RQs, setup, systems, triplets |
| `docs/intent/paper-focus.md` | the owner's decisions (negative-result headline, RQ2 scope, triplets, ...) |
| `docs/notes/paper-notes.md` | source register, findings so far, limitations |
| `docs/notes/decision-rerank-status.md` | latest project report: tables per RQ |
| `src/cdm/evaluation.py` | `calibration()` (ECE with 15 equal-width bins, Brier), `judged_pairs()` |
| `src/cdm/cli/rerank.py` | what a run stores |

## 1. The problem with the calibration RQ as it stood

Wording until 2026-10-09: *"Are the relevance probabilities of System One models calibrated, and does joint
scoring change that?"*. The evidence is ECE and Brier on the judged pairs of DL19-23.

A reviewer's first question will be: why does a reranker care? Ranking uses only the order of the scores. Any
monotone transformation (scaling the logits, applying a sigmoid, temperature) leaves nDCG unchanged and moves ECE
anywhere. Calibration matters only when **the value of the probability is used**. So RQ2 needs to show uses in
which calibrated probabilities do something that uncalibrated scores cannot do without extra fitting.

What we have so far, as context (snapshot 2026-10-08; regenerate from `results/results.csv`):
- Joint scoring lowers ECE for four of the five System One models. Kev-27B goes from 0.262 (isolated) to 0.156
  (joint), mean over DL19-23; Kev-4B is the exception.
- Trained rerankers' native probabilities are badly calibrated on DL21-23: monoT5-3B 0.36-0.42, Qwen3-Reranker-8B
  0.30-0.37, Set-Encoder 0.47-0.58 (a sigmoid of a logit never meant as a probability).
- Laya is the best calibrated model (mean 0.131) and the weakest ranker: calibration and ranking quality are
  separate properties.

## 2. Proposed RQ2

Proposed here: *"Are the relevance probabilities of System One models calibrated out of the box, and can they be
used without tuning?"* Adopted wording (`paper/rq-macros.tex`): *"Where reranker fine-tuning ranks better, are the
calibrated probabilities of decision models still of use?"*

The joint-vs-isolated effect on ECE then becomes one finding within RQ2 instead of all of it. The triplets
(`docs/roadmap.md`) add a controlled axis on the same backbone: base model log-probabilities, decision fine-tuning
(Kev and Clef are trained and temperature-calibrated for decisions), and reranker fine-tuning (trained for ranking).

## 3. Three candidate parts

### Part A: one cutoff across collections (proposed core)

**Use case.** In a RAG pipeline the reranker decides which passages go to the generator. A fixed top-k passes too
many passages for narrow queries and too few for broad ones. With calibrated probabilities, the natural rule is
"keep every passage with p ≥ 0.5". It needs no tuning, adapts its depth per query, and could work unchanged across
collections.

**Experiment.**
- For every query, keep the candidates with p ≥ τ, where τ is fixed in advance (0.5; maybe also 0.3 and 0.7 as a
  sensitivity check).
- Score the kept set against the judgments: precision, recall and F1 of the set, the mean set size, and the share
  of queries with an empty set.
- Oracle: the best τ per dataset (chosen afterwards on that dataset) and its F1. The **calibration regret** is the
  oracle F1 minus the F1 at τ = 0.5. Report also how far the oracle τ moves between datasets.
- Baselines: fixed top-k (k = 5, 10, 20) with the same set metrics; τ = 0.5 on the trained rerankers' native
  probabilities; τ fitted on msmarco-dev (connects to Part C).
- Datasets: DL19-23 (judged non-relevant passages exist). BEIR only with unjudged = non-relevant, which biases
  precision downward; report it as a secondary result or not at all.
- Systems: every System One model (joint and isolated), the trained rerankers with their native probabilities, and
  the triplets' base models (P(yes) zero-shot).

**Hypothesis to test.** Calibrated models have a small regret, and their oracle τ stays near 0.5 on every dataset.
For trained rerankers the oracle τ moves between collections, so any fixed cutoff fails somewhere.

**Cost.** None. The probability for every query and passage is stored (`outputs/<dataset>/<candidates>/<system>/
<phrasing>/scores.jsonl.gz`, field `prob`).

**Open questions for the investigation.**
- Is set F1 the right utility? Alternatives: precision at the cutoff, utility with a cost per passage passed on,
  nDCG of the truncated list, or downstream RAG answer quality (out of scope for four pages).
- Which relevance grade counts as relevant (the paper uses grade ≥ 2 on DL)?
- How to handle queries where the right answer is "keep nothing"?
- Related work on score cutoffs and truncation (e.g. learned truncation in IR, Choppy, the "where to stop
  reading a ranked list" literature): what is the standard metric?

### Part B: query performance prediction (one paragraph; Kanishka)

**Use case.** Know without labels whether the reranked list is good, and then reformulate the query, retrieve
again, or answer "no relevant passage found".

**Experiment.**
- Per query, a predictor computed from the probabilities: the maximum p, the sum of p over the top-k (the expected
  number of relevant passages), or an expected DCG.
- Correlate it with the query's actual nDCG@10 across queries (Kendall's τ, Pearson's r, as is standard in QPP).
- Optional: AUROC for "the query has no relevant passage in the top 100".
- Compare with standard post-retrieval QPP baselines.

**Context.** Kanishka has QPP from Jev's probabilities on DL19/20 (`origin/master:results/qpp.md`,
`origin/master:results/qpp_depth.md`): first on the official 1000-candidate pools (commit 12ac0a1), then on the standard BM25 run
next to QPP-GenRE (a2fb674), and with deeper Jev judgments to estimate the ideal DCG (66a7a7b). Check his candidate
set and code before redoing it on our BM25 top-100 outputs for several models.

**Cost.** None (stored outputs).

**Open questions.** Which baselines (e.g. NQC, WIG, SMV on the BM25 scores; QPP-GenRE)? Is a calibrated model a
better predictor than an uncalibrated one with the same ranking, or does QPP only need the order within a query and
the spread across queries? That last point decides whether QPP belongs in a *calibration* RQ at all.

### Part C: calibrated out of the box vs. calibrated after the fact (the fair baseline)

**The objection.** "Just calibrate the trained reranker." Temperature or Platt scaling fits one or two parameters
on labelled data, leaves the ranking unchanged, and usually fixes ECE in-domain.

**Experiment.**
- Fit temperature scaling (one parameter) and Platt scaling (two) for each trained reranker on `msmarco-dev`
  judged pairs. The repository rules forbid fitting on test splits, and DL19-23 and BEIR are test splits.
- Evaluate on DL19-23 (ECE, Brier, and Part A's cutoff F1) and on BEIR (ECE with unjudged = non-relevant, as a
  shift check).
- Optional: apply the same scaling to the System One models, to show how much headroom they have.

**Possible outcomes (both publishable).**
- The scaling does not transfer: fitted on MS MARCO, wrong on BEIR or the shifted DL21-23. Then the claim is
  strong: decision models are calibrated across domains without fitting data.
- The scaling works. Then the claim narrows to: decision models need no labelled fitting data, rerankers do. That is
  still a practical advantage, and it answers the reviewer in advance.

**Cost.** One reranking run per trained reranker on a msmarco-dev subset (e.g. 1,000 queries x 100 BM25
candidates; `msmarco-dev` is registered in `src/cdm/datasets.py` and in `DEV_DATASETS` in `slurm/config.sh`). A few
GPU-hours in total.

**Caveat.** MS MARCO dev has mostly one judged relevant passage per query; fitting treats unjudged passages as
non-relevant, which biases the scaling towards low probabilities. Quantify the effect, or fit on a densely judged
non-test set if one exists (an open question for the investigation).

## 4. What the investigation should deliver

1. **A literature check** (with citations, preferably peer-reviewed):
   - calibration of neural rerankers and of LLM relevance judgments (e.g. work on calibrated rankers, LLM-as-judge
     calibration, verbalised vs. logit confidence);
   - score cutoffs and truncation of ranked lists;
   - QPP from reranker or LLM probabilities.

   Which of Parts A-C is new, which is established, and which metrics are standard?
2. **A recommendation**: which parts belong in the 4-page paper and which go to the arXiv extension, the final RQ2
   wording, and the exact metrics (with the bin scheme for ECE; consider adaptive bins or a debiased ECE).
3. **An experiment plan** in this repository's terms: which outputs to read, which new runs (only Part C needs any),
   and the analysis code to write, e.g. a new module `cdm.calibration_analysis` producing tables into `paper/tables/` via
   `results/*.csv`.
4. **Threats to validity**: the sparse judgments on BEIR and MS MARCO dev, the grade threshold, the ties caused by
   Jev's rounding to two decimals, and the fact that the triplets' fine-tunes saw different training data.

## 5. Constraints

- Never tune or fit anything on test splits (DL19-23, BEIR test); fitting data is `msmarco-dev` only.
- No computation on the login node: analyses that read stored outputs are light enough, new runs go through
  `slurm/submit.sh`.
- Numbers in the paper come from `results/*.csv` through generated tables, never typed by hand.
- Log what was tried in `docs/experiment-log.md`; decisions go to `docs/intent/` only once the owner has made them.

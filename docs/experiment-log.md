# Experiment log

Append-only, newest entry first. One entry per decision, failed attempt or surprising result -- the research plan
asks to "log every configuration tried, including failures". Successful runs are already recorded in
`results/results.csv` and in the `manifest.json` of their output directory; link to those instead of copying numbers.

Format:

```
## YYYY-MM-DD -- <short title>
What was tried / decided, why, and what happened. Job ids, output paths, commits.
Open questions or follow-ups.
```

## 2026-10-10 -- 9B validation run of our reranker recipe started

Owner's go (2026-10-10): run the 9B validation; if it succeeds, the 27B fine-tune follows
(`docs/intent/paper-focus.md` 5, criterion fixed before any result: average nDCG@10 over the ten datasets >= 57.0,
i.e. within 2 points of Prism-9B's 59.0, and above Clef-Flash isolated, 54.7). `qwen3.5-9b-reranker`: LoRA r 32 on
Qwen3.5-9B (post-trained), msmarco-mix-100k, `--loss both`, 2 x 80 GB, jobs 4411777 (30 h) and 4411778 (resume
after it, exits at once if the first finished). Before that: paper tables regenerated after the causal-conv1d
rerun, with `cdm cutoff` and `cdm posthoc` recomputed on the new probabilities (commit fa96d1e).

Training finished: 1,562 steps in 8.3 h on 2 H200s (kuat2), peak 25.7 GiB per GPU, final loss ~0.6; the resume job
(4411778) found the run finished and exited. Checkpoint selection on the first 200 msmarco-dev queries (4427678-
4427685), MRR@10 x 100: step 200 39.6, 400 38.9, 600 40.3, **800 41.8**, 1000 40.1, 1200 39.6, 1400 40.5, final
40.2. By the rule fixed in the plan (keep `final` unless an earlier checkpoint is more than 1 point better),
checkpoint-800 (+1.6) is the selected model; the differences are within noise for 200 queries, but the rule was
fixed in advance. Scored on all ten datasets with `--adapter checkpoints/qwen3.5-9b-reranker/checkpoint-800`
(4428637-4428646).

## 2026-10-09 -- Preparing our own 27B reranker fine-tune (not trained yet)

On the owner's request (in conflict with `docs/intent/paper-focus.md` decision 5, left untouched): everything needed
to fill the empty reranker cell of the Qwen3.8-27B triplet with our own fine-tune, for approval. Plan and decisions:
`docs/notes/reranker-27b-plan.md`. Code (commits 1ed5a9b, 2aafe8e, 9f96ce1 and the one with this entry):
`cdm finetune-data` (`src/cdm/finetune_data.py`), `cdm finetune` (`src/cdm/finetune.py`, LoRA, Qwen3-Reranker prompt,
listwise/pointwise loss, torchrun data parallel, checkpoint/resume, stop before the Slurm limit),
`slurm/finetune.{sbatch,interactive.sh}`, adapters in the Qwen3Reranker reader (`cdm rerank --adapter`), systems
`qwen3.8-27b-reranker`, `qwen3.5-9b-reranker`, `qwen3.5-0.8b-reranker` registered but not in the grid.

- **Framework:** ms-swift 4.5.3 pins transformers < 5.17 and peft < 0.21 (container: 5.19 / 0.21.2; Kev and Clef
  need >= 5.17), so a small custom trainer instead.
- **Data** (prefetch job 4395282, 3 min): Tevatron/msmarco-passage-aug and Tevatron/msmarco-passage train files
  (pinned revisions). `msmarco-mix-100k` (4 min on a CPU node): 100k queries, 1 positive + 8 dense + 7 BM25
  negatives, text without title. The leakage filter (id, normalised text or bag of words of any DL19-23, BEIR or
  msmarco-dev topic) removed 197 of 491,007 train queries, among them 12 DL21-23 queries (e.g. "what is Morgellons
  disease", DL22) and 3 DL19/20 queries that occur verbatim in MS MARCO train.
- **Smoke tests.** 0.8B on one L40S (exec 4395318; sbatch path 4397261-4397262 with a deadline stop at step 29 and
  `--resume` to step 64): gradient checkpointing and the backward passes of fla (Gated DeltaNet) and causal-conv1d
  work on Qwen3_5ForConditionalGeneration; 4.5 s per step of 16 groups, 3.1 GiB, loss from ln 16 = 2.8 to ~1.
  Scored with the reader (`--adapter`, 5 DL19 queries, `--no-save`): nDCG@10 65.7 after 1,024 training groups vs
  34.6 for `qwen3.5-0.8b-prompted` and 69.8 for Prism-0.8B -- the adapter loads and has learned (5 queries only).
  First sbatch attempt (4396726-4396727) stopped after one step: the 20-min safety margin equalled the 20-min job
  (margin now capped at 10% of the job).
- **Throughput.** 9B on one L40 (4398134): 22.6 GiB, 9.3 s per step of 4 groups, 1,090 real / 1,620 padded tokens/s
  (~48% of the L40's dense bf16 peak). Two groups per forward pass: no gain (4398146, L40S); no gradient
  checkpointing: OOM at 44 GiB. Extrapolated for 100k groups: 9B ~40 A100-hours (2 GPUs, ~20 h), 27B ~120
  A100-hours (4 GPUs, ~30 h), +-30%.
- **Not done:** the 27B memory/throughput probe (4397306, 1 x 80 GB, 50 min) and a 2-GPU check of the data-parallel
  path (4397307) are queued; Slurm estimated the 80 GB probe's start for Oct 13 -- the 80 GB nodes are the
  bottleneck. The multi-rank code itself was checked on CPU instead (exec 4398196: 2 ranks with gloo, fp32, CUDA
  kernels hidden; gradient all-reduce, rank-0 checkpoints, resume from step 2 to 4, final adapter); NCCL on GPUs
  waits for 4397307.

## 2026-10-09 -- Qwen3.5/3.8 grid rerun on the causal-conv1d image

Owner's decision: rerun every grid cell of a Qwen3.5/3.8-based system on the new container (causal-conv1d kernel),
so the paper's numbers come from one implementation. Submitted with `--overwrite` (jobs 4393693-4393880, 188 jobs):
Clef-Flash, Clef, Kev-4B, Kev-27B, Kev-0.8B, Kev-9B (joint and isolated), Prism-0.8B/4B/9B and the prompted
Qwen3.5-0.8B/4B/9B and Qwen3.8-27B on all ten datasets. Skipped, because they already ran on the new image: Kev-9B
joint on SciFact (4392314) and Qwen3.5-9B prompted on NFCorpus (4392309). Everything derived from these outputs
(main and triplet tables, RQ2 cutoff rows, post-hoc rows that read their probabilities) has to be regenerated when
the jobs are done. In parallel: a subagent prepares the 27B reranker fine-tune (plan, code, smoke tests; no
training before the owner's go).

Outcome (2026-10-10): 186 of 188 jobs completed. Per cell nDCG@10 moved by at most 0.6 points, averages by at most
0.2 (Clef isolated 56.2 -> 56.4); no conclusion changes. Two cells failed again and were resubmitted: Clef joint on
Touché ran out of memory on an 80 GB GPU (now an H200, 4409289), Kev-27B joint on DL20 hit 1h30 on kuat2 (now 3 h,
4409290); their rows still hold the old-image numbers. 27B fine-tune probe (4397306, 1 x A100-80GB, LoRA r 32,
4 groups of 16 per step): 61.2 GiB peak, ~16 s per step, ~630 real tokens/s, i.e. ~4 s per group -- ~110
A100-hours for 100k groups, consistent with the plan's extrapolated 120. The 2-GPU check (4397307) hit its 15-minute
limit; multi-GPU training is still untested on GPUs.

## 2026-10-09 -- RQ2 Part C: rerankers calibrated after the fact on msmarco-dev

Runs: `cdm first-stage msmarco-dev --max-queries 1000` (job 4393343, 77 s; `runs/msmarco-dev/bm25-d100-max1000`), then
the eight trained rerankers on it (4393344-4393351; 6 min to 1 h 14 min each). `cdm posthoc`
(`src/cdm/posthoc.py`) fits temperature (sigmoid(a z)) and Platt (sigmoid(a z + b)) scaling on each reranker's stored
logit and evaluates them on the ten test collections (rows `job = posthoc`, `variant` = native/temperature/platt;
table `paper/tables/posthoc.tex`).

- **The fit data is the problem.** The 1,000 dev queries x 100 candidates have 683 relevant passages (0.7%):
  msmarco-dev judges about one passage per query, and every unjudged candidate counts as non-relevant. Platt
  therefore learns a large negative bias (b = -5.3 to -8.6) and transfers badly to the densely judged DL sets:
  ECE on judged DL pairs ends near 0.23 for every reranker (better than native for Qwen3-Reranker-0.6B 0.511,
  monoT5-3B 0.312 and Set-Encoder 0.495; worse for Prism-4B 0.218 -> 0.238 and bge 0.231 -> 0.236; about equal for
  Prism-0.8B and Prism-9B), and the set at p >= 0.5 is almost
  empty (F1@0.5 5.7-7.4, regret 41-48). ECE with unjudged = non-relevant does improve (e.g. DL23 0.05-0.06), but only
  because that convention matches the sparse fit labels.
- **Temperature scaling** lowers DL ECE for every reranker (Qwen3-Reranker-0.6B 0.511 -> 0.326, monoT5-3B 0.312 ->
  0.257, Qwen3-Reranker-8B 0.257 -> 0.237) but none reaches the decision models' 0.14-0.16 (Kev-27B joint 0.156,
  Clef joint 0.147, Clef-Flash joint 0.140), and by construction it leaves the set at 0.5 unchanged.
- **Bug fixed on the way:** without a bias term the Set-Encoder's uniformly positive logits made the fit choose a
  negative slope (a = -0.095), which reverses the ranking. The temperature is now constrained positive; for the
  Set-Encoder it then collapses to a ~ 0 (every probability 0.5) -- degenerate, to be read as "temperature scaling
  cannot fix it".
- Reading so far: "just calibrate the reranker" needs densely judged in-domain labels; with what MS MARCO offers,
  post-hoc scaling does not give the reranker fine-tunes the decision models' out-of-the-box calibration, and Platt
  makes the fixed cutoff unusable. Whether a fairer fit set exists is open (owner decision): msmarco-dev with a
  prior correction, or a densely judged non-test collection.

## 2026-10-09 -- RQ2 Part A: one fixed cutoff (p >= 0.5) on DL19-23

New: `cdm cutoff` (`src/cdm/cutoff.py`, `src/cdm/cli/cutoff.py`) reads the stored probabilities of every rerank row on
the default candidates of DL19-23 and writes one `job = cutoff` row per system and dataset to results.csv: set
precision/recall/F1, size, share of empty sets and of unjudged passages at p >= 0.3/0.5/0.7, the per-dataset oracle
cutoff (grid of 0.001) with its F1 and the regret of 0.5, and the F1 of the top-5/10/20. Conventions: relevant from
grade 2, unjudged = non-relevant, recall relative to the relevant candidates, macro means over judged queries.
bge-m3 and the Set-Encoder on DL19/20 come from the efficiency run (`variant = a100`, same candidates) because their
default cells are literature numbers. Run with `slurm/exec.sh` on a CPU node (a few minutes, all 39 systems with
probabilities, including dropped ones whose outputs still exist). Table: `paper/tables/cutoff.tex`.

First reading (means over DL19-23, x 100; F1 at 0.5 / regret against the oracle cutoff):
- **The reranker fine-tune wins here too.** Per backbone, best F1 at 0.5: Qwen3-Reranker-0.6B 31.8 (Kev-0.6B 20.7),
  Qwen3-Reranker-8B 46.3 (Kev-8B isolated 40.1), Prism-0.8B 46.9 (Kev-0.8B 24.7), Prism-4B 51.0 (Kev-4B 41.0),
  Prism-9B 50.7 (Clef-Flash isolated 49.2, Kev-9B 41.0). Kev-27B joint is the best system overall (55.1 / 2.9).
- **One cutoff fits every collection for some models, not only decision models.** Low regret: Kev-27B joint 2.9,
  Clef joint 3.4, Clef-Flash joint 2.4, Laya 1.2 -- but also Prism-0.8B/4B/9B 2.7 / 2.9 / 4.0 and bge-reranker
  3.8, whose oracle cutoffs stay around 0.5-0.8. High regret: Set-Encoder 21.1, Qwen3-Reranker-0.6B 14.9, monoT5-3B
  9.9, Qwen3-Reranker-8B 9.0, with oracle cutoffs of 0.85-0.99 (Qwen3-Reranker-8B 0.72-0.95): their
  probabilities are too high.
- Joint mode lowers the regret of the strong decision models (Kev-27B 2.9 vs 7.9 isolated, Clef 3.4 vs 5.8, Jev 6.2
  vs 7.8), in line with their lower ECE. Jev's regret (6.2) is higher than the open 27B models'.
- Small models and zero-shot base models keep almost everything at 0.5 (Qwen3-0.6B, Qwen3.5-0.8B base: ~99 of 100
  passages); low regret there means a flat F1 curve, not good calibration. Several base models saturate at p = 1.0
  (oracle cutoff 1.000).
- So far RQ2 reads: a usable fixed cutoff is a property of some training recipes (decision fine-tuning, Prism, bge),
  not of decision models alone. Part C (post-hoc scaling of Qwen3-Reranker/monoT5 on msmarco-dev) decides whether
  the high-regret rerankers are one fitted parameter away.

## 2026-10-09 -- New narrative: reranker fine-tuning beats decision fine-tuning; calibration as RQ2

Owner's decision (`docs/intent/paper-focus.md`, rewritten): the headline "open matches closed" is dropped. Open decision
fine-tunes matching the closed Jev within weeks is read as a sign of a training paradigm (RLCD), not a breakthrough,
and becomes motivation. The paper now leads with a negative result on the triplets: reranker fine-tuning beats
decision fine-tuning on every backbone where both exist (0.6B-9B; better decision mode vs. reranker fine-tune, Avg
nDCG@10: -29.1, -6.7, -21.2, -7.3, -4.3), while decision fine-tuning stays within -3.6 to +3.7 of the zero-shot base
model. An earlier reading in the project report ("decision fine-tuning gains over the base model only at 27B") was
wrong: Kev-0.6B isolated is +3.2 over its base and Clef-Flash +1.0 at 9B.

- **RQs:** RQ1 = decision vs. reranker fine-tuning on one backbone, joint vs. isolated folded in (explained by the
  state length the model was trained on); RQ2 = is calibration still of use, argued in a discussion with Part A
  (one cutoff p >= 0.5, set F1, regret against the oracle cutoff; stored outputs) and Part C (temperature/Platt
  scaling of the reranker fine-tunes fitted on msmarco-dev). The three-RQ structure (effectiveness, joint vs.
  isolated, calibration) is gone.
- **Scope:** the claim covers 0.6B-9B; no 27B reranker fine-tune exists (limitation, not planned).
- **Title:** a question, *Do System One Models Make Better Rerankers? Decision vs. Reranker Fine-Tuning on One
  Backbone* (working title).
- **Renamed/removed:** `docs/notes/rq3-calibration-proposal.md` -> `docs/notes/calibration-proposal.md` (status and RQ
  numbering updated); paper sections `05_joint_scoring.tex` removed, `06_calibration` -> `05_calibration`,
  `07_conclusion` -> `06_conclusion`; `paper/tables/main.tex` replaced by `triplets.tex` (main table, RQ1) and
  `context.tex` (other systems) from `src/cdm/paper_tables.py`. Stale terms added to the straighten-branch
  deprecated list. Roadmap: new steps 5-7 (triplet runs, RQ1 analysis, RQ2 analysis incl. msmarco-dev runs); Phase 2
  fine-tuning reframed as "rank like a reranker fine-tune, stay calibrated like a decision model".
- Grid and registries unchanged: every system stays (Jev, Clef, Laya and the trained/listwise rerankers form the
  context table). No queued job is affected.

## 2026-10-09 -- Triplet runs; why small Kev models fail in joint mode; causal-conv1d

- **Triplet runs** mostly done (325 of 350 cells). 10 failures: GPU memory on long BEIR passages -- the yes/no
  readers computed logits for every position over the vocabulary (now `logits_to_keep=1`, identical scores checked
  on 3 DL19 queries for three models), Kev-8B/9B joint on 44 GiB GPUs (now 80 GB) -- and one 3 h timeout of
  Qwen3.5-9B prompted on NFCorpus (now 6 h). Resubmitted 4392305-4392314.
- **Small Kev models below BM25** (Kev-0.6B: 27.0 isolated, 13.2 joint on average). On DL19/20 (judged pairs,
  relevant = grade >= 2): Kev-0.6B joint does not discriminate at all -- AUC of relevant vs. judged non-relevant
  0.49 / 0.47, almost every passage gets p ~ 0.83, and p *rises* with the passage's position in the state (Spearman
  +0.28 / +0.33). Isolated it reaches AUC 0.64. Kev-0.8B: 0.63 joint vs 0.69 isolated; Kev-4B: 0.81 vs 0.82;
  Kev-27B: 0.92 vs 0.92 / 0.89. The cards explain it: Kev-0.8B and Kev-4B were trained on states of at most 7,552
  tokens (validated to 8,192), Kev-27B on up to 32,768 (validated to 65,536). A joint DL state of 100 passages is
  about 13k tokens, a BEIR one 40k or more -- far outside the small models' training range, where a question about
  `documents.D57` is no longer tied to the right passage. This supports explaining RQ2 by request length
  (owner's decision 8). Possible control: joint requests cut to each model's trained state length.
- **Container**: causal-conv1d 1.7.0 compiled into `container/cdm.sif` (pip nvcc 13.0, archs 8.0/8.6/8.9/9.0;
  the pip CUDA wheels needed a `libcudart.so` link name), so Qwen3.5/3.8 models (Kev, Clef, Prism, prompted) no
  longer use the slow reference convolution. Kernel check and trial on L40S: Kev-4B joint 8.1 s per DL19 query.
  The kernel changes scores in the last digits: Qwen3.5-0.8B prompted on 3 DL19 queries keeps ECE/Brier but
  nDCG@10 moves 0.4945 -> 0.4650 through near-ties. The grid results of Qwen3.5/3.8 models were computed without
  the kernel (except jobs that started after 2026-10-09 10:08); the efficiency rows were rerun with it: all
  Qwen3.5/3.8 and triplet systems on DL19/20, `--variant a100 --overwrite` (54 jobs).

## 2026-10-09 -- RankZephyr reproduces; efficiency run done; triplets set up

- **RankZephyr check passes**: DL19 74.3 / DL20 70.0 (jobs 4377662-4377663) against the published 74.2 / 70.2
  (Abdallah et al.); DL21-23 68.0 / 50.5 / 44.1. 6-15% of the windows' answers miss identifiers (55-150 per
  dataset), filled in their old order as rank_llm does.
- **Efficiency run complete** (variant `a100`, 4377668-4377696, all on A100-SXM4-80GB): e.g. on DL19, per query,
  Set-Encoder 0.24 s / 83 J, monoT5-3B 0.68 s / 258 J, Qwen3-Reranker-8B 2.85 s / 1,139 J, Clef-Flash joint
  2.26 s / 850 J, Kev-27B joint 14.3 s / 3,311 J, RankZephyr 22.3 s / 4,364 J. Kev is slow for its size (Kev-4B
  joint 10.7 s): its exact-dtype path (`kev-exact`), worth a look before the efficiency table is written.
- **Triplets** (owner's decision, `docs/intent/paper-focus.md` 9-11): base model zero-shot / decision fine-tune /
  reranker fine-tune on Qwen3-0.6B, Qwen3-8B, Qwen3.5-0.8B/4B/9B and Qwen3.8-27B. Checked on the cards:
  Qwen3-Reranker-0.6B/4B/8B start from Qwen3-*-Base, the same checkpoints as Kev-0.6B/8B (Kev-8B's provenance
  records Qwen3-8B-Base revision 49e3418f); Kev-27B starts from Qwen3.8-27B's post-trained release (revision
  1d4bf0f2); Prism's paper and card say only "the Qwen3.5 causal LM", base or post-trained unstated. Training
  data: Qwen3-Reranker and Prism (KaLM-Embedding aggregation) include MS MARCO, NQ, HotpotQA, FiQA; Kev's sources
  are classification and QA sets without MS MARCO or BEIR. New code: `src/cdm/rerankers/prompted.py` (zero-shot
  base), Prism through the Qwen3-Reranker reader. Model downloads: job 4388029.
- **Qwen3.7-Flash dropped** from the paper (code and grid entry removed; results stay).

## 2026-10-08 -- RankZephyr and the efficiency run

Owner's decisions (`docs/intent/paper-focus.md`): RankZephyr run by us, one fixed-GPU latency/energy run, main table
joint only, open title, RQ1 led by open vs. closed.

- **RankZephyr** (`src/cdm/rerankers/listwise.py`): rank_llm's prompt and sliding window (20/10, 9 windows per
  query) with transformers in bf16, greedy. Trial on 3 DL19 queries (L40S, no-save): nDCG@10 0.777, 20.4k input
  and ~800 output tokens per query (Chen et al.: 18.2k / 711), 21 s per query, 5 of 27 answers missing identifiers
  (filled in their old order, as rank_llm does). Submitted DL19-23 (4377662-4377666); BEIR stays from the literature.
- **GPU energy** (`src/cdm/energy.py`): NVML's cumulative energy counter via ctypes around the timed loop; results.csv
  gets `energy_j`, `energy_j_per_query`, `rerank_wall_s`. Check on bge-m3 (5 DL19 queries, L40S): 123 J per query
  at 0.33 s, i.e. ~370 W.
- **Efficiency run**: all 14 local systems on DL19/20 with `--gres gpu:a100-sxm:1 --variant a100`
  (4377668-4377696); bge-m3 and Set-Encoder with `--include-literature` (their DL19/20 effectiveness cells stay from
  the literature).

## 2026-10-08 -- Grid batch 4360xxx: failures fixed; Jev through OpenRouter prepared

The grid batch of 2026-10-07 (jobs 4360571-4360672) left 114 of 160 cells done. Failures and fixes:

- **NFCorpus, every joint System One run** (jev, clef-flash, clef, kev-4b, kev-27b; e.g. 4360594):
  `range() arg 3 must not be zero`. A query without BM25 candidates gives a joint window of 0 passages.
  SystemOneReranker.score now returns empty scores for an empty list (isolated runs were never affected).
- **set-encoder-large, 7 datasets** (4360577-4360583): ir_datasets, imported by lightning-ir, creates ~/.ir_datasets,
  read-only since the container runs with --no-home. IR_DATASETS_HOME now points to data/ir_datasets (slurm/env.sh).
- **qwen3-reranker-8b, 5 datasets** (4360571-4360575): CUDA out of memory on 44 GiB L40S. Now gpumem.80gib
  (A100-80GB/H200, gpu partition).
- **Time limits:** kev-4b-joint on Touche timed out at 45 min (4360656); kev-27b-isolated Touche took 1h27 of 1h30
  (4360663), kev-4b-joint SciFact 2h51 of 3h (4360655). Touche budgets raised to 1h30 (3h for the 27B models),
  Kev-4B on NFCorpus/SciFact/DBPedia to 5h. clef-joint Touche ran out of memory on an 80 GB GPU (4360636): open.
- **jev-isolated, NFCorpus/SciFact/DBPedia** (4360587, 4360588, 4360590): TypeSafe 402, the organisation's credits
  are used up.
- **Rerun (4372149-4372198):** with the new TypeSafe key all Jev cells finished. 15 NFCorpus queries have no BM25 hit
  (e.g. PLAIN-1214 "Fosamax"); Qwen3-Reranker-8B, the Set-Encoder and Qwen3.7-Flash failed on them too (4372151,
  4372157, 4372195). `cdm rerank` now skips a query without candidates (it counts with nDCG 0, as before);
  resubmitted as 4372201-4372203.

Jev through OpenRouter: OpenRouter serves Jev on the System One endpoint (`https://openrouter.ai/api/v1/systemone`,
the TypeSafe SDK with another base URL; openrouter.ai/docs/guides/community/typesafe-sdk). TypeSafe is the only
provider, the price is the same ($0.042 per million input tokens), the context 32k. OpenRouter has no patch
versions: `typesafe/jev-1.13`, answered by the snapshot `typesafe/jev-1.13-20260917`. `JEV_PROVIDER=openrouter` in
.env switches the Jev backend (system names unchanged; the manifest records provider, requested and served model).
Before OpenRouter results count as jev-1.13.0, the snapshot has to answer like TypeSafe's jev-1.13.0: rescore the
first queries of DL19 in both modes (`--max-queries 5 --variant openrouter-check`) and compare the probabilities with
the stored TypeSafe responses. Also unchecked: whether OpenRouter passes on TypeSafe's `max_tokens_exceeded` error
(the backend also accepts a 400/413 that mentions the context); joint DL19 requests always need a split, so the check
run exercises it.

## 2026-10-07 -- One setup (BM25 top-100, literature numbers), straightened branch

Owner's decisions: simplify towards S1Rank's setup so published numbers can stand in for runs; Kev in two sizes;
RankZephyr from the literature; S1Rank's terms joint/isolated pointwise instead of listwise/pointwise; three paper RQs
plus a Phase 2 (control signal, fine-tuning, BRIGHT). Plan of record: docs/roadmap.md.

- **Candidates: BM25 top-100 everywhere.** The DL19/20 official pools are not the literature's candidate set: our
  BM25 order of the 1000-candidate DL19 pool gives nDCG@10 0.522 (DL20 0.488), Pyserini's standard BM25 run 0.5058
  (0.4796), which S1Rank (50.6), RankGPT, RankZephyr and Abdallah et al. (2025) rerank. DL19/20 therefore use
  bm25-d100; DL21-23 keep the official top-100 files, which equal Pyserini's BM25 top-100 (2CR match). The
  unranked-pool code (tsv pools, BM25 re-ordering, top1000.dev) is removed.
- **Datasets:** FiQA and SciDocs out; Touche and DBPedia in (published RankGPT/RankZephyr/monoT5 numbers).
- **Systems:** dropped minilm-l6, mxbai-base, monot5-base, qwen3-reranker-0.6b/4b, set-encoder-base,
  laya-multilingual, kev-0.8b, kev-9b. Kept: bge-m3, monot5-3b, qwen3-reranker-8b, set-encoder-large, qwen3.7-flash,
  jev/clef-flash/clef/kev-4b/kev-27b (joint and isolated), laya. Draft phrasings p1-p3 removed (only s1rank).
- **results/literature.csv** (new): Abdallah et al. (2025) Tables 2-3 for RankGPT-4, RankZephyr-7B, monoT5-3B,
  bge-reranker-v2-m3; Set-Encoder-large DL19/20 (model card); S1Rank and Pyserini 2CR BM25 as references.
  `from_literature` in slurm/config.sh makes slurm/submit.sh skip those cells. RankZephyr's own paper reports DL21/22
  only on SPLADE++ candidates, so it has no DL21-23 cells.
- **Renamed** with `cdm rename-system` (new): `<model>-pointwise` -> `-isolated`, `-listwise` -> `-joint` (jev, clef-flash,
  clef, kev-0.8b, kev-4b, kev-27b; 35 rows). Output directories keep the old names.
- **Cancelled** the 15 remaining DL19/20 official-pool jobs (2 running Clef-27B full-pool runs, 13 queued).
- **99 of the DL21-23/DL19-20 jobs had failed** on two infrastructure problems, now fixed in slurm/env.sh:
  node skadi could not create the home mount point ("mkdir /home/dinzinger: file exists", 97 jobs, all within
  seconds), and Triton's cache ~/.triton was unwritable or full on compute nodes (Kev, Laya). The container now runs
  with --no-home and TRITON_CACHE_DIR under data/cache; checked on skadi (job 4360426). slurm/exec.sh got --nodelist.
- Measured run times on 100 candidates (DL21-23): Kev-27B isolated 29-44 min, Clef 27B 7-18 min, all else < 10 min;
  budgets in slurm/config.sh now follow these.
- New project skill `straighten-branch` (with a consistency check and deprecated.txt); `experiment-status` follows
  the new grid (lit cells, `next` section with submit commands).

## 2026-10-07 -- DL 2021-2023: Jev done; DL19/20 submitted

- First DL 2021-2023 chain: prefetch 4356461 failed after 140 GB (hf_xet "os error 11"), leaving 78 jobs stuck;
  cancelled, prefetch now retries (f1077cb), resubmitted: prefetch 4357122 (5 min), first stage 4357123-4357125
  (4 min each), 75 reranks (logs/submitted-dl21-23.txt). Jev pointwise/listwise finished (40 s / 2-3 min per
  dataset, $0.45 in total); 69 GPU reranks waiting for GPUs (Priority).
- Official-pool BM25 reproduces Pyserini's 2CR BM25 on msmarco-v2-passage exactly: nDCG@10 0.4458 / 0.2692 / 0.2627.
- Jev listwise above pointwise on all three years (about +0.02 nDCG@10, lower ECE and Brier), 3x cheaper; no
  significance test yet. results.csv committed (efa4661).
- DL19/20 submitted from efa4661 (logs/submitted-dl19-20.txt): prefetch 4357489 (MS MARCO v1 index + pools),
  first stage 4357490/4357491, then 46 reranks on all candidates (every system but Set-Encoder; the comparison
  with Kanishka) and 50 at --depth 100 (all systems; RQ4). System One models with s1rank only. Budgets for the 27B
  models on the full pools raised to 16 h (a69268e), unmeasured.

## 2026-10-07 -- Pointwise/listwise naming, Set-Encoder, first DL 2021-2023 submission

- **Naming (owner's decision):** IR terms. System One models are `<model>-pointwise` (one candidate per request,
  formerly `-isolated`) and `<model>-listwise` (all candidates in one state, formerly the plain name); Laya stays
  `laya` (pointwise only). `Reranker.joint` is now `Reranker.listwise`; the manifest field likewise. RQ4's wording
  in paper/rq-macros.tex and the Section 6 skeleton use the same terms; RQ4 compares Jev, Clef and Kev.
- **Set-Encoder** (`set-encoder-base`, `set-encoder-large`, webis, pinned revisions) through lightning-ir 0.0.6, in a
  second image `container/lightning-ir.sif` (slurm/config.sh `container_for`). The PyPI release of lightning-ir 0.0.6
  does not pin transformers < 5 (its main branch does); the first build resolved transformers 5.19, so the .def now
  pins `transformers>=4.50,<5` (rebuilt: transformers 4.57.6, torch 2.14.1). Refuses lists longer than its trained
  depth (100): use `--depth 100` on the 1000-candidate pools. fp32 (the authors evaluate in bf16-mixed).
- Containers rebuilt on freia: cdm.sif (torch 2.14.1, transformers 5.19.0, laya 0.3.29, kev 0.1.0 at 5e42a7a,
  flash-linear-attention 0.5.2), lightning-ir.sif. Import check on a compute node: job 4356457 (cdm.sif).
- **Submitted** (ids in logs/submitted-dl21-23.txt, not tracked): prefetch 4356461 (MS MARCO v2 passage index + the
  23 GPU systems, CPU node, 12 h budget), first stage 4356462-4356464 (dl21-23, official top-100 pools), and 69
  reranks (23 GPU systems x 3 datasets), each after its dataset's first stage. System One models with
  `--phrasings s1rank` only (frozen; the draft phrasings stay off test sets until WP3). Not run: `jev-*` and
  `qwen3.7-flash` (no API keys in .env on the cluster yet), `laya-multilingual`. Code: a32d416 plus uncommitted
  changes (REVISION.patch in every manifest), so these runs are preliminary until rerun from a commit.

## 2026-10-07 -- Working on the login node; TREC DL 2021-2023

Owner's decision: the repository now lives on the cluster (`/shared/dinzinger/GitHub/decision-rerank`) and is used
from the login node (freia) directly, so the SSH wrappers in `cluster/` are gone. `slurm/submit.sh` and
`slurm/exec.sh` now write the per-submission code snapshot and its REVISION/REVISION.patch themselves
(`slurm/snapshot.sh`; formerly `cluster/sync.sh`). Results are written to `results/` in place, nothing is fetched.
FAITH forbids unsupervised autonomous agents and compute-heavy work on login nodes; CLAUDE.md now spells out that an
agent works here only in a supervised session and runs nothing heavy.

- Mistake in the previous session: believing it ran on the user's machine, the agent installed PyTorch and ran
  monoT5-base, Qwen3-Reranker-0.6B and Kev-0.8B on three passages each on the CPU of freia (a few minutes, ~7 GB of
  downloads in /tmp, which is RAM-backed). Removed afterwards; not to be repeated (CLAUDE.md).
- **DL 2021-2023 registered** (`dl21`, `dl22`, `dl23`): Pyserini index `msmarco-v2-passage` (39 GB, stores the raw
  passages), topics `dl2x`, qrels `dl2x-passage` (names from Pyserini's 2CR), relevant from grade 2, official
  `202x_passage_top100.txt.gz` pools (ranked BM25 runs, kept in their order: `official-d100`). The index is
  downloaded by a new batch job, `slurm/submit.sh prefetch --datasets dl21,dl22,dl23 --systems none`
  (slurm/prefetch.sbatch; the login-node variant is slurm/prefetch.interactive.sh, formerly slurm/prefetch.sh).
  Untested: the first prefetch shows whether compute nodes reach Pyserini's download hosts and whether every judged
  query has a pool.

## 2026-10-07 -- Official rerank pools; Kanishka's systems added

Owner's decision (docs/intent/models.md): rerank official pools where they exist, and add the systems of Kanishka's
study (branch `master`, package `decisionrerank`, an unrelated history) with larger sizes for the H200s. Not yet run.

- **Pools.** `cdm first-stage` defaults to the official pool where a dataset has one (`official-d1000`): DL19/DL20
  (`msmarco-passagetest{2019,2020}-top1000.tsv.gz`) and MS MARCO dev (`top1000.dev`, from the 687 MB
  `top1000.dev.tar.gz`; whether it covers all 6,980 dev-small queries is printed by `cdm prefetch`). The tsv pools are
  unranked: they are ordered by Pyserini BM25 (k1 0.9, b 0.4) over the pool, ties by docid. Kanishka orders the same
  pool by bm25s (Lucene variant, k1 1.2, b 0.75), so BM25 rows and depth cut-offs differ slightly; the full-pool
  reranking does not. `cdm rerank` defaults to the dataset's default first stage. BEIR stays on our BM25 top-100.
  DL 2021-2023 also have official (ranked, top-100) pools, checked at msmarco.z22.web.core.windows.net, but their
  passages are in MS MARCO v2 (138M); not registered.
- **Systems.** monoT5 (base, 3B-10k) and Qwen3-Reranker (0.6B, 4B, 8B) with Kanishka's prompts and yes/no readout,
  fp32 + TF32 (his finding: bf16 makes 89% of Qwen3-0.6B's pool scores tie); Kev 0.8B/4B/9B/27B as a fourth
  SystemOneReranker backend (joint and isolated; Kev's exact path, fp32, shipped temperature); Qwen3.7-Flash via
  OpenRouter, P(yes) from top-5 log-probs, Kanishka's prompt.
- **Laya now runs with autocast off** (fp32 weights, TF32 matmuls), as Kanishka does: under the default bf16 autocast
  32% of its DL pool scores tied within a query. Earlier Laya trial runs (scifact, not saved) used bf16.
- **Container.** Kev is installed with `--no-deps` from a pinned commit (its `torch<2.9` pin would downgrade torch
  2.14); flash-linear-attention 0.5.2 added (Qwen3.5 fast path for Kev and Clef, so Clef numbers may shift slightly
  from the 2026-10-06 trial). Needs a rebuild; Kev on torch 2.14 is unverified until the first run.
- Differences from Kanishka that remain: our System One state is `{"query", "documents": {"D1": ...}}` with the
  phrasings of src/cdm/phrasings.py, his `{"query", "passage"}` with "Does the 'passage' answer the 'query'?";
  Jev through TypeSafe (`jev-1.13.0`) here, OpenRouter (`typesafe/jev-1.13`) there.

## 2026-10-06 -- First Laya and Clef-Flash runs (scifact, 20 queries)

Container 35af14a adds gcc (Triton's run-time C build; Laya on GPU failed without it, job 4346919) and accelerate
(device_map for Clef, job 4346907). Reruns on corellia (L40S), 20 scifact queries, bm25-d100-max20, phrasing s1rank,
--no-save:

| system | job | nDCG@10 | ECE_unjudged0 | requests | splits | truncated | p50 / query |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | -- | 0.614 | | | | | |
| minilm-l6 | exec | 0.674 | | | | 19.4% | 0.11 s |
| jev (joint) | exec | 0.759 | | 38 | 18 | 0 | 1.09 s |
| clef-flash (joint) | 4346927 | 0.666 | 0.067 | 82 | 62 | 0 | 9.5 s |
| laya (English, isolated) | 4346926 | 0.374 | 0.310 | 2,000 | 0 | 22.7% | 0.74 s |

- Laya zero-shot ranks below BM25 on these queries, with 22.7% of passages cut to its ~320-token state budget: an
  early WP2 no-go signal, to confirm on TREC DL with all queries.
- Clef-Flash (16k context) needed 62 splits for 20 queries (~4 requests of ~25 abstracts each); about MiniLM level.
- scifact judges no non-relevant documents (`n_judged_nonrel_pairs` = 0), so only `ECE_unjudged0` means anything.
- 20 queries only: the pipelines work; no conclusions.

## 2026-10-06 -- Laya and Clef wired up; evaluation fixes

One `SystemOneReranker` with three backends (TypeSafe API, laya, Clef's joint_schema_model.py), commit f96953a.
Container rebuilt with laya 0.3.28 (transformers 5.18.0 provides Qwen3_5ForConditionalGeneration; Clef's encoder runs
under it). Prefetched Laya (7b928d82) and Clef-Flash (17f0b0ad, ~19 GB).

- Laya, 2 scifact queries on CPU (`--max-queries 2`, not saved): the code path works; 200 requests, 35 of 200
  passages (17.5%) cut to the English checkpoint's state budget, 109 s per query on 8 CPU cores (a 20-query CPU run
  was killed by the client timeout after 28 min). Laya warns that its checkpoint ships an out-of-range temperature for
  choice questions with 11+ options (clamped); Noul is unaffected.
- No GPU was free in `short` or `gpu` (expected start 22:43). Queued as batch jobs, 20 scifact queries, s1rank,
  --no-save: Clef-Flash 4346907, Laya 4346919.

Fixes found on the way:

- **Evaluation over a query subset.** `evaluate` passed all judged queries to ir_measures, which scores a judged query
  missing from the run as 0, so `--max-queries 2` on a 20-query cache averaged over 20 (nDCG@10 0.0). Now only ranked
  queries are evaluated. Earlier runs were unaffected (run and cache had the same queries).
- **Calibration variants** (owner's decision): ECE/Brier on judged pairs as before, plus `ECE_unjudged0` /
  `Brier_unjudged0` with unjudged candidates as non-relevant, and `n_judged_nonrel_pairs`, so tables can restrict the
  main calibration results to sets with judged non-relevant documents.
- SSH: the shared connection dropped three times (after ~90, ~7, ~25 min); keepalives (ServerAliveInterval 60) were
  added to `~/.ssh/config`. Opening it from a separate terminal has held since 12:51.

## 2026-10-06 -- Laya and CLEF join; calibration on sets with judged negatives

Owner's decisions: integrate Laya (open weights, `convaiinnovations/laya`) and Cloudflare's CLEF (`Cloudflare/clef`,
`Cloudflare/clef-flash`) next to Jev behind one System One reranker. Unjudged documents count as non-relevant, but the
main calibration results use only sets with judged non-relevant documents. See `docs/intent/models.md`.

First real Jev run (`cdm rerank jev scifact --first-stage bm25-d100-max20 --phrasings s1rank --no-save`, 20 queries,
not saved): nDCG@10 0.759 (BM25 0.614, MiniLM 0.674), 38 requests for 20 queries (18 split in half because 100 scifact
abstracts exceed Jev's context), 890k input tokens, $0.037 ($1.87 per 1k queries), latency p50 1.09 s per query.
Compute nodes reach api.typesafe.ai and huggingface.co directly.

## 2026-10-06 -- Jev first; Laya API pending

Owner's answer to open question 3: the TypeSafe AI API key for Jev is in `.env` (`TYPESAFE_AI_API_KEY`); the owner
is trying to get Laya API access. Work starts with Jev. No API budget stated. See `docs/intent/models.md`.

## 2026-10-06 -- Venue: arXiv, then WSDM 2027 short paper

Owner's decision on open question 1: an arXiv preprint, then a WSDM 2027 short paper (deadline Nov 17, 2026, 4 pages
plus references, ACM sigconf, anonymised; arXiv preprints allowed). This replaces the 2026-10-05 decision of no fixed
deadline. Details and consequences in `docs/intent/venue.md`.

Open question 2: the arXiv preprint goes up before Nov 17, 2026 (owner, 2026-10-06).

## 2026-10-06 -- Cluster bring-up

Bring-up sequence of the setup session run against FAITH; everything works after two fixes.

- Container: `cluster/login.sh bash container/build.sh` built `container/cdm.sif` (3.4 GB, ~13 min) with fakeroot on
  the login node. Resolved versions: torch 2.14.1, transformers 5.18.0, sentence-transformers 6.1.0, pyserini 2.4.0,
  ir_measures 0.4.3, numpy 2.5.3, pandas 3.0.6 (Python 3.12).
- Prefetch `--datasets scifact --systems minilm-l6`: the BEIR scifact index/topics/qrels names resolve (300 queries,
  339 judgments); MiniLM snapshot `233902d2`. The other datasets' Pyserini names are still unchecked.
- `cluster/submit.sh hello`: job 4345829, 49 s on pistis07 (budget 2 min, kept).
- `cluster/exec.sh --gpu -- cdm info`: job 4345830, NVIDIA L40S on corellia, CUDA 13.0.
- `cdm first-stage scifact --max-queries 20`: BM25 nDCG@10 0.6143 on 20 queries (`runs/scifact/bm25-d100-max20`).
- `cdm rerank minilm-l6 scifact --first-stage bm25-d100-max20 --no-save`: nDCG@10 0.6738, ECE 0.170, latency p50
  112 ms/query (100 pairs), 19.4% of pairs truncated at 512 tokens. The cross-encoder loads and scores under
  sentence-transformers 6.1.0. Not saved (trial run).

Fixes:

- **GPU constraint.** `gpumem.40gib` does not exist on FAITH. Memory tiers: 6, 8, 11, 16, 20, 24, 48, 80, 141 GiB.
  Switched to `gpumem.48gib` (slurm/config.sh, README, comments): A100-SXM, H200, L40, L40S; in `short` only L40
  and L40S. `gpumem.24gib` selects the same nodes; 16gib would admit the AMD rx_9060_xt nodes.
- **Test order dependency.** With the real ir_measures, `test_rerank_cli` runs (it was skipped with the stand-in) and
  reloads `cdm.runs`; `test_runs::test_roundtrip` then compared instances of the old and new dataclasses and failed.
  The test now imports inside the function. All 11 tests pass in the container.

Notes:

- Slurm account `chair-ds` is valid for `general`, `gpu` (QOS lopri/normal) and `short` (QOS short: at most 1 GPU,
  20 running and 100 submitted jobs per user).
- pytest is not in the image (dev extra). For test runs it is installed to `data/devtools` on the cluster:
  `cluster/exec.sh -- bash -c 'PYTHONPATH=$PYTHONPATH:$CDM_ROOT/data/devtools python -m pytest -q tests'`.

## 2026-10-05 -- Repository set up

Repository scaffolded from the research plan (`docs/research-plan.md`): cluster wrappers (`cluster/`), Slurm jobs
(`slurm/`), container (`container/`), package `cdm` with BM25 first-stage caching, cross-encoder reranking and
evaluation (WP1 starters), LNCS paper skeleton (`paper/`). Laya and Jev are stubs until WP0.

Not yet verified on the cluster: container build, Pyserini index names, `gpumem.40gib` feature, `--time` budgets.
First steps: `cluster/connect.sh`, `cluster/login.sh bash container/build.sh`, `cluster/submit.sh hello`.

Open decisions for the project owner (from the plan): target venue and deadline; compute and API budget;
MIRACL in or out; RQ5 in scope or not.

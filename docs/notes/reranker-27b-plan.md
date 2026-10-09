# Our own 27B reranker fine-tune: plan for approval

Prepared 2026-10-09 (branch `dev-michael`). Nothing below has been trained beyond smoke tests; the real runs wait for
the owner's go. Code: `src/cdm/finetune_data.py`, `src/cdm/finetune.py`, `src/cdm/cli/finetune.py`,
`slurm/finetune.{sbatch,interactive.sh}`; smoke-test record: `docs/experiment-log.md`, 2026-10-09.

**Conflict with the owner's decisions.** `docs/intent/paper-focus.md`, decision 5, says "a 27B reranker fine-tune is
not planned". This plan was prepared on the owner's later request; the decision file is untouched. If the run is
approved, decision 5 and the roadmap ("Triplets", "Order of work") need the owner's update (decision D1 below).

## 1. Goal and how the result enters the paper

The triplet table has one empty cell: the Qwen3.8-27B backbone has a base row (`qwen3.8-27b-prompted`, 57.4 / 52.8
Avg / ZS nDCG@10) and decision fine-tunes (Kev-27B joint 61.1 / 57.8, the best system overall; Clef), but no reranker
fine-tune. On every smaller backbone (0.6B-9B) the reranker fine-tune wins (by 4-29 points on average). Kev-27B
only edges reranker fine-tunes a third of its size (Qwen3-Reranker-8B 59.4, Prism-9B 59.0), so the paper currently
cannot say whether decision fine-tuning catches up at 27B or whether a 27B reranker fine-tune would win again.

Our fine-tune `qwen3.8-27b-reranker` fills that cell: LoRA on **exactly Kev-27B's and Clef's starting checkpoint**
(Qwen/Qwen3.8-27B, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`), trained as a reranker, scored by the same
reader as Qwen3-Reranker and Prism on the same BM25 top-100. It is a test, not a foregone conclusion:

- If it beats Kev-27B joint, the headline extends from 0.6B-9B to 0.6B-27B.
- If it does not, the paper reports that at 27B decision fine-tuning matches or beats a reranker fine-tune trained
  with a public recipe -- a real qualification of the headline, provided the recipe is shown to be competitive
  (section 4). Either way the scope sentence of decision 5 changes.
- A caveat that stays: our reranker sees only MS MARCO (in-domain for DL19-23), Kev none. The zero-shot subset
  (TREC-COVID, Touché, DBPedia) is the fair comparison, as for the other triplets.

Paper changes once results exist: one row in `paper/tables/triplets.tex` (the 27B reranker column, via
`src/cdm/paper_tables.py`), the 9B validation row as a footnote or in the context table, two sentences in the setup
("our reranker fine-tune: LoRA, MS MARCO, recipe of RankLLaMA, validated against Prism-9B"), one in the limitations.
RQ2: if the loss includes the pointwise term (D3), the model has a meaningful P(yes) and enters the cutoff and
post-hoc tables like the other reranker fine-tunes.

## 2. Recipe (recommendation first, alternatives after)

| Choice | Recommendation | Why |
| --- | --- | --- |
| Starting checkpoint | Qwen3.8-27B post-trained release, rev `1d4bf0f2` | The checkpoint Kev-27B and Clef start from: the triplet differs only in training. No -Base of Qwen3.8-27B is public (Qwen3-Reranker starts from -Base, but that option does not exist here). |
| Prompt and readout | Qwen3-Reranker's system prompt, web-search instruction and `<Instruct>/<Query>/<Document>` layout; score z = z_yes - z_no at the first answer token | Our existing reader (`Qwen3Reranker`) scores it unchanged; trainer and reader share the input builder (`qwen3_reranker_ids`), so training sees exactly the scoring input. |
| Objective | **listwise + pointwise** (`--loss both`): InfoNCE over a group of 1 positive + 15 hard negatives, plus BCE of every z against its label | InfoNCE is the ranking objective of RankLLaMA (LoRA LLM reranker on MS MARCO) and SWIFT's listwise generative_reranker loss. InfoNCE alone is shift-invariant per group, so it leaves P(yes) = sigmoid(z) undefined; the BCE term (Qwen3-Reranker's documented pointwise loss) anchors it, so the model can enter RQ2 (cutoff p >= 0.5, ECE). Alternatives: `listwise` only (pure ranking recipe; RQ2 then reports its probabilities as not meaningful), `pointwise` only (monoT5/Qwen3-Reranker style; usually a little weaker at ranking). |
| Adaptation | LoRA r = 32, alpha = 64, dropout 0.05, on every linear layer of the language model (attention q/k/v/o, Gated DeltaNet in_proj_qkv/in_proj_z/out_proj, MLP); vision tower and MTP head untouched | RankLLaMA's rank and lr. Full fine-tuning of 27B needs ~16 bytes/parameter (bf16 weights + grads + fp32 Adam) = ~430 GB before activations: 8 x 80 GB with ZeRO-3 at best, not on our budget or tooling. LoRA: 55.5 GB frozen bf16 weights per GPU + ~3-4 GB LoRA state + a few GB of checkpointed activations -- ~65-67 GB, fits one 80 GB GPU (estimated from the 9B probe; the 27B probe is queued, section 6). |
| Data | MS MARCO passage train, 100k queries (of 400k eligible), 1 judged positive + 8 dense-retriever hard negatives (Tevatron/msmarco-passage-aug) + 7 BM25 negatives (Tevatron/msmarco-passage), passage text without title | Section 3. |
| Hyperparameters | lr 1e-4, linear decay, 3% warm-up, AdamW (no weight decay), grad-norm clip 1.0, 1 epoch, effective batch 64 groups (1,024 passages), max length 1,024 tokens (the reader's), bf16 autocast, gradient checkpointing, seed 42 | RankLLaMA-like (Ma et al., SIGIR 2024, arXiv:2310.08319: LoRA r 32, lr 1e-4, groups of 1 + 15 hard negatives, InfoNCE, 1 epoch on MS MARCO -- from memory, verify before citing); 1 epoch over 100k queries = 1,562 optimizer steps. Only the checkpoint (not these values) is selected on msmarco-dev (section 5): no hyperparameter search on a 27B model. |
| Parallelism | data parallel, 4 GPUs on one node, each with the full frozen model; LoRA gradients all-reduced per step | No model sharding needed; one all-reduce of ~200 MB per step. |

Not reproducible and therefore not attempted: Qwen3-Reranker's own data and multi-stage recipe (not public; the
Qwen3 Embedding technical report, arXiv:2506.05176, describes supervised fine-tuning on large labelled + synthetic
data followed by model merging -- verify before citing), and
Prism's (distils a commercial reranker API and five LLM judges; no public code or data found).

## 3. Training data

| Source (Hub id, pinned revision) | Content | Size | Licence | Use |
| --- | --- | --- | --- | --- |
| `Tevatron/msmarco-passage-aug` @ `d2f82a90` | MS MARCO passage train: query, judged positive(s), 30 hard negatives mined with a dense retriever (coCondenser per Tevatron's docs -- to verify) | 2.2 GB gz, 491,007 queries | card "unknown"; MS MARCO terms | queries, positives, 8 negatives per group |
| `Tevatron/msmarco-passage` @ `330a160b` | same queries, up to 30 BM25 negatives | 1.7 GB gz, ~400k queries (400,632 shared with the first set after the filter) | card apache-2.0; MS MARCO terms | 7 negatives per group |

MS MARCO is released by Microsoft for non-commercial research use; both Tevatron repos repackage it with passage
texts (no extra download of the collection needed). Both repos also contain `dl19`/`dl20` files -- the code reads
only `train.jsonl.gz`.

Considered and not chosen: `sentence-transformers/msmarco-hard-negatives` (ids only, 50 negatives from 13 retrievers
plus cross-encoder scores for denoising -- better negatives, but needs the 8.8M-passage collection and a join; worth
it if the validation run shows false-negative trouble); `webis/rank-distillm` (RankZephyr/monoELECTRA teacher
rankings for distillation; a different objective); `castorini/rank_zephyr_training_data` (listwise permutations
from GPT-4, for a listwise generator, not a pointwise reader); BGE/KaLM-style multi-task mixtures (closest to what
Qwen3-Reranker and Prism use, but tens of sources, mixed licences and BEIR training splits, e.g. NFCorpus/SciFact,
that would leak into our test sets).

Why mixed negatives: BM25 negatives are the candidate distribution we rerank (BM25 top-100); dense-retriever
negatives are harder and are what RankLLaMA-style rerankers train on. The mix halves the share of coCondenser
negatives, which contain false negatives (unjudged relevant passages; MS MARCO judges ~1 passage per query).

How many: 100k queries x 16 passages = 1.6M scored passages, 1,562 steps. RankLLaMA trained one epoch on all of MS
MARCO (~400k queries); large backbones are data-efficient and the 27B run is the cost driver (section 6), so we
take a quarter and check on msmarco-dev that the learning curve has flattened (checkpoints every 200 steps, i.e.
every 12.8k queries). The query sample is nested (`select_qids`: a larger sample contains the smaller), so a
200k-query set extends this one if the curve still rises at 100k.

**Leakage filter** (built in, `cdm finetune-data`): a training query is dropped if its id, its normalised text or
its bag of words equals that of any topic of DL19-23, the five BEIR sets or msmarco-dev (all 6,980 dev queries,
not only the 1,000 we rerank). Result for `msmarco-mix-100k` (`data/finetune/msmarco-mix-100k/manifest.json`): 197
of 491,007 train queries removed -- 12 DL21-23 queries (10 exact texts, e.g. "what is Morgellons disease" from DL22, and 2 word bags),
3 DL19/20 texts/word bags, 88 msmarco-dev duplicates, 1 Touché text, and 93 coincidental id matches with TREC-COVID
and SciFact (different id spaces, removed anyway). Passages are not filtered: DL19/20 share MS MARCO v1's collection
with the training data (as for monoT5, Qwen3-Reranker and Prism); what must not leak is a test query and its labels.

## 4. Recipe validation: is our recipe weak or strong?

A reviewer will ask whether a 27B result reflects the backbone or our recipe. Answer: run **the same recipe, same
data, same code on Qwen3.5-9B** (post-trained release, the `qwen3.5-9b-prompted` checkpoint) -> `qwen3.5-9b-reranker`,
and compare it on all ten datasets with **Prism-9B** (59.0 / 53.7), the published reranker fine-tune of that backbone.

Why 9B/Prism rather than Qwen3-8B/Qwen3-Reranker-8B:
- Same architecture family and code path as the 27B (Qwen3_5ForConditionalGeneration, hybrid Gated DeltaNet + full
  attention, fla and causal-conv1d kernels): the validation also validates the training code on that architecture.
  Qwen3-8B is a plain transformer.
- Same starting point type: post-trained release (as for the 27B). Qwen3-Reranker-8B starts from -Base, so a
  post-trained Qwen3-8B fine-tune would differ from it in two ways at once.
- The 9B triplet then has two reranker fine-tunes (Prism and ours) next to Kev-9B and Clef-Flash, which strengthens
  the headline at 9B regardless of the 27B outcome.
- Qwen3-Reranker-8B (59.4) is in the same table anyway as a second reference point.

Reading: within ~2 points of Prism-9B on the average (and clearly above the 9B decision fine-tunes, 50.1-54.7) =
competitive recipe; then the 27B result stands for "a reranker fine-tune of this backbone". Clearly below = our recipe
is the weaker one, and the 27B cell is reported as a lower bound (or the run is not started, D2). Expectation: lower
than Prism on the zero-shot BEIR sets (MS MARCO only vs. Prism's broad mixture), close on DL.

The 0.8B size (`qwen3.5-0.8b-reranker` vs Prism-0.8B) serves as the smoke-test size; a full 0.8B run is cheap
(~8 L40S-hours) and could be added as a third point of a recipe check (D6).

## 5. Selection on msmarco-dev only

No test split is used for any choice (CLAUDE.md). The only selected quantity is the checkpoint: every 200 steps an
adapter is kept (`checkpoints/<system>/checkpoint-<step>/`), and each is scored on the first 200 queries of the
cached msmarco-dev candidates (BM25 top-100, `bm25-d100-max1000`) with MRR@10 / nDCG@10:

```bash
slurm/submit.sh rerank qwen3.5-9b-reranker --datasets msmarco-dev --first-stage bm25-d100-max1000 \
    --max-queries 200 --adapter checkpoints/qwen3.5-9b-reranker/checkpoint-800 --variant ckpt800
```

(rows with `variant = ckpt<step>`, `subset = max200`, dataset msmarco-dev: never reported.) Rule fixed in advance: take
`final` (end of the linear decay) unless an earlier checkpoint is better by more than 1 point MRR@10; then point
`checkpoints/<system>/final` at it and log it. The 200 dev queries overlap with Part C's 1,000 fit queries -- both are
dev uses, no test data involved.

## 6. Compute

Measured in the smoke tests (`docs/experiment-log.md`, 2026-10-09):

| Probe (job) | GPU | Setting | Peak memory | s/step | Tokens/s real / padded |
| --- | --- | --- | ---: | ---: | ---: |
| 0.8B, 64 steps, stop + resume (4397261-4397262) | L40S 48 GB | 16 groups/step, 1 group/forward | 3.1 GiB | 4.5 | 9,000 / 13,000 |
| 9B, 24 steps (4398134) | L40 48 GB | 4 groups/step, 1 group/forward | 22.6 GiB | 9.3 (2.3 per group) | 1,090 / 1,620 |
| 9B, 12 steps (4398146) | L40S 48 GB | 2 groups/forward | 26.2 GiB | 9-12 | 860-1,140 / 1,440-1,800 |
| 9B, no gradient checkpointing (4398146) | L40S 48 GB | 1 group/forward | OOM (> 44 GiB) | -- | -- |
| 27B, 30 steps (4397306) | 80 GB | 4 groups/step | **queued** (Slurm's estimate: start Oct 13) | | |

The 9B probe reaches ~48% of the L40's dense bf16 peak (6 x 9B FLOPs per padded token: forward, recomputed forward,
backward to the activations; LoRA weight gradients are negligible). Two groups per forward pass do not help (more
padding across groups); without gradient checkpointing the 9B does not fit 44 GiB, so checkpointing stays on.
**27B memory, estimated (not measured):** 55.6 GB bf16 weights (incl. the unused vision tower) + ~210M LoRA
parameters x 16 bytes (fp32 weights, grads, Adam) = 3.4 GB + checkpointed activations ~6-8 GB -> ~65-67 GB on an
80 GB GPU with one group per forward pass.

Tokens per example: Qwen3-Reranker prompt + query + passage = 163 real tokens per passage on average (MS MARCO
passages ~58 words), ~235 tokens after padding within a group of 16 (padding costs ~30%; packing would save it, not
worth the risk now). One query group = 16 passages = ~3.8k processed tokens.

Estimate for the recommended data (100k groups x 16 passages x ~235 padded tokens = 376M padded tokens, 1,562
steps). A100 numbers scale the L40 measurement by dense bf16 peak (312 vs 181 TFLOPs, x1.6 at the same utilisation);
27B scales by FLOPs per token (x3.0 vs 9B). Uncertainty about +-30% until the 27B probe has run.

| Run | GPUs | Throughput (padded tokens/s, all GPUs) | GPU-hours | Wall time |
| --- | --- | ---: | ---: | ---: |
| 9B validation | 2 x A100-80GB | ~5,200 | ~40 | ~20 h (one 30 h job) |
| 27B | 4 x A100-80GB | ~3,450 | ~120 | ~30 h (one 48 h job) |
| 27B | 4 x H200 | ~8,000-9,000 (H200 ~2-2.5 x A100) | ~50 | ~12-15 h |
| 27B, 50k queries (if the 9B curve is flat by then) | 4 x A100-80GB | ~3,450 | ~60 | ~15 h |
| 0.8B (optional, D6) | 1 x L40S | ~13,000 | ~8 | ~8 h |

Plus checkpoint scoring on 200 msmarco-dev queries (20k pairs; ~10-20 min per 27B checkpoint on one 80 GB GPU,
~8 checkpoints) and the final scoring of the ten datasets (below). Total for this plan: roughly 200 GPU-hours on
80 GB GPUs (9B ~55 with scoring, 27B ~150 with scoring).

Slurm: partition `gpu`, max 48 h per job, account `chair-ds` (QOS normal, no per-user GPU cap found); jobs with 4 x
80 GB GPUs (A100-SXM4-80GB on rhaegal/starforge, 8 each; H200 141 GB on kuat2, 8) are allowed. The trainer stops
with a checkpoint 20 min (at most a tenth of the job) before the job's end (`SLURM_JOB_END_TIME`), and `--resume` continues with the same data
order, so a second chained job (`--after`) finishes a run longer than 48 h. **The real constraint is queueing**: on
2026-10-09 the 80 GB nodes were full (multi-day jobs of other users, an 8-GPU job waiting, our own ~60 rerank jobs);
Slurm estimated the start of a 1-GPU 45-minute probe four days out.

Disk: adapters ~0.9 GB each (r = 32 at 27B, fp32), 8 checkpoints + final ~8 GB; one optimizer state (~1.8 GB) kept
for resuming. 9B: ~0.3 GB per adapter. Under `checkpoints/` (gitignored).

Scoring afterwards (not part of this request): `qwen3.8-27b-reranker` on the ten datasets costs about what
`qwen3.8-27b-prompted` cost (bf16 27B, pointwise; ~25-30 GPU-hours on 80 GB GPUs), the 9B about 15 GPU-hours.

## 7. Framework

**Custom trainer (chosen), not ms-swift.** ms-swift 4.5.3 (latest on PyPI, 2026-09-08) requires
`transformers<5.17` and `peft<0.21`; our container has transformers 5.19 and peft 0.21.2, and Kev/Clef need
transformers >= 5.17. Installing it would mean a second container or `--no-deps` on an untested combination, for a
loss that is 20 lines. The trainer (`src/cdm/finetune.py`, ~350 lines) uses only torch, transformers and peft from
the container: PEFT LoRA, gradient checkpointing (non-reentrant), bf16 autocast, fused AdamW, manual data parallelism
via torchrun, checkpoint/resume, manifest via `cdm.provenance.write_manifest`. Smoke tests confirmed that training
Qwen3_5ForConditionalGeneration works with gradient checkpointing and the backward passes of flash-linear-attention
(Gated DeltaNet) and causal-conv1d (0.8B and 9B on L40/L40S), that the deadline stop and `--resume` continue the run
exactly (0.8B, sbatch path), that the trained adapter loads in the reader, and that the multi-rank path (gradient
all-reduce, rank-0 checkpoints, resume) works with 2 ranks on CPU (gloo). Not yet run: NCCL on 2+ GPUs (job 4397307,
queued) and the 27B itself (4397306, queued).

## 8. Risks and fallbacks

| Risk | Fallback |
| --- | --- |
| No 4 x 80 GB GPUs for days (queue) | Run on what is free: the trainer runs on any number of GPUs (same effective batch via gradient accumulation); 2 GPUs double the wall time, resume chains the jobs. Ask the cluster admins / pause our own rerank jobs for a slot. |
| Recipe too weak (9B well below Prism-9B) | Do not start the 27B; try one change on the 9B, selected on msmarco-dev only: better negatives (sentence-transformers hard negatives with cross-encoder denoising) or more data (200k). Otherwise report the 27B cell as missing, as now. |
| 27B run diverges or dev curve is flat | Checkpoints every 200 steps; lower lr (5e-5) on resume is a documented, dev-selected change. |
| Long BEIR passages (Touché ~860 tokens) unseen in training (MS MARCO ~160) | Same max length 1,024 as every reranker in the reader; the zero-shot subset will show it. Prism and Qwen3-Reranker saw longer training texts, so this caveat is ours alone. |
| False negatives in mined negatives | Mixed with BM25 negatives; sentence-transformers' cross-encoder scores as a denoising fallback. |
| bf16 score ties at 27B scoring time (Kanishka's 0.6B observation) | 27B is scored in bf16 like Kev-27B, Clef and `qwen3.8-27b-prompted`; check the tie rate on msmarco-dev first. |
| Paper timeline | See section 10; the 27B cell can be filled late as one table row. |

## 9. Decisions for the owner (recommendation first)

1. **D1 -- Run the 27B reranker fine-tune at all?** Recommended: yes, after the 9B validation (it turns the scope
   limitation into a result either way). Requires updating `docs/intent/paper-focus.md` decision 5 and the roadmap.
2. **D2 -- Order.** Recommended: 9B validation first; start the 27B once the 9B's msmarco-dev curve is sane (first
   checkpoints, ~day 1 of the 9B run), without waiting for the full 9B evaluation -- unless GPUs are scarce, then
   strictly sequential.
3. **D3 -- Loss.** Recommended: `both` (InfoNCE + BCE) so the model has a usable probability for RQ2. Alternative:
   `listwise` (pure ranking recipe, RQ2 leaves it out).
4. **D4 -- Data size.** Recommended: 100k queries, 1 epoch, mixed negatives 8 dense + 7 BM25; extend to 200k only if
   the 9B dev curve still rises at the end.
5. **D5 -- GPUs.** Recommended: 27B on 4 x A100-80GB (or H200) for ~1.5 days; 9B on 2 x 80 GB for ~1 day. Accept
   fewer GPUs and longer wall time if the queue demands it.
6. **D6 -- Validation scope.** Recommended: 9B vs Prism-9B only; optional 0.8B vs Prism-0.8B (~5 GPU-hours) as a
   second point.
7. **D7 -- System names.** `qwen3.8-27b-reranker`, `qwen3.5-9b-reranker` (registered, not in the grid yet).

## 10. Timeline (deadline Nov 17)

| When | What |
| --- | --- |
| Oct 10 | Owner's go (D1-D7); add the systems to `slurm/config.sh` and the triplet table code |
| Oct 10-13 | 9B validation run (~1 day on 2 GPUs, after queue wait); msmarco-dev checkpoint scores |
| Oct 12-16 | 27B run (~1.5 days on 4 GPUs, after queue wait), started once the 9B dev curve is sane |
| Oct 14-18 | Score 9B on ten datasets (~15 GPU-h), compare with Prism-9B; score 27B (~30 GPU-h) |
| Oct 19-24 | Tables, paired tests, text; RQ2 rows if D3 = both |
| Nov 17 | Submission; slack of ~3 weeks absorbs a queue delay of several days or one rerun of the 9B |

## 11. Commands once approved

Done already (preparation): data download and the training groups.

```bash
slurm/submit.sh prefetch --datasets none --systems none --training-data tevatron-aug,tevatron-bm25   # job 4395282
slurm/exec.sh --time 01:00:00 --mem 64G -- cdm finetune-data msmarco-mix-100k --queries 100000         # 4 min
```

After approval (from the repo root on the login node; commit first):

```bash
# 1. 9B validation run (2 x 80 GB from slurm/config.sh, 30 h budget); the second job only resumes if the first
#    stopped at its limit, otherwise it exits at once.
slurm/submit.sh finetune qwen3.5-9b-reranker --datasets msmarco-mix-100k --loss both
slurm/submit.sh finetune qwen3.5-9b-reranker --datasets msmarco-mix-100k --loss both --resume --after <job id>

# 2. Checkpoint selection on msmarco-dev (one job per checkpoint, 200 queries; steps 200, 400, ..., 1400)
for s in 200 400 600 800 1000 1200 1400; do
    slurm/submit.sh rerank qwen3.5-9b-reranker --datasets msmarco-dev --first-stage bm25-d100-max1000 \
        --max-queries 200 --adapter checkpoints/qwen3.5-9b-reranker/checkpoint-$s --variant ckpt$s --time 01:00:00
done
slurm/submit.sh rerank qwen3.5-9b-reranker --datasets msmarco-dev --first-stage bm25-d100-max1000 \
    --max-queries 200 --time 01:00:00                                                       # final

# 3. 27B run (4 x 80 GB, 48 h budget), same commands with qwen3.8-27b-reranker
slurm/submit.sh finetune qwen3.8-27b-reranker --datasets msmarco-mix-100k --loss both
slurm/submit.sh finetune qwen3.8-27b-reranker --datasets msmarco-mix-100k --loss both --resume --after <job id>

# 4. Scoring (after adding the systems to SYSTEMS in slurm/config.sh -- GPU tier and budgets are already there)
slurm/submit.sh rerank qwen3.5-9b-reranker --datasets all
slurm/submit.sh rerank qwen3.8-27b-reranker --datasets all
```

Fewer GPUs than configured: add `--gres gpu:2` (any count; the effective batch stays 64 groups through gradient
accumulation). Follow a run: `tail -f logs/<id>_finetune_*`, `checkpoints/<system>/train_log.jsonl`.

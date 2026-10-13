# Our reranker fine-tunes (9B, 27B): recipe and remaining steps

Code: `src/cdm/finetune_data.py`, `src/cdm/finetune.py`, `src/cdm/cli/finetune.py`,
`slurm/finetune.{sbatch,interactive.sh}`. Decisions: `docs/intent/paper-focus.md` 5 and 14. History (smoke tests,
the 9B validation, the training jobs): `docs/experiment-log.md`.

**State (2026-10-13).** `qwen3.5-9b-reranker` (*Ours-9B*): trained, checkpoint `checkpoint-800` selected on
msmarco-dev, scored on all ten datasets, in the paper. `qwen3.8-27b-reranker`: trained (job 4429626, 1-04:44:02;
checkpoints 200-1562 and `final` in `checkpoints/qwen3.8-27b-reranker/`); checkpoint selection and scoring are open
(sections 4 and 6).

## 1. Why

The 27B triplet has a base row and decision fine-tunes (Kev-27B joint, the best system overall; Clef) but no
published reranker fine-tune. On every smaller foundation model the reranker fine-tune wins. Our fine-tune fills that
cell with LoRA on **exactly Kev-27B's and Clef's starting checkpoint** (Qwen/Qwen3.8-27B, revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`), scored by the same reader as Qwen3-Reranker and Prism on the same BM25
top-100. If it beats Kev-27B, the headline extends to 0.6B-27B; if not, the paper reports that at 27B decision
fine-tuning matches or beats a reranker fine-tune trained with a public recipe (validated at 9B against Prism-9B).
Caveat: our reranker sees only MS MARCO (in-domain for DL19-23), Kev none; the zero-shot subset (TREC-COVID, Touché,
DBPedia) is the fair comparison. With the pointwise loss term the model has a meaningful P(yes) and enters the
calibration, cutoff and post-hoc tables like the other reranker fine-tunes.

## 2. Recipe

| Choice | Recommendation | Why |
| --- | --- | --- |
| Starting checkpoint | Qwen3.8-27B post-trained release, rev `1d4bf0f2` | The checkpoint Kev-27B and Clef start from: the triplet differs only in training. No -Base of Qwen3.8-27B is public (Qwen3-Reranker starts from -Base, but that option does not exist here). |
| Prompt and readout | Qwen3-Reranker's system prompt, web-search instruction and `<Instruct>/<Query>/<Document>` layout; score z = z_yes - z_no at the first answer token | Our existing reader (`Qwen3Reranker`) scores it unchanged; trainer and reader share the input builder (`qwen3_reranker_ids`), so training sees exactly the scoring input. |
| Objective | **listwise + pointwise** (`--loss both`): InfoNCE over a group of 1 positive + 15 hard negatives, plus BCE of every z against its label | InfoNCE is the ranking objective of RankLLaMA (LoRA LLM reranker on MS MARCO) and SWIFT's listwise generative_reranker loss. InfoNCE alone is shift-invariant per group, so it leaves P(yes) = sigmoid(z) undefined; the BCE term (Qwen3-Reranker's documented pointwise loss) anchors it, so the model can enter RQ2 (cutoff p >= 0.5, ECE). Alternatives: `listwise` only (pure ranking recipe; RQ2 then reports its probabilities as not meaningful), `pointwise` only (monoT5/Qwen3-Reranker style; usually a little weaker at ranking). |
| Adaptation | LoRA r = 32, alpha = 64, dropout 0.05, on every linear layer of the language model (attention q/k/v/o, Gated DeltaNet in_proj_qkv/in_proj_z/out_proj, MLP); vision tower and MTP head untouched | RankLLaMA's rank and lr. Full fine-tuning of 27B needs ~16 bytes/parameter (bf16 weights + grads + fp32 Adam) = ~430 GB before activations: 8 x 80 GB with ZeRO-3 at best, not on our budget or tooling. LoRA: 55.5 GB frozen bf16 weights per GPU + ~3-4 GB LoRA state + a few GB of checkpointed activations -- ~65-67 GB, fits one 80 GB GPU. |
| Data | MS MARCO passage train, 100k queries (of 400k eligible), 1 judged positive + 8 dense-retriever hard negatives (Tevatron/msmarco-passage-aug) + 7 BM25 negatives (Tevatron/msmarco-passage), passage text without title | Section 3. |
| Hyperparameters | lr 1e-4, linear decay, 3% warm-up, AdamW (no weight decay), grad-norm clip 1.0, 1 epoch, effective batch 64 groups (1,024 passages), max length 1,024 tokens (the reader's), bf16 autocast, gradient checkpointing, seed 42 | RankLLaMA-like (Ma et al., SIGIR 2024, arXiv:2310.08319: LoRA r 32, lr 1e-4, groups of 1 + 15 hard negatives, InfoNCE, 1 epoch on MS MARCO -- from memory, verify before citing); 1 epoch over 100k queries = 1,562 optimizer steps. Only the checkpoint (not these values) is selected on msmarco-dev (section 4): no hyperparameter search on a 27B model. |
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
MARCO (~400k queries); large backbones are data-efficient and the 27B run is the cost driver, so we
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

## 4. Selection on msmarco-dev only

No test split is used for any choice (CLAUDE.md). The only selected quantity is the checkpoint: every 200 steps an
adapter is kept (`checkpoints/<system>/checkpoint-<step>/`), and each is scored on the first 200 queries of the
cached msmarco-dev candidates (BM25 top-100, `bm25-d100-max1000`) with MRR@10 / nDCG@10:

```bash
slurm/submit.sh rerank qwen3.5-9b-reranker --datasets msmarco-dev --first-stage bm25-d100-max1000 \
    --max-queries 200 --adapter checkpoints/qwen3.5-9b-reranker/checkpoint-800 --variant ckpt800
```

(rows with `variant = ckpt<step>`, `subset = max200`, dataset msmarco-dev: never reported.) Rule fixed in advance: take
`final` (end of the linear decay) unless an earlier checkpoint is better by more than 1 point MRR@10; then enter
it in `SELECTED_CHECKPOINT` (`src/cdm/rerankers/__init__.py`) and log it. The 200 dev queries overlap with Part C's 1,000 fit queries -- both are
dev uses, no test data involved.

## 5. Framework

**Custom trainer (chosen), not ms-swift.** ms-swift 4.5.3 (latest on PyPI, 2026-09-08) requires
`transformers<5.17` and `peft<0.21`; our container has transformers 5.19 and peft 0.21.2, and Kev/Clef need
transformers >= 5.17. Installing it would mean a second container or `--no-deps` on an untested combination, for a
loss that is 20 lines. The trainer (`src/cdm/finetune.py`, ~350 lines) uses only torch, transformers and peft from
the container: PEFT LoRA, gradient checkpointing (non-reentrant), bf16 autocast, fused AdamW, manual data parallelism
via torchrun, checkpoint/resume, manifest via `cdm.provenance.write_manifest`. Smoke tests confirmed that training
Qwen3_5ForConditionalGeneration works with gradient checkpointing and the backward passes of flash-linear-attention
(Gated DeltaNet) and causal-conv1d (0.8B and 9B on L40/L40S), that the deadline stop and `--resume` continue the run
exactly (0.8B, sbatch path), that the trained adapter loads in the reader, and that the multi-rank path (gradient
all-reduce, rank-0 checkpoints, resume) works with 2 ranks on CPU (gloo).

## 6. Remaining commands (27B)

From the repo root on the login node; commit first.

```bash
# Checkpoint selection on msmarco-dev (one job per checkpoint, 200 queries), then final
for s in 200 400 600 800 1000 1200 1400; do
    slurm/submit.sh rerank qwen3.8-27b-reranker --datasets msmarco-dev --first-stage bm25-d100-max1000 \
        --max-queries 200 --adapter checkpoints/qwen3.8-27b-reranker/checkpoint-$s --variant ckpt$s
done
slurm/submit.sh rerank qwen3.8-27b-reranker --datasets msmarco-dev --first-stage bm25-d100-max1000 --max-queries 200

# Scoring, after adding the system to SYSTEMS in slurm/config.sh (GPU tier and budgets are already there)
slurm/submit.sh rerank qwen3.8-27b-reranker --datasets all
```

Then the msmarco-dev fit for post-hoc scaling (`cdm posthoc`, as for the 9B fine-tune) and `make tables` in `paper/`.
Follow a run: `tail -f logs/<id>_*`.

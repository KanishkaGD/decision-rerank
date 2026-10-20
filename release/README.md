# Decision Models as Rerankers

Code and results for the ECIR 2027 short paper submission *Do System One Models Make Better Rerankers? Decision vs.
Reranker Fine-Tuning on One Backbone*.

System One decision models (Jev, Clef, Kev, Laya) answer a typed question about an input with a calibrated
probability. We ask whether they make better rerankers than reranker fine-tunes of the same backbone (RQ1), and
whether their calibration is of use (RQ2). Every number in the paper's tables is generated from `results/` by
`src/cdm/paper_tables.py`; this repository holds the code that produced those results.

## Contents

```
src/cdm/            the `cdm` package and command line
  first_stage.py      BM25 top-100 candidates (Pyserini; the official top-100 files for TREC DL 2021-2023)
  rerankers/          every system of the paper (registry with pinned model revisions in __init__.py)
  phrasings.py        the question asked of the System One models (S1Rank's, verbatim)
  evaluation.py       nDCG@10, ECE (15 bins), Brier on the judged pairs
  significance.py     paired t-tests over queries, Holm-corrected across the ten collections (RQ1)
  cutoff.py           the fixed cutoff p >= 0.5, its regret against the oracle cutoff, the count test (RQ2)
  posthoc.py          temperature and Platt scaling fitted on MS MARCO dev (RQ2)
  finetune_data.py,   our reranker fine-tune of Qwen3.5-9B ("Ours-9B")
  finetune.py
  paper_tables.py     results/*.csv -> tables/*.tex
results/results.csv   one row per system, collection and analysis (all metrics; see below)
results/literature.csv published numbers on the same candidates, used where we did not rerun a system
tables/               the generated LaTeX tables
tests/                unit tests (pytest; no GPU, no downloads)
container/            Apptainer definitions: cdm.def (everything), lightning-ir.def (the Set-Encoder only)
requirements-frozen*.txt  the exact package versions of the two images the results were computed with
run_all.sh            every step from downloads to tables
```

## Reproduce the tables (no GPU)

The tables need only the Python standard library:

```bash
PYTHONPATH=src python -m cdm.paper_tables      # -> tables/{triplets,context-short,calibration}.tex (in the paper)
                                               #    and tables/{context,calibration-full}.tex (supplementary)
```

`triplets.tex` is Table 1 (RQ1), `context-short.tex` the other systems, `calibration.tex` RQ2 (ECE and Brier).
The paper states some results without numbers for space; they are in the supplementary tables: `context.tex` has
every other system per collection, `calibration-full.tex` the fixed cutoff (F1 at p >= 0.5, regret, set size), the
count test and post-hoc temperature/Platt scaling.

## Reproduce the results

Environment: build the image (`apptainer build container/cdm.sif container/cdm.def`, and
`container/lightning-ir.sif` from `lightning-ir.def` for the Set-Encoder, which needs transformers < 5), or install
`pyproject.toml` with Python 3.12 (`pip install -e .`, then Kev, flash-linear-attention and causal-conv1d as in
`container/cdm.def`; `requirements-frozen.txt` has the versions we used). Copy `.env.example` to `.env` and fill in a
Hugging Face token and, for Jev, a TypeSafe API key.

```bash
bash run_all.sh                      # all steps; or one at a time:
bash run_all.sh prefetch             # indexes, topics, qrels, official pools, models, fine-tuning data (needs internet)
bash run_all.sh first-stage          # BM25 top-100 of the ten test collections and of 1,000 MS MARCO dev queries
bash run_all.sh finetune             # Ours-9B: LoRA on Qwen3.5-9B, 2 GPUs; checkpoint selection on MS MARCO dev
bash run_all.sh rerank               # every system on every collection (GPU; Jev via API)
bash run_all.sh analyses             # significance, cutoff and count test, post-hoc scaling (CPU, stored outputs)
bash run_all.sh tables
CDM_CONTAINER=1 bash run_all.sh ...  # the same inside the Apptainer images
```

Single runs: `python -m cdm.cli rerank kev-4b-joint dl19` (`--max-queries 20 --no-save` for a quick check;
`python -m cdm.cli systems` and `... datasets` list what is registered). Each run stores every candidate's score and
probability, per-query metrics, the reranked run and a `manifest.json` (code revision, command, package versions,
model revisions, hardware) under `outputs/<dataset>/<first stage>/<system>/<phrasing>/`, and its metrics as one row
of `results/results.csv`. Seeds are fixed (42).

Compute: we ran on single L40/L40S (48 GB), A100 (80 GB) and H200 GPUs (48 GB suffice for most systems; the 27B
models and the 8B/9B models in fp32 need 80 GB; the `device` column records each run's GPU) and trained Ours-9B on
two H200s for one epoch (1,562 steps). The adapter is not part of this repository; `run_all.sh finetune` recreates
it. The rerankers run in fp32 with TF32 matmuls
unless a model was trained in bf16 (Kev-27B, Clef, Qwen3.8-27B): in bf16 many scores tie within a query.

## Setup in brief

- **Candidates**: BM25 top-100 (Pyserini, k1 0.9, b 0.4) on TREC DL 2019-2023 (passage) and five BEIR collections
  (TREC-COVID, NFCorpus, SciFact, Touché, DBPedia) -- the candidates S1Rank, RankGPT, RankZephyr and Abdallah et al.
  (2025) rerank. For DL 2021-2023 these are the tracks' official top-100 files.
- **Question**: all System One models get S1Rank's question (`src/cdm/phrasings.py`), fixed in advance, with a state
  that holds all candidates (`<model>-joint`) or one (`<model>-isolated`); both modes are pointwise.
- **Triplets**: per backbone, the base model asked zero-shot (`*-prompted`), its decision fine-tunes (Kev, Clef) and
  its reranker fine-tunes (Qwen3-Reranker, Prism, Ours-9B).
- **No tuning on test data**: post-hoc calibration is fitted on MS MARCO dev, and Ours-9B's checkpoint was chosen on
  MS MARCO dev.
- **Published numbers**: where a system was not rerun on the same candidates (RankGPT-4; RankZephyr on BEIR; bge,
  monoT5-3B and the Set-Encoder on some collections), the cell comes from `results/literature.csv` (with its source)
  and is marked with a dagger in the tables. We reran several of them as checks; those rows are in `results.csv`.

## results.csv

One row per configuration; the key is `job, system, model, dataset, first_stage, phrasing, variant, subset`.

| `job` | what | main columns |
| --- | --- | --- |
| `first-stage` | BM25 itself | `nDCG@10`, `R@100` |
| `rerank` | a reranker on a collection | `nDCG@10`, `RR@10`, `AP`, `ECE`, `Brier` (judged pairs), `ECE_unjudged0` (unjudged as non-relevant), latency, tokens, API cost |
| `significance` | paired t-test against `sig_other` | `sig_comparison`, `sig_delta`, `sig_p`, `sig_holm` (1: significant after Holm) |
| `cutoff` | fixed cutoff and count test (TREC DL) | `cut_F1@0.5`, `cut_regret`, `cut_oracle_tau`, `cut_size@0.5`, `topk_F1@10`, `count_MAE` |
| `posthoc` | native / temperature / Platt (`variant`) | `ECE`, `cut_F1@0.5`, `posthoc_a`, `posthoc_b`, `fit_*` |

`phrasing` is `s1rank` for System One models and the base models and `none` otherwise. `variant` `a100` marks a
second run of a system on the same candidates (efficiency measurements on one GPU type; also the source of the
probabilities where the default cell is a published number), `ckpt<N>` the checkpoint selection of Ours-9B on 200
MS MARCO dev queries (`subset` `max200`).

## Tests

```bash
pip install pytest click numpy pandas scipy ir-measures && pytest
```

## License

MIT (code). Model weights and datasets keep their own licenses.

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

## 2026-10-06 -- Cluster bring-up

Bring-up sequence from `docs/handoff.md` run against FAITH; everything works after two fixes.

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

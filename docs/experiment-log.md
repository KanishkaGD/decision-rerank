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

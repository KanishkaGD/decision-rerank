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

## 2026-10-05 -- Repository set up

Repository scaffolded from the research plan (`docs/research-plan.md`): cluster wrappers (`cluster/`), Slurm jobs
(`slurm/`), container (`container/`), package `cdm` with BM25 first-stage caching, cross-encoder reranking and
evaluation (WP1 starters), LNCS paper skeleton (`paper/`). Laya and Jev are stubs until WP0.

Not yet verified on the cluster: container build, Pyserini index names, `gpumem.40gib` feature, `--time` budgets.
First steps: `cluster/connect.sh`, `cluster/login.sh bash container/build.sh`, `cluster/submit.sh hello`.

Open decisions for the project owner (from the plan): target venue and deadline; compute and API budget;
MIRACL in or out; RQ5 in scope or not.

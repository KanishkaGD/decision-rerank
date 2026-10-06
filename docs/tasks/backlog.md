# Backlog

Open work nobody has claimed. Claim with `scripts/task.sh start <id> "<description>"` and remove the line in that
task; add lines for work you discover but don't do. Active tasks: `scripts/task.sh list`. Order = priority.

- `vm-bringup` -- Run docs/vm-setup.md on the VM, then one `cdm rerank jev scifact --first-stage bm25-d100-max20
  --phrasings s1rank --no-save` from a task worktree to prove the whole chain; log it.
- `prefetch-all` -- `cluster/login.sh bash slurm/prefetch.sh --datasets all --systems minilm-l6,bge-m3,mxbai-base`
  (checks the remaining Pyserini names: dl19, dl20, msmarco-dev, trec-covid, nfcorpus, fiqa, scidocs); fix names.
- `first-stage-all` -- `cluster/submit.sh first-stage --datasets all`; record run times and correct the `--time`
  budgets in `slurm/config.sh` with job ids.
- `wp2-dl` -- All systems on DL19/DL20 (main calibration sets: judged non-relevant documents), then the BEIR subset.
  Replicate S1Rank's Jev numbers (scifact 80.5, DL19 73.7) as the sanity check. Ask before API runs over ~$10.
- `wp2-go-no-go` -- From those results: is zero-shot Laya far below MiniLM (it was below BM25 on 20 scifact
  queries)? Write the decision up for the owner **before any WP4 work** (research plan).
- `wp0-landscape-note` -- `docs/notes/wp0-landscape.md`: S1Rank (joint vs isolated, window sizes, position effects,
  calibration by prevalence, negative adaptive routing), the Decision Index (BRIGHT/ToolRet), Laya and Clef model
  cards; overlap with RQ2-RQ4. Notes so far are in `docs/experiment-log.md` (2026-10-06).
- `paper-acm` -- Move `paper/` from LNCS to ACM sigconf (WSDM 2027 short paper, 4 pages, anonymous; deadline
  Nov 17, 2026, `docs/intent/venue.md`).
- `cdm-table` -- A `cdm table` command that writes `paper/tables/*.tex` from `results/results.csv`, with the
  calibration tables restricted to datasets with `n_judged_nonrel_pairs > 0`.
- `open-questions` -- Ask the owner open questions 4-17 from the first handoff (one at a time; 1-3 are answered,
  `docs/intent/`).
- `clef-27b` -- Prefetch `Cloudflare/clef` (55 GB) and run it on an 80 GB GPU, if Clef-Flash looks worth it.

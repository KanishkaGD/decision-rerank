# Handoff

Written 2026-10-06, end of the first Claude Code session (on Michael's laptop). Work continues on a shared VM with a
colleague. Read in this order: this file, `CLAUDE.md`, `docs/collaboration.md`, `docs/vm-setup.md`,
`docs/research-plan.md` (the owner's plan, never edit), `README.md`. History of decisions:
`docs/experiment-log.md` and `docs/intent/`.

## Where the project stands

- **Venue:** arXiv preprint before Nov 17, 2026, plus a WSDM 2027 short paper (4 pages, ACM sigconf, anonymous,
  deadline Nov 17) -- `docs/intent/venue.md`. The paper is still the LNCS skeleton (backlog: `paper-acm`).
- **Systems (all working, trial runs only):** BM25 (Pyserini) first stage; cross-encoders MiniLM-L6, bge-m3,
  mxbai-base; one `SystemOneReranker` for Jev (TypeSafe API, `jev-1.13.0`), Laya (`convaiinnovations/laya`, open
  weights) and Clef (`Cloudflare/clef-flash`, `Cloudflare/clef`), all asked byte-identical requests in S1Rank's
  format (`src/cdm/rerankers/system_one.py`). Open-weights models are pinned to Hugging Face commits.
- **First numbers** (20 scifact queries, BM25 top-100, phrasing `s1rank`, not saved; `docs/experiment-log.md`):
  BM25 0.614, MiniLM 0.674, Jev 0.759, Clef-Flash 0.666, Laya 0.374 nDCG@10. Laya below BM25 is an early WP2
  no-go signal; confirm on TREC DL.
- **Evaluation:** calibration is reported on judged pairs (main results; `n_judged_nonrel_pairs` says whether a set
  has judged negatives) and with unjudged documents as non-relevant (`ECE_unjudged0`, secondary) -- owner's decision.
- **Cluster:** FAITH bring-up done (container builds on the login node; GPU feature is `gpumem.48gib`, 80 GB for
  the 27B Clef). Each checkout syncs to its own `checkouts/<user>-<worktree>/` on `/shared`; data, results, logs and
  the container are shared (tested 2026-10-06).
- **Open questions answered:** 1-3 (`docs/intent/`). Questions 4-17 of the first handoff are still open (backlog:
  `open-questions`); ask them one at a time.

## What the owner has to do (agents can't)

1. Run `docs/vm-setup.md` on the VM (one shared Linux user, a profile per person). Step 3 creates the Kerberos
   keytab and needs the FAITH password, typed into `ktutil` by the owner; never give it to an agent. Step 4: each
   person creates their profile and logs in to GitHub and Claude Code with it.
2. Copy `.env` (TypeSafe key `TYPESAFE_AI_API_KEY`, optional `HF_TOKEN`) from the laptop to `/srv/cdm/repo/.env`.
3. Decide how `dev-michael` and the unrelated `master` of `KanishkaGD/decision-rerank` are combined; until then the
   VM's `main` tracks `dev-michael` (`git config cdm.remoteBranch`). `master` moved on 2026-10-06 (to `66a7a7b`),
   so someone works there; agree before merging.
4. Check that sharing the FAITH account `dinzinger` with the colleague's agent is allowed, or get a project account.
5. Each person logs in to Claude Code with their own subscription through their profile and checks it with `/status`
   (docs/vm-setup.md, step 4).
6. Remove the stray test checkout `/shared/dinzinger/calibrated-decision-models-for-ir/checkouts/michael-alice-two`
   on the cluster (left by a guard test; harmless).

## Next steps

`docs/tasks/backlog.md`, in order: `vm-bringup`, `prefetch-all`, `first-stage-all`, `wp2-dl` (all systems on
DL19/DL20, replicate S1Rank), `wp2-go-no-go` (report to the owner before WP4), `wp0-landscape-note`, `paper-acm`,
`cdm-table`, `open-questions`, `clef-27b`.

## Known gaps

- `cdm table` (paper tables from `results.csv`) doesn't exist yet; significance tests exist in
  `cdm.evaluation` but no CLI uses them.
- RQ3 analyses (adaptive depth, no-relevant-document AUROC, QPP) and RQ4 ablations (order, set size, distractors,
  random neighbours, the Decision Index's candidates-in-questions format) are not implemented.
- Only the Noul primitive is implemented for System One models (Laya's Noul has a known label-following issue,
  #156 in its repo; WP3 should compare a two-option Choice).
- pytest is not in the container; `scripts/check.sh` runs the tests locally if possible, else in the container with
  pytest from `data/devtools` on the cluster.
- Old `checkouts/` and `snapshots/` on `/shared` are never cleaned up automatically.

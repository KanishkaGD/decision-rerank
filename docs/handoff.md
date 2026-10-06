# Handoff to Claude Code

Written 2026-10-05 at the end of the Cowork setup session. Read this first, then `CLAUDE.md` (conventions, must
follow), `docs/research-plan.md` (the owner's plan, do not edit) and `README.md` (commands).

## Where the project stands

- **Repository scaffolded, nothing run on the cluster yet.** Initial commit `55a7c44`. The repo is delivered as
  plain files plus `repo-setup.bundle`. If `git status` says this is not a repository, the owner hasn't
  initialised it yet. Ask them to run:
  `git init -b main && git fetch repo-setup.bundle main && git reset --hard FETCH_HEAD && rm repo-setup.bundle`.
  This also restores `paper/Makefile` and the executable bits of the scripts. Then commit this file
  (`docs/handoff.md`).
- **Code (`src/cdm/`):** BM25 first-stage caching (Pyserini), three cross-encoder rerankers, evaluation (nDCG,
  recall, RR, AP, ECE, Brier, latency, truncation), a locked `results/results.csv`, and a `manifest.json` per
  output directory.
  - **Laya and Jev are stubs** (`src/cdm/rerankers/system_one.py`) that raise `NotImplementedError`.
  - The phrasings in `src/cdm/phrasings.py` are a **draft** (none is `frozen`).
- **Paper (`paper/`):** LNCS skeleton with one file per section, the five RQs in `rq-macros.tex`, and
  `sources.bib` seeded with the plan's prior work. It compiles; the content is all `\TODO`/`\PENDING` markers.
- **Verified so far:**
  - `bash -n` passes on every script.
  - The `cluster/` scripts were tested against a simulated ssh/rsync/Slurm setup.
  - The unit tests pass, with ir_measures replaced by a stand-in.
  - The paper builds.
- **Not verified, check on the first real runs:**
  - the container build on the login node (`apptainer build --fakeroot`);
  - every Pyserini index, topic and qrels name in `src/cdm/datasets.py`;
  - the cross-encoders under the real sentence-transformers (`activation_fn` argument);
  - whether the GPU feature `gpumem.40gib` exists (`cluster/login.sh sinfo -N -o "%N %G %f"`);
  - the `--time` budgets in `slurm/config.sh`, which are first guesses.

## How you work here (summary of CLAUDE.md)

- You run on the owner's machine. You reach the FAITH cluster only through `cluster/*.sh`, which send commands over
  one shared SSH connection. **Only the owner can open it**, with `cluster/connect.sh`, because the cluster takes a
  password or Kerberos, not keys. If a script says "no connection", stop and ask. Never run work on the login node
  beyond light commands.
- What each command does:
  - `cluster/exec.sh [--gpu] -- <cmd>`: quick blocking run on a compute node.
  - `cluster/submit.sh <job> ... [--dry-run]`: batch jobs.
  - `cluster/status.sh`, `cluster/logs.sh <id>`: follow jobs.
  - `cluster/fetch.sh`: copies `results/` back to the repo; commit it afterwards.
- Code is synced on every call. Commit before runs whose numbers go into the paper.
- Rebuild the container only after changing `pyproject.toml`.
- Downloads only via `slurm/prefetch.sh` on the login node; jobs run offline.

## Decisions made

| Date | Decision |
| --- | --- |
| 2026-10-05 | Code reaches the cluster by rsync of the working tree, with a per-submission snapshot of `src/` for provenance. |
| 2026-10-05 | Paper template LNCS for now. |
| 2026-10-05 | **Target: an arXiv preprint, with no fixed deadline**, so that all important experiments can be run. The ECIR 2027 short-paper deadline (Oct 12) was considered and dropped. Whether a peer-reviewed venue follows is still open (question 1 below). |

## Open decisions: ask the owner, one at a time, without assuming the answer

The owner asked to be **asked** these, not given guesses. Give the facts each decision needs, then ask. Record
each answer in `docs/intent/` and in `docs/experiment-log.md`.

**These block WP0/WP1, ask them first:**

1. arXiv only, or arXiv as a step towards a venue (SIGIR 2027: Jan 21, 2027, ACM 9 pages; CIKM 2027 about May
   2027; ECIR 2028 about Oct 2027)? This decides the template, the page budget and whether to anonymise.
2. Target month for the preprint. The plan's top risk is being scooped on RQ3/RQ4.
3. Jev: API access or an application? What API budget?
4. Laya: which checkpoints (English, multilingual, enterprise) and primitives (Noul, Score, Choice)? Does the
   owner already know the API, the probability output and a joint mode?
5. GPU-hour budget on FAITH, and any priority or QOS limits.

**Scope:**

6. Which RQs are in (RQ1–RQ4 core, RQ5 stretch; RQ2 is the most expensive)?
7. MIRACL in or out, and which languages?
8. Which dense retriever for the robustness check, or BM25 only?
9. Is the cross-encoder line-up final, or should newer rerankers be checked in WP0?
10. Include an LLM listwise reranker (e.g. RankZephyr) as a quality ceiling?

**At specific work packages:**

11. (WP2) The numeric go/no-go threshold: how far below MiniLM is "far below"?
12. (WP3) Which dev set, how many phrasings, and does the owner write or approve them?
13. (WP4) Full fine-tuning or LoRA, how much training data, which calibration-preserving loss?
14. Which fixed GPU type for latency, and is CPU latency measured too?
15. Is there a contamination-free test set (post-2026 or private)?

**Release and people:**

16. Co-authors and roles.
17. Where code and raw outputs are released, and under which licence.

## Next steps

1. With the owner: initialise git (see above), copy `cluster/ssh_config.example` into `~/.ssh/config`, and run
   `cluster/connect.sh`.
2. Bring-up:

   ```bash
   cluster/login.sh bash container/build.sh
   cluster/login.sh bash slurm/prefetch.sh --datasets scifact --systems minilm-l6
   cluster/submit.sh hello
   cluster/exec.sh --gpu -- cdm info
   cluster/exec.sh -- cdm first-stage scifact --max-queries 20
   cluster/exec.sh --gpu -- cdm rerank minilm-l6 scifact --first-stage bm25-d100-max20 --no-save
   ```

   Fix whatever breaks. Note every fix and every corrected `--time` budget (with the job id) in
   `docs/experiment-log.md`.
3. Ask open questions 1–5.
4. WP0:
   - Search for System One reranking work newer than the plan.
   - Verify Laya (API, checkpoints, licence, fine-tuning notebook) and Jev access.
   - Write `docs/notes/wp0-landscape.md`, flagging any overlap with RQ2–RQ4.
   - Then implement `LayaReranker` (contract in the docstring of `system_one.py`) and register the real model
     id in `src/cdm/rerankers/__init__.py`.
5. WP1/WP2:
   - `cluster/submit.sh first-stage --datasets all`, then every reranker with `--after <ids>`.
   - Replicate one S1Rank number for Jev if access exists.
   - Report the go/no-go decision to the owner **before starting WP4**.

## Known gaps in the code

- There is no `cdm table` command yet to generate `paper/tables/*.tex` from `results.csv`. The rule is that
  numbers in the paper are never typed by hand.
- No significance testing across systems yet: `evaluation.paired_bootstrap` and `holm` exist, but no CLI uses them.
  Per-query metrics are stored in `outputs/.../perquery.csv`.
- Not implemented yet:
  - the RQ3 analyses: adaptive depth, no-relevant-document detection (AUROC), QPP;
  - the RQ4 ablations: order, set size, distractors, random neighbours;
  - fine-tuning (WP4), a dense first stage, MIRACL, BRIGHT.
- Old `snapshots/` directories on the cluster are never cleaned up automatically.

## Where else context lives

- The claude.ai project "Calibrated Decision Models for IR" has `claude/repository-setup.md`, a setup summary.
- The skill `md-setup-research-project` was proposed to the owner; it may or may not have been saved. It describes
  the conventions this repo follows.
- The original plan file at the repo root
  (`Research plan Calibrated System One decision models as rerankers (Laya Jev).md`) is an untracked duplicate of
  `docs/research-plan.md`. The owner can delete it.

# Collaboration and infrastructure

Decided by the owner on 2026-10-06:

- **Work moves to a shared VM** inside the university network, used by Michael and a colleague and their coding
  agents, in one repository. Jobs are submitted to FAITH directly from the VM.
- **No manual ticket renewal**: access must keep working without anyone logging in every few hours.
- **All jobs run under Michael's FAITH account** (`dinzinger`); the colleague's agent submits as Michael too. The
  owner was told that a keytab is password-equivalent and that sharing an account may conflict with FAITH's terms;
  checking that, or requesting a project account, is the owner's call.
- **One git worktree per agent/task, merged back to main as soon as a task is finished and works**, so no two versions
  drift apart for long.
- **One Linux user on the VM for both people** (decided later the same day, after being told what that shares):
  per-person profiles keep names, git identity, Claude Code and GitHub logins apart; claims are owned by a person and
  the guard keeps agents out of the other person's tasks. Nothing is private between the two on the VM.
- Code is hosted at `https://github.com/KanishkaGD/decision-rerank` (private); Michael's history goes to branch
  `dev-michael` for now. How it combines with that repository's unrelated `master` is open.

Implemented in `docs/vm-setup.md`, `docs/collaboration.md`, `scripts/task.sh`, `scripts/agent_guard.py`,
`cluster/vm/`.

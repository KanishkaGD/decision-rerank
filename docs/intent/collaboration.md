# Collaboration

Decided by the owner on 2026-10-07, replacing the shared-VM plan of 2026-10-06 (task claims, worktrees, guard hook,
one Linux user, Kerberos keytab -- all removed; see the git history up to commit cdd8ed8):

- **Ordinary branch-based collaboration.** Michael and his colleague work on feature or dev branches and merge their
  changes into `master` regularly.
- Code is hosted at `https://github.com/KanishkaGD/decision-rerank` (private). Michael's history is on `dev-michael`;
  `master` holds the colleague's separate codebase (package `decisionrerank`) with an unrelated history. How the two
  are combined in `master` is still open.

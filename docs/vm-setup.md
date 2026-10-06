# Setting up the shared VM

Goal: Michael, a colleague and their coding agents work on one VM inside the university network, in one repository,
and submit to FAITH from there -- with no password prompts and no tickets to renew by hand.

How it works: FAITH accepts only password or Kerberos logins (no SSH keys). A **Kerberos keytab** for Michael's FAITH
account (decided 2026-10-06, `docs/intent/collaboration.md`) lets a cron job get a
fresh ticket every 30 minutes, and `cluster/lib.sh` opens the shared SSH connection on its own whenever it is missing.
Nobody types the password after step 3, and agents never see it.

**Security, read first.** The keytab is password-equivalent: whoever can read it can log in to FAITH as Michael. It is
readable only by members of the `cdm` group on the VM (and by the VM's root). Everything submitted from the VM runs
under Michael's FAITH account, including the colleague's jobs -- check that FAITH's terms of use allow this, or ask
the FAITH admins for a project account (then repeat step 3 with that principal). Changing the FAITH password
invalidates the keytab: re-run step 3.

## Steps

Prerequisites on the VM: Ubuntu/Debian with `git`, `rsync`, `python3`, `jq`, `krb5-user`, `cron`, `openssh-client`
(`sudo apt install git rsync python3 python3-venv jq krb5-user cron openssh-client`; when `krb5-user` asks for the
default realm, enter `FIM.UNI-PASSAU.DE`). The VM must reach `freia.fim.uni-passau.de` and the KDCs
(`ilium`, `ischii`, `neurocranium.fim.uni-passau.de`, port 88; found through DNS, no `krb5.conf` edits needed).
Install Claude Code for each user as usual.

1. **Shared repository** (Michael, with sudo; from any copy of this repo, e.g. a temporary clone):

   ```bash
   sudo cluster/vm/setup-repo.sh https://github.com/KanishkaGD/decision-rerank.git \
       --remote-branch dev-michael --users "michael <colleague>"
   ```

   Creates group `cdm`, `/srv/cdm/{repo,worktrees,secrets}`, a group-writable clone on `main` that tracks
   `origin/dev-michael`, and sets `cdm.mainBranch`, `cdm.remote`, `cdm.remoteBranch` for `scripts/task.sh`.
   When the team decides which GitHub branch is canonical (see "Open" below), change
   `git -C /srv/cdm/repo config cdm.remoteBranch <branch>`. Both users log out and in again (group membership).

2. **SSH config** (each VM user):

   ```bash
   /srv/cdm/repo/cluster/vm/setup-ssh.sh
   ```

   Adds `Host faith` (user `dinzinger`, Kerberos, connection sharing, keepalives, no ticket forwarding). Copy the
   `@cert-authority freia.fim.uni-passau.de,frigga.fim.uni-passau.de ...` line into `~/.ssh/known_hosts` (from the
   FAITH docs or `grep cert-authority ~/.ssh/known_hosts` on Michael's laptop).

3. **Keytab and automatic tickets** (Michael, once; asks for the FAITH password inside `ktutil`):

   ```bash
   /srv/cdm/repo/cluster/vm/setup-kerberos.sh
   ```

   Writes `/srv/cdm/secrets/faith.keytab` (0640, group `cdm`), installs the cron job for Michael's VM user, gets a
   ticket and tests a password-less login. If `ktutil` fails to make a working keytab, ask the FIM IT
   (`https://www.fim.uni-passau.de/it-dienste/`) for one.

4. **Tickets for the colleague's VM user** (the colleague, once; no password):

   ```bash
   /srv/cdm/repo/cluster/vm/setup-kerberos.sh --cron-only
   ```

5. **Secrets and local settings** (Michael, in `/srv/cdm/repo`): copy `.env` (TypeSafe key, HF token) from the laptop
   to `/srv/cdm/repo/.env` and `chmod 640 .env; chgrp cdm .env`. `scripts/task.sh start` links it into every task
   worktree. Optionally `.claude/settings.local.json` (personal permission rules) the same way.

6. **Check** (each user):

   ```bash
   cd /srv/cdm/repo
   cluster/connect.sh --check || cluster/login.sh --no-sync hostname   # connects without a prompt
   scripts/task.sh list
   ```

   The first `cluster/*.sh` call from a new checkout syncs it to its own
   `/shared/dinzinger/calibrated-decision-models-for-ir/checkouts/<user>-<worktree>/` and links the shared data,
   results, logs and container into it (cluster/cluster.env).

Then start Claude Code in `/srv/cdm/repo` and claim a task (`docs/collaboration.md`); the agent works in the task's
worktree under `/srv/cdm/worktrees/<id>`.

## Maintenance

- **Password change:** re-run step 3; the cron jobs keep working with the new keytab.
- **Ticket problems:** `klist` (expiry), `kinit -k -t /srv/cdm/secrets/faith.keytab dinzinger@FIM.UNI-PASSAU.DE`
  (manual refresh), `crontab -l | grep cdm-kerberos` (job installed?).
- **Old checkout directories** on `/shared/.../checkouts/` are not removed automatically; delete those of finished
  tasks now and then (`cluster/login.sh --no-sync ls ../`), like the old `snapshots/`.

## Open

- The GitHub repository `KanishkaGD/decision-rerank` has its own `master` with a history unrelated to this one.
  Until the team decides how to combine them, the VM's `main` tracks `dev-michael`. Merging the two histories
  (`git merge --allow-unrelated-histories`) is a one-time decision for the owners, not for an agent.

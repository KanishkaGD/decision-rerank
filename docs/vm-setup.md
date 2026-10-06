# Setting up the shared VM

Goal: Michael, a colleague and their coding agents work on one VM inside the university network, in one repository,
and submit to FAITH from there -- with no password prompts and no tickets to renew by hand.

**One Linux user for everyone** (decided 2026-10-06, `docs/intent/collaboration.md`): both people log in to the VM as
the same Linux user (below: `cdm`; any name works). What would otherwise be shared is kept apart by a **profile per
person** (`~/people/<name>.env`, `cluster/vm/setup-person.sh`) that each person sources at the start of every session:
their name on task claims (`CDM_WHO`), git authorship, their own Claude Code login and history
(`CLAUDE_CONFIG_DIR`) and their own GitHub login (`GH_CONFIG_DIR`). The repository is marked `cdm.sharedUser`, so
`scripts/task.sh` and the guard hook refuse to work without a profile and keep each person's agent out of the other
person's tasks.

**Cluster access:** FAITH accepts only password or Kerberos logins (no SSH keys). A **Kerberos keytab** for Michael's
FAITH account lets a cron job get a fresh ticket every 30 minutes, and `cluster/lib.sh` opens the shared SSH connection
on its own whenever it is missing. Nobody types the password after step 3, and agents never see it. With one Linux
user there is one keytab, one cron job and one shared SSH connection for both people.

**Security, read first.**
- The keytab is password-equivalent: whoever can read it can log in to FAITH as Michael. On the VM it is readable by
  the shared Linux user, i.e. by both people and their agents (and by root). Everything submitted from the VM runs
  under Michael's FAITH account, including the colleague's jobs -- check that FAITH's terms of use allow this, or ask
  the FAITH admins for a project account (then repeat step 3 with that principal). A password change invalidates the
  keytab: re-run step 3.
- Under one Linux user, nothing is private between the two people: each can read the other's Claude Code history,
  `.env` (API keys) and GitHub token. The profiles separate identities, not permissions.

## Steps

Prerequisites on the VM: Ubuntu/Debian with `git`, `rsync`, `python3`, `jq`, `krb5-user`, `cron`, `openssh-client`,
`gh` (`sudo apt install git rsync python3 python3-venv jq krb5-user cron openssh-client gh`; when `krb5-user` asks
for the default realm, enter `FIM.UNI-PASSAU.DE`). The VM must reach `freia.fim.uni-passau.de` and the KDCs
(`ilium`, `ischii`, `neurocranium.fim.uni-passau.de`, port 88; found through DNS, no `krb5.conf` edits needed).
Install Claude Code for the shared user.

All steps run as the shared Linux user (`cdm`), except step 1's `sudo`.

1. **Shared repository** (needs sudo once; GitHub access for the clone: `gh auth login` as either person first, the
   per-person logins follow in step 4):

   ```bash
   git clone -b dev-michael https://github.com/KanishkaGD/decision-rerank.git /tmp/cdm-setup
   sudo /tmp/cdm-setup/cluster/vm/setup-repo.sh https://github.com/KanishkaGD/decision-rerank.git \
       --remote-branch dev-michael --shared-user
   rm -rf /tmp/cdm-setup
   ```

   Creates `/srv/cdm/{repo,worktrees,secrets}` owned by the shared user, a clone on `main` that tracks
   `origin/dev-michael`, and sets `cdm.mainBranch`, `cdm.remote`, `cdm.remoteBranch` and `cdm.sharedUser=true` for
   `scripts/task.sh`. When the team decides which GitHub branch is canonical (see "Open" below), change
   `git -C /srv/cdm/repo config cdm.remoteBranch <branch>`.

2. **SSH config:**

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

   Writes `/srv/cdm/secrets/faith.keytab` (0640), installs the cron job, gets a ticket and tests a password-less
   login. If `ktutil` can't make a working keytab, ask the FIM IT (`https://www.fim.uni-passau.de/it-dienste/`).

4. **One profile per person** (each person once, with their own name):

   ```bash
   /srv/cdm/repo/cluster/vm/setup-person.sh michael "Michael Dinzinger" michael.dinzinger@uni-passau.de
   source ~/people/michael.env
   gh auth login        # this person's GitHub account; git pushes use it
   claude               # this person's Claude login; /status shows the config directory ~/.claude-michael
   ```

   The colleague does the same with their name. `setup-person.sh` also tells git to take GitHub credentials from
   `gh`, which reads the person's `GH_CONFIG_DIR`.

   **Claude Code subscriptions:** each person logs in with their own subscription. Claude Code keeps the login in
   `CLAUDE_CONFIG_DIR` (`~/.claude-<name>`), so a session uses the subscription of whoever sourced their profile.
   Check it on the first day:
   - `/status` in each person's session shows that person's account and config directory;
   - `ls ~/.claude-michael ~/.claude-<colleague>` shows separate login files;
   - no `ANTHROPIC_API_KEY` in the shared user's `~/.bashrc` or `~/.profile` (it would override both logins).

   If both sessions show the same account, `CLAUDE_CONFIG_DIR` doesn't separate logins in that Claude Code version;
   fall back to separate Linux users. Under one Linux user each person can read the other's login files, so starting
   Claude Code without sourcing one's own profile uses whatever login sits in the default `~/.claude` -- don't log in
   there at all.

5. **Secrets:** copy `.env` (TypeSafe key `TYPESAFE_AI_API_KEY`, optional `HF_TOKEN`) from the laptop to
   `/srv/cdm/repo/.env` (`chmod 600`). `scripts/task.sh start` links it into every task worktree. Optionally
   `.claude/settings.local.json` (permission rules) the same way.

6. **Check** (each person, after sourcing their profile):

   ```bash
   cd /srv/cdm/repo
   cluster/login.sh --no-sync hostname   # connects without a prompt
   scripts/task.sh list
   ```

## Every session

```bash
source ~/people/<name>.env           # always first: name, git identity, Claude and GitHub logins
cd /srv/cdm/repo
scripts/task.sh list                 # what is going on
scripts/task.sh start <id> "<what>"  # or continue your own task
cd /srv/cdm/worktrees/<id> && claude
```

The prompt shows `[<name>]` once the profile is sourced. Without it, `scripts/task.sh` refuses to claim or finish,
and the guard stops the agent from editing. Two people (and several agents) can work at the same time, each in their
own task worktree; the first `cluster/*.sh` call from a worktree syncs it to its own
`/shared/dinzinger/calibrated-decision-models-for-ir/checkouts/<user>-<worktree>/` and links the shared data,
results, logs and container into it (cluster/cluster.env).

Optional: run each session in `tmux` (a window per person), so it survives a dropped VM login.

## Maintenance

- **Password change:** re-run step 3; the cron job keeps working with the new keytab.
- **Ticket problems:** `klist` (expiry), `kinit -k -t /srv/cdm/secrets/faith.keytab dinzinger@FIM.UNI-PASSAU.DE`
  (manual refresh), `crontab -l | grep cdm-kerberos` (job installed?).
- **Old checkout directories** on `/shared/.../checkouts/` are not removed automatically; delete those of finished
  tasks now and then (`cluster/login.sh --no-sync ls ../`), like the old `snapshots/`.

## Open

- The GitHub repository `KanishkaGD/decision-rerank` has its own `master` with a history unrelated to this one.
  Until the team decides how to combine them, the VM's `main` tracks `dev-michael`. Merging the two histories
  (`git merge --allow-unrelated-histories`) is a one-time decision for the owners, not for an agent.

#!/usr/bin/env bash
# Adds the `faith` host to ~/.ssh/config of the calling VM user (idempotent) and trusts FAITH's host certificate
# authority, so cluster/*.sh can log in with a Kerberos ticket (cluster/vm/setup-kerberos.sh) and no prompt.
# Every VM user who runs agents runs this once. Usage: cluster/vm/setup-ssh.sh [cluster-user]  (default: dinzinger)
set -euo pipefail
CLUSTER_USER="${1:-dinzinger}"
mkdir -p ~/.ssh/cm
chmod 700 ~/.ssh ~/.ssh/cm
touch ~/.ssh/config
chmod 600 ~/.ssh/config

if grep -q '^Host faith$' ~/.ssh/config; then
    echo "~/.ssh/config already has a 'Host faith' entry; left unchanged."
else
    cat >>~/.ssh/config <<EOF

# FAITH Slurm cluster (calibrated-decision-models-for-ir, cluster/vm/setup-ssh.sh). Kerberos login, shared connection.
Host faith
    HostName freia.fim.uni-passau.de
    # Alternative, identical login node: frigga.fim.uni-passau.de
    User ${CLUSTER_USER}
    ControlMaster auto
    ControlPath ~/.ssh/cm/%r@%h:%p
    ControlPersist 12h
    ServerAliveInterval 60
    ServerAliveCountMax 5
    GSSAPIAuthentication yes
    # Log in with the ticket, but don't forward it to the cluster (FIM's advice for ssh -K).
    GSSAPIDelegateCredentials no
EOF
    echo "Added 'Host faith' (user ${CLUSTER_USER}) to ~/.ssh/config."
fi

# FAITH's host certificate authority, from https://faith.pages.fim.uni-passau.de/public-docs/slurm/ -- so the first
# connection doesn't stop at an unknown-host-key question nobody can answer.
if ! grep -q '^@cert-authority freia.fim.uni-passau.de' ~/.ssh/known_hosts 2>/dev/null; then
    echo "Add FAITH's '@cert-authority freia.fim.uni-passau.de,frigga.fim.uni-passau.de ...' line from the docs page"
    echo "above to ~/.ssh/known_hosts (or copy it from a machine that has it: grep cert-authority ~/.ssh/known_hosts)."
fi

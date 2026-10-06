#!/usr/bin/env bash
# One-time setup, BY A HUMAN, on the shared VM, so that cluster/*.sh reach FAITH without anyone ever renewing a
# ticket or typing a password again (docs/vm-setup.md):
#
#   1. a Kerberos keytab for the FAITH account, created with ktutil -- the password is typed into ktutil here, once,
#      and never stored anywhere else (not in the repo, not in .env, not in a chat with an agent);
#   2. a cron job for the calling VM user that gets a fresh ticket from the keytab every 30 minutes (and at boot);
#   3. a first ticket, and a test login to the FAITH login node without a password.
#
# The keytab is password-equivalent: anyone who can read it can log in as that FAITH user. It is stored with mode
# 0640, owned by the calling user and the group given (default: cdm), so only members of that group can use it.
# Changing the FAITH password invalidates it; run this script again afterwards.
#
# Usage: cluster/vm/setup-kerberos.sh [--keytab PATH] [--group GROUP] [--principal USER@FIM.UNI-PASSAU.DE]
#        cluster/vm/setup-kerberos.sh --cron-only    another VM user: install only the cron job for an existing keytab
set -euo pipefail

KEYTAB="/srv/cdm/secrets/faith.keytab"
GROUP="cdm"
PRINCIPAL="dinzinger@FIM.UNI-PASSAU.DE"
CRON_ONLY=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --keytab) KEYTAB="$2"; shift ;;
        --group) GROUP="$2"; shift ;;
        --principal) PRINCIPAL="$2"; shift ;;
        --cron-only) CRON_ONLY=1 ;;
        *) echo "setup-kerberos.sh: unknown option $1" >&2; exit 2 ;;
    esac
    shift
done

for tool in kinit klist ktutil crontab ssh; do
    command -v "${tool}" >/dev/null || {
        echo "Missing '${tool}'. Install: sudo apt install krb5-user cron openssh-client" >&2
        exit 1
    }
done

if [ "${CRON_ONLY}" -eq 0 ]; then
    [ -t 0 ] || { echo "Run this in a terminal: ktutil asks for the FAITH password." >&2; exit 1; }
    mkdir -p "$(dirname "${KEYTAB}")"
    chgrp "${GROUP}" "$(dirname "${KEYTAB}")" && chmod 2750 "$(dirname "${KEYTAB}")"
    rm -f "${KEYTAB}.new"
    echo "ktutil will ask for the FAITH password of ${PRINCIPAL} (typed here only, never stored in plain text)."
    # -f asks the KDC for the right salt, so the key matches what the KDC stores (MIT krb5 >= 1.17).
    printf 'addent -password -p %s -k 1 -f\nwkt %s\nquit\n' "${PRINCIPAL}" "${KEYTAB}.new" | ktutil
    chgrp "${GROUP}" "${KEYTAB}.new"
    chmod 0640 "${KEYTAB}.new"
    if ! kinit -k -t "${KEYTAB}.new" "${PRINCIPAL}"; then
        rm -f "${KEYTAB}.new"
        echo "The keytab does not work (wrong password, or the KDC uses another salt/encryption type)." >&2
        echo "Ask the FIM IT for a keytab, or try: ktutil -> addent -password -p ${PRINCIPAL} -k 1 -e aes256-cts-hmac-sha1-96" >&2
        exit 1
    fi
    mv -f "${KEYTAB}.new" "${KEYTAB}"
    echo "Keytab written: ${KEYTAB}"
fi

[ -r "${KEYTAB}" ] || { echo "Cannot read ${KEYTAB}: are you in group ${GROUP}? (sudo usermod -aG ${GROUP} \$USER, log in again)" >&2; exit 1; }

# The cron job: a fresh ticket every 30 minutes and at boot, in the default ticket cache, which ssh (and so
# cluster/lib.sh's automatic reconnect) uses. Idempotent: replaces an earlier line of this script.
LINE="*/30 * * * * kinit -k -t ${KEYTAB} ${PRINCIPAL} >/dev/null 2>&1 # cdm-kerberos"
BOOT="@reboot sleep 30 && kinit -k -t ${KEYTAB} ${PRINCIPAL} >/dev/null 2>&1 # cdm-kerberos"
{ crontab -l 2>/dev/null | grep -v '# cdm-kerberos$' || true; echo "${LINE}"; echo "${BOOT}"; } | crontab -
echo "Cron job installed for $(id -un):"
crontab -l | grep '# cdm-kerberos$'

kinit -k -t "${KEYTAB}" "${PRINCIPAL}"
klist | head -5
echo "Testing a password-less login to FAITH ..."
if ssh -o BatchMode=yes -o ControlMaster=no -o ConnectTimeout=15 faith hostname; then
    echo "OK: cluster/*.sh now (re)connect on their own."
else
    echo "Login failed. Check ~/.ssh/config (cluster/vm/setup-ssh.sh) and that this VM reaches freia.fim.uni-passau.de." >&2
    exit 1
fi

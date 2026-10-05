#!/usr/bin/env bash
# Prints a job's log from the cluster. Log files are logs/<jobid>_<job>_<args>_<timestamp>.out while the job runs
# and logs/done/... once it has finished (see slurm/log_command.sh).
#
# Usage: cluster/logs.sh <jobid> [-n lines]   last lines (default 100)
#        cluster/logs.sh <jobid> -f           follow the log until you press Ctrl-C
#        cluster/logs.sh --list [n]           the n most recent log files (default 20)
source "$(dirname "$0")/lib.sh"
ensure_connection

if [ "${1:-}" = "--list" ]; then
    remote_shell "ls -t logs/*.out logs/done/*.out 2>/dev/null | head -n $(printf '%d' "${2:-20}")"
    exit 0
fi

JOB_ID="${1:?Usage: $0 <jobid> [-n lines|-f] | --list [n]}"
[[ "${JOB_ID}" =~ ^[0-9]+$ ]] || die "not a job id: ${JOB_ID}"
shift

LINES=100
FOLLOW=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        -f) FOLLOW=1 ;;
        -n) LINES="$(printf '%d' "$2")"; shift ;;
        *) die "unknown option $1" ;;
    esac
    shift
done

if [ "${FOLLOW}" -eq 1 ]; then
    # tail -f follows the open file, so it keeps going when the job moves its log to logs/done/ at the end.
    remote_shell "f=\$(ls logs/${JOB_ID}_*.out logs/${JOB_ID}_*.out.tmp logs/done/${JOB_ID}_*.out 2>/dev/null | head -n 1); [ -n \"\$f\" ] || { echo 'no log for job ${JOB_ID} (yet)' >&2; exit 1; }; tail -n ${LINES} -f \"\$f\""
else
    remote_shell "for f in logs/${JOB_ID}_*.out logs/${JOB_ID}_*.out.tmp logs/done/${JOB_ID}_*.out; do [ -e \"\$f\" ] && { echo \"==> \$f <==\"; tail -n ${LINES} \"\$f\"; }; done; true"
fi

#!/usr/bin/env bash
# Shows your Slurm jobs, from your machine.
#
# Usage: cluster/status.sh                 queued and running jobs (squeue --me)
#        cluster/status.sh <jobid>...      state, run time, exit code and peak memory of those jobs (sacct)
#        cluster/status.sh --today         every job since midnight, finished ones included (sacct)
#        cluster/status.sh --seff <jobid>  CPU/GPU/memory efficiency of a finished job (seff-gpu)
source "$(dirname "$0")/lib.sh"
ensure_connection

FORMAT='JobID%12,JobName%28,Partition%9,State%12,Elapsed%11,Timelimit%11,ExitCode%6,MaxRSS%10,NodeList%14'
case "${1:-}" in
    "")
        cluster_ssh 'squeue --me --format="%.12i %.28j %.9P %.10T %.11M %.11l %.6D %R"'
        ;;
    --today)
        cluster_ssh "sacct -X --starttime=today --format=${FORMAT}"
        ;;
    --seff)
        [ -n "${2:-}" ] || die "usage: $0 --seff <jobid>"
        cluster_ssh "seff-gpu $(printf '%q' "$2")"
        ;;
    *)
        cluster_ssh "sacct -j $(IFS=,; echo "$*") --format=${FORMAT}"
        ;;
esac

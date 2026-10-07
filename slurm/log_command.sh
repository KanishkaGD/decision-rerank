# Source this, don't run it. Defines log_job_command and start_job_log, which every slurm/*.sbatch calls first,
# so each log file begins with exactly what was run and ends up under a readable name.

# Prints a header. Usage, before any `shift`: log_job_command <job>.sbatch "$@"
# (pass the script's own name: inside a job, $0 is Slurm's spooled copy, not slurm/<job>.sbatch).
#   "Submitted with": the slurm/submit.sh command line, if the job came from there (exported as CDM_SUBMIT_COMMAND;
#                     sbatch passes the submitting environment on to the job).
#   "sbatch command": the sbatch command line as Slurm recorded it (sacct's SubmitLine), incl. --time.
#   "Job script":     this script and its exact arguments.
#   "Code":           the revision of the code snapshot the job runs (REVISION, slurm/snapshot.sh).
log_job_command() {
    local script_name="$1"
    shift
    local submit_line=""
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        submit_line="$(sacct -j "${SLURM_JOB_ID}" -X -n -P --format=SubmitLine 2>/dev/null | head -n 1 || true)"
    fi
    echo "================================================================================"
    [ -n "${CDM_SUBMIT_COMMAND:-}" ] && echo "Submitted with: ${CDM_SUBMIT_COMMAND}"
    [ -n "${submit_line}" ] && echo "sbatch command: ${submit_line}"
    echo "Job script:     slurm/${script_name}$( [ "$#" -gt 0 ] && printf ' %q' "$@" )"
    if [ -f "${CDM_CODE_DIR:-${SLURM_SUBMIT_DIR:-.}}/REVISION" ]; then
        echo "Code:           $(sed -n 's/^describe=//p' "${CDM_CODE_DIR:-${SLURM_SUBMIT_DIR:-.}}/REVISION")"
    fi
    echo "Job ${SLURM_JOB_ID:-?} on $(hostname), started $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================================"
}

# Renames the job's log from Slurm's logs/<jobid>_<job>.out.tmp (the #SBATCH --output name, which can't contain
# the arguments or a timestamp) to logs/<jobid>_<job>[_<label>]_<timestamp>.out, and moves it to logs/done/ when
# the job exits, successfully or not. Usage: start_job_log <job> [label]
start_job_log() {
    local job="$1" label="${2:-}"
    [ -n "${label}" ] && label="_${label//\//_}"
    LOG_FILE="logs/${SLURM_JOB_ID}_${job}${label}_$(date +%Y-%m-%d_%H-%M-%S).out"
    mv "logs/${SLURM_JOB_ID}_${job}.out.tmp" "${LOG_FILE}"
    mkdir -p logs/done
    # shellcheck disable=SC2064  # expand LOG_FILE now
    trap "mv -f '${LOG_FILE}' 'logs/done/$(basename "${LOG_FILE}")'" EXIT
}

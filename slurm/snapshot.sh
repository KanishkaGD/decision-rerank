# Source this, don't run it. Defines snapshot_code, which slurm/submit.sh and slurm/exec.sh call before they hand
# work to Slurm.
#
# A queued job starts later, perhaps after src/ has been edited again, so every submission freezes the code it was
# submitted with: snapshots/<timestamp>_<pid>/ gets a copy of src/ plus
#   REVISION        commit=, describe=, dirty= (uncommitted or untracked files count), snapshot_at=, snapshot_on=
#   REVISION.patch  the uncommitted changes against HEAD, untracked files included
# The job puts the snapshot first on PYTHONPATH (slurm/env.sh, CDM_CODE_DIR) and every manifest records its REVISION
# and copies the patch (src/cdm/provenance.py). Snapshots are small (src/ only); delete old ones whenever no job of
# theirs is queued.
#
# Usage: snapshot_code   (with CDM_ROOT set to the repo root; exports CDM_CODE_DIR)
snapshot_code() {
    : "${CDM_ROOT:?set CDM_ROOT to the repo root before calling snapshot_code}"
    local dir="${CDM_ROOT}/snapshots/$(date +%Y-%m-%d_%H-%M-%S)_$$"
    mkdir -p "${dir}"
    cp -r "${CDM_ROOT}/src" "${dir}/"
    if git -C "${CDM_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
        {
            echo "commit=$(git -C "${CDM_ROOT}" rev-parse HEAD 2>/dev/null || echo none)"
            echo "describe=$(git -C "${CDM_ROOT}" describe --always --dirty 2>/dev/null || echo none)"
            if [ -n "$(git -C "${CDM_ROOT}" status --porcelain)" ]; then
                echo "dirty=true"
            else
                echo "dirty=false"
            fi
            echo "snapshot_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            echo "snapshot_on=$(hostname)"
        } >"${dir}/REVISION"
        {
            git -C "${CDM_ROOT}" diff HEAD 2>/dev/null || true
            # Untracked paths are relative to the repo root, so diff them from there.
            (cd "${CDM_ROOT}" && git ls-files --others --exclude-standard -z | while IFS= read -r -d '' f; do
                git diff --no-index -- /dev/null "${f}" 2>/dev/null || true
            done)
        } >"${dir}/REVISION.patch"
    else
        echo "commit=none" >"${dir}/REVISION"
        : >"${dir}/REVISION.patch"
    fi
    export CDM_CODE_DIR="${dir}" # sbatch and srun pass the environment on to the job
}

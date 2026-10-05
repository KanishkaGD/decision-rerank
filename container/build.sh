#!/usr/bin/env bash
# Builds container/cdm.sif from cdm.def. Needs fakeroot (enabled on the login nodes) and network access, so it runs
# on the login node -- from your machine: cluster/login.sh bash container/build.sh
# Alternatives if the login node refuses (see the FAITH Apptainer docs): build on your own x86-64 machine with the
# same command and copy the .sif to /shared/<user>/calibrated-decision-models-for-ir/container/.
#
# Rebuild only after changing dependencies in pyproject.toml: the code comes from the synced src/ at run time
# (PYTHONPATH, slurm/env.sh). A single .sif file also avoids Ceph's many-small-files slowdown that a venv on
# /shared would hit.
set -euo pipefail
# %files paths in cdm.def are relative to the invocation directory: run from the repo root.
cd "$(dirname "$0")/.."

# Build caches on node-local /local, not /shared or the small login-node /home; removed afterwards.
mkdir -p "/local/${USER}" 2>/dev/null || true
export APPTAINER_TMPDIR="$(mktemp -d --tmpdir="/local/${USER}" apptainer.tmp.XXXXXXXX 2>/dev/null || mktemp -d)"
export APPTAINER_CACHEDIR="$(mktemp -d --tmpdir="/local/${USER}" apptainer.cache.XXXXXXXX 2>/dev/null || mktemp -d)"
trap 'rm -rf -- "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"' EXIT

apptainer build --force --fakeroot container/cdm.sif.new container/cdm.def
mv -f container/cdm.sif.new container/cdm.sif
echo "Built container/cdm.sif ($(du -h container/cdm.sif | cut -f1)) from $(sed -n 's/^describe=//p' REVISION 2>/dev/null || echo 'unknown revision')"

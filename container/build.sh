#!/usr/bin/env bash
# Builds container/<image>.sif from <image>.def. Needs fakeroot (enabled on the login nodes) and network access, so it
# runs on the login node:
#   bash container/build.sh                  the main image, cdm.sif
#   bash container/build.sh lightning-ir     the Set-Encoder image, lightning-ir.sif (slurm/config.sh container_for)
# Alternative if the login node refuses (see the FAITH Apptainer docs): build on an x86-64 machine of your own with
# the same command and copy the .sif to container/ here.
#
# Rebuild only after changing dependencies in pyproject.toml or container/cdm.def: the code comes from src/ at run time
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

IMAGE="${1:-cdm}"
[ -f "container/${IMAGE}.def" ] || { echo "container/build.sh: no container/${IMAGE}.def" >&2; exit 2; }
apptainer build --force --fakeroot "container/${IMAGE}.sif.new" "container/${IMAGE}.def"
mv -f "container/${IMAGE}.sif.new" "container/${IMAGE}.sif"
REV="$(git describe --always --dirty 2>/dev/null || echo 'unknown revision')"
echo "Built container/${IMAGE}.sif ($(du -h "container/${IMAGE}.sif" | cut -f1)) from ${REV}"

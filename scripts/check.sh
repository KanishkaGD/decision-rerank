#!/usr/bin/env bash
# The checks to run before merging a branch into master:
#   1. shell syntax of every script, 2. Python byte-compilation of src/ and tests/, 3. the unit tests.
# The tests run locally when pytest and the dependencies are importable, else in the container on a compute node
# (cluster/exec.sh, which needs the cluster connection). CDM_SKIP_TESTS=1 skips step 3 -- only for changes that
# touch no code (docs, the paper), and say so in the commit message.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "check: shell syntax"
for f in cluster/*.sh slurm/*.sh slurm/*.sbatch scripts/*.sh container/*.sh; do
    [ -f "${f}" ] && bash -n "${f}"
done

echo "check: python compiles"
python3 -m compileall -q src tests scripts >/dev/null

if [ "${CDM_SKIP_TESTS:-0}" = 1 ]; then
    echo "check: tests SKIPPED (CDM_SKIP_TESTS=1)"
    exit 0
fi
if python3 -c "import pytest, numpy, pandas, click, ir_measures" 2>/dev/null; then
    echo "check: tests (local)"
    PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}" python3 -m pytest -q -p no:cacheprovider tests
else
    echo "check: tests (container on a compute node; no local test environment)"
    cluster/exec.sh -- bash -c \
        'PYTHONPATH=$PYTHONPATH:$CDM_ROOT/data/devtools python -m pytest -q -p no:cacheprovider -W ignore::DeprecationWarning tests'
fi

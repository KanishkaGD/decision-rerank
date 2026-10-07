"""
manifest.json for every output directory: what was run, on which code, data, models, library versions and hardware.
The research plan requires recording library, model and checkpoint versions and logging every configuration tried.
"""

import functools
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from typing import Any, Dict

from cdm.config import CDM_ROOT

PACKAGES = [
    "cdm",
    "torch",
    "transformers",
    "sentence-transformers",
    "pyserini",
    "ir-measures",
    "numpy",
    "pandas",
    "typesafe-sdk",
    "laya",
    "kev",
    "peft",
    "flash-linear-attention",
]


# Where the running code came from: the per-submission snapshot of slurm/submit.sh, else the repo root.
CODE_DIR = os.environ.get("CDM_CODE_DIR", CDM_ROOT)


@functools.lru_cache(maxsize=None)
def code_revision() -> Dict[str, str]:
    """
    The REVISION file of the code snapshot the job runs (slurm/snapshot.sh), else git itself. Read once, at
    the first call, so a change to the repo during a long job can't change what this run reports.
    """
    path = os.path.join(CODE_DIR, "REVISION")
    if os.path.exists(path):
        with open(path) as f:
            return dict(line.rstrip("\n").split("=", 1) for line in f if "=" in line)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=CDM_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=CDM_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return {"commit": commit, "dirty": str(bool(status.strip())).lower()}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown"}


def package_versions() -> Dict[str, str]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def hardware() -> Dict[str, Any]:
    info: Dict[str, Any] = {"host": socket.gethostname(), "cpu_count": os.cpu_count(), "platform": platform.platform()}
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return info


def manifest(**config: Any) -> Dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": sys.argv,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "code": code_revision(),
        "packages": package_versions(),
        "container_pip_freeze": os.environ.get("CDM_PIP_FREEZE"),
        "python": sys.version.split()[0],
        "hardware": hardware(),
        "config": config,
    }


def write_manifest(directory: str, **config: Any) -> Dict[str, Any]:
    """Writes manifest.json, plus REVISION.patch (the uncommitted changes the run used), if there were any."""
    os.makedirs(directory, exist_ok=True)
    data = manifest(**config)
    patch = os.path.join(CODE_DIR, "REVISION.patch")
    if os.path.exists(patch) and os.path.getsize(patch) > 0:
        shutil.copyfile(patch, os.path.join(directory, "REVISION.patch"))
        data["code"] = {**data["code"], "patch": "REVISION.patch"}
    with open(os.path.join(directory, "manifest.json"), "w") as f:
        json.dump(data, f, indent=2, default=str)
    return data

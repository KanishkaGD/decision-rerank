#!/usr/bin/env python3
"""
Consistency check for straightening the branch: do code, Slurm config, docs and paper describe the same, current
plan? Light (git + file reads, standard library), fine on the login node. Prints findings; exit status 1 if any.

  python3 .claude/skills/straighten-branch/scripts/check.py

Checks:
  1. stale terms: every regex in ../deprecated.txt, searched in all tracked files except the history files
     (docs/experiment-log.md, docs/research-plan.md, results/) and the deprecated list itself;
  2. registries: SYSTEMS / DATASETS in slurm/config.sh vs. src/cdm/rerankers and src/cdm/datasets.py;
  3. results/literature.csv: datasets registered, systems either registered or literature-only;
  4. pointers: files that docs mention by path (`docs/...`, `slurm/...`, `src/...`, ...) exist.
"""

import csv
import os
import re
import subprocess
import sys

ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip() or "."
HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY = ("docs/experiment-log.md", "docs/research-plan.md", "results/")
SKIP = (".claude/skills/straighten-branch/deprecated.txt",)
TEXT = (".md", ".py", ".sh", ".sbatch", ".tex", ".bib", ".toml", ".def", ".txt", ".yaml", ".cfg", ".example")


def tracked():
    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=ROOT).stdout.split()
    files += subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True,
                            cwd=ROOT).stdout.split()
    return [f for f in files if f.endswith(TEXT) and not f.startswith(HISTORY) and f not in SKIP]


def stale_terms(files):
    patterns = []
    with open(os.path.join(HERE, "..", "deprecated.txt")) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(re.compile(line))
    hits = []
    for f in files:
        try:
            lines = open(os.path.join(ROOT, f), errors="replace").read().split("\n")
        except OSError:
            continue
        for n, line in enumerate(lines, 1):
            for p in patterns:
                m = p.search(line)
                if m:
                    hits.append(f"{f}:{n}: '{m.group(0)}' ({p.pattern})  {line.strip()[:100]}")
    return hits


def config_list(name):
    out = subprocess.run(["bash", "-c", f'source slurm/config.sh >/dev/null 2>&1 && printf "%s\\n" "${{{name}[@]}}"'],
                         capture_output=True, text=True, cwd=ROOT).stdout
    return [x for x in out.split("\n") if x]


def registries():
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from cdm.datasets import DATASETS
    from cdm.rerankers import SYSTEMS

    out = []
    sys_cfg, ds_cfg = set(config_list("SYSTEMS")), set(config_list("DATASETS") + config_list("DEV_DATASETS"))
    for name in sorted(set(SYSTEMS) - sys_cfg):
        out.append(f"system '{name}' is registered in src/cdm/rerankers but not in slurm/config.sh SYSTEMS")
    for name in sorted(sys_cfg - set(SYSTEMS)):
        out.append(f"system '{name}' is in slurm/config.sh SYSTEMS but not registered")
    for name in sorted(set(DATASETS) - ds_cfg):
        out.append(f"dataset '{name}' is registered in src/cdm/datasets.py but not in slurm/config.sh")
    for name in sorted(ds_cfg - set(DATASETS)):
        out.append(f"dataset '{name}' is in slurm/config.sh but not registered")
    return out, set(SYSTEMS), set(DATASETS)


def literature(systems, datasets):
    path = os.path.join(ROOT, "results", "literature.csv")
    if not os.path.exists(path):
        return []
    out = []
    for i, r in enumerate(csv.DictReader(open(path)), 2):
        if r["dataset"] not in datasets:
            out.append(f"results/literature.csv:{i}: dataset '{r['dataset']}' is not registered")
        if r["role"] == "baseline" and r["system"] in systems:
            lit = subprocess.run(["bash", "-c", f'source slurm/config.sh && from_literature "{r["system"]}" '
                                                f'"{r["dataset"]}"'], cwd=ROOT).returncode == 0
            if not lit:
                out.append(f"results/literature.csv:{i}: {r['system']} on {r['dataset']} is a baseline from the "
                           f"literature but from_literature (slurm/config.sh) doesn't list it -- it would be recomputed")
    return out


def pointers(files):
    out = []
    pat = re.compile(r"`((?:docs|slurm|src|container|paper|results|tests|\.claude)/[\w./-]+\.\w+)`")
    for f in files:
        if not f.endswith(".md"):
            continue
        for n, line in enumerate(open(os.path.join(ROOT, f), errors="replace").read().split("\n"), 1):
            for m in pat.finditer(line):
                target = m.group(1)
                if "<" in target or "*" in target:
                    continue
                if not os.path.exists(os.path.join(ROOT, target)):
                    out.append(f"{f}:{n}: points to missing file {target}")
    return out


def main():
    files = tracked()
    sections = [("Stale terms (deprecated.txt)", stale_terms(files))]
    reg, systems, datasets = registries()
    sections += [("Registries", reg), ("Literature numbers", literature(systems, datasets)),
                 ("File pointers in docs", pointers(files))]
    total = 0
    for title, items in sections:
        print(f"## {title}: {len(items) or 'ok'}")
        for item in items:
            print("  " + item)
        total += len(items)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())

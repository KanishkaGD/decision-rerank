"""
Copy results/results.csv and results/literature.csv into the code release (release/results/), filtered the way the
ECIR release was (feature/release-20261012, cb3c6d5): only rows of the systems registered in the pruned release code
(release/src/cdm/rerankers) and of the first stage; no columns that point into our cluster (output paths, Slurm job
ids, code revisions, source outputs, the debug counter no_yes_no_answer). Rows and values are copied unchanged, in
their order. Standard library only:

    python scripts/release_results.py            (from the repository root)
"""

import csv
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE = os.path.join(ROOT, "release")
DROP_COLUMNS = ["code", "no_yes_no_answer", "output", "slurm_job_id", "source_output"]
FIRST_STAGE_SYSTEMS = {"bm25", "official"}
# The 9B fine-tune's msmarco-dev selection row without a checkpoint variant scored the `final` adapter before
# checkpoint-800 was selected; the registered system now reads checkpoint-800, so the row would be mislabelled (the
# ECIR release left it out too). The 27B one's row without a variant is `final`, the selected checkpoint: kept.
DROP_KEYS = {("rerank", "qwen3.5-9b-reranker", "msmarco-dev", "", "max200")}


def release_systems() -> set:
    sys.path.insert(0, os.path.join(RELEASE, "src"))
    from cdm.rerankers import SYSTEMS

    return set(SYSTEMS)


def main() -> None:
    keep_systems = release_systems() | FIRST_STAGE_SYSTEMS
    src = os.path.join(ROOT, "results", "results.csv")
    dst = os.path.join(RELEASE, "results", "results.csv")
    with open(src, newline="") as f:
        reader = csv.DictReader(f)
        columns = [c for c in reader.fieldnames if c not in DROP_COLUMNS]
        rows = [
            r
            for r in reader
            if r["system"] in keep_systems
            and (r["job"], r["system"], r["dataset"], r["variant"], r["subset"]) not in DROP_KEYS
        ]
    with open(dst, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    shutil.copyfile(os.path.join(ROOT, "results", "literature.csv"), os.path.join(RELEASE, "results", "literature.csv"))
    print(f"Wrote {dst}: {len(rows)} rows, {len(columns)} columns")


if __name__ == "__main__":
    main()

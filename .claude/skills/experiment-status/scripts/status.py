#!/usr/bin/env python3
"""
Experiment status report for this repository: Slurm jobs, our results, coverage of the planned grid, and what the
other branches (Kanishka's) have run. Prints Markdown to stdout.

Light by design -- it runs on the FAITH login node: git, squeue/sacct, and reading CSV/log files. Standard library
only (no pandas, no container), so it works with the system python3.

Usage (from anywhere inside the repo):
  python3 .claude/skills/experiment-status/scripts/status.py [--days 7] [--no-fetch] [--section jobs,results,...]
Sections: jobs, results, coverage, next, others (default: all).
"""

import argparse
import csv
import glob
import io
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip() or "."
SECTIONS = ["jobs", "results", "coverage", "next", "others"]
ACTIVE = {"PENDING", "RUNNING", "REQUEUED", "SUSPENDED", "CONFIGURING", "COMPLETING"}
BAD = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "CANCELLED", "BOOT_FAIL", "DEADLINE", "PREEMPTED"}
# Kanishka's system labels (branch master, results/effectiveness.csv) -> our closest system. His runs rerank the
# official 1000-candidate DL19/20 pools, ours the BM25 top-100 (docs/roadmap.md), so no number is directly comparable;
# the mapping only says which of our systems is the same model.
KANISHKA_TO_OURS = {
    "BM25 (bm25s order of the pool)": ("bm25", "other candidates; bm25s k1=1.2 b=0.75"),
    "Laya noul: P(yes)": ("laya", "other candidates, question and state"),
    "Jev noul: P(yes)": ("jev-isolated", "other candidates, question and state"),
    "Qwen3.7-Flash noul: P(yes)": ("qwen3.7-flash", "other candidates; same prompt"),
}


def sh(cmd, check=False):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, shell=isinstance(cmd, str))
    if check and r.returncode:
        raise RuntimeError(r.stderr)
    return r.stdout


def config_list(name):
    """An array from slurm/config.sh (bash), e.g. SYSTEMS or DATASETS."""
    out = sh(["bash", "-c", f'source slurm/config.sh >/dev/null 2>&1 && printf "%s\\n" "${{{name}[@]}}"'])
    return [x for x in out.split("\n") if x]


def table(header, rows):
    if not rows:
        return "_(none)_\n"
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def fmt(v, nd=3):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "" if v is None else str(v)


# ---------------------------------------------------------------------------------------------------------- jobs
def job_args(submit_line):
    """(job, system, dataset, extra args) from an sbatch SubmitLine of slurm/submit.sh."""
    m = re.search(r"slurm/([\w-]+)\.sbatch\s*(.*)$", submit_line or "")
    if not m:
        return None, None, None, ""
    job, rest = m.group(1), m.group(2).split()
    if job == "rerank":
        return job, (rest[0] if rest else None), (rest[1] if len(rest) > 1 else None), " ".join(rest[2:])
    if job == "first-stage":
        return job, None, (rest[0] if rest else None), " ".join(rest[1:])
    return job, None, None, " ".join(rest)


def log_file(jobid):
    hits = glob.glob(os.path.join(ROOT, "logs", f"{jobid}_*.out")) + glob.glob(
        os.path.join(ROOT, "logs", "done", f"{jobid}_*.out")) + glob.glob(os.path.join(ROOT, "logs", f"{jobid}_*.tmp"))
    return os.path.relpath(hits[0], ROOT) if hits else None


def last_error(path, n=3):
    """The last lines of a log that look like the cause of a failure."""
    try:
        with open(os.path.join(ROOT, path), errors="replace") as f:
            lines = [l.rstrip() for l in f.read().replace("\r", "\n").split("\n") if l.strip()]
    except OSError:
        return ""
    pat = re.compile(r"Error|Exception|Traceback|CANCELLED|DUE TO TIME LIMIT|oom|Killed|No such file|ClickException")
    hits = [l for l in lines if pat.search(l)]
    return " / ".join((hits or lines)[-n:])[:300]


SACCT_FORMAT = "JobID,JobName,State,SubmitLine,Elapsed,Timelimit,Start,End,Reason,WorkDir"


def jobs(days):
    """
    This checkout's cdm-* jobs: finished ones of the last `days` days (sacct) plus everything still queued (squeue;
    sacct's time window leaves out jobs that have not started). Newest first.
    """
    out = sh(["sacct", "-X", "-P", "-n", "-S", f"now-{days}days", "--format", SACCT_FORMAT])
    queued = {}
    for line in sh(["squeue", "--me", "-h", "-o", "%i|%T|%r"]).splitlines():
        jobid, state, reason = line.split("|")
        queued[jobid] = (state, reason)
    missing = set(queued) - {line.split("|", 1)[0] for line in out.splitlines()}
    if missing:
        out += "\n" + sh(["sacct", "-X", "-P", "-n", "-j", ",".join(sorted(missing)), "--format", SACCT_FORMAT])
    rows = []
    for line in out.splitlines():
        p = line.split("|")
        if len(p) < 10 or not p[1].startswith("cdm-"):
            continue
        jobid, name, state, submit, elapsed, limit, start, end, reason, workdir = p[:10]
        if jobid in queued:  # squeue knows the live state and the pending reason (sacct says "None")
            state, reason = queued[jobid]
        if os.path.realpath(workdir) != os.path.realpath(ROOT):
            continue  # another checkout (e.g. the old calibrated-decision-models-for-ir copy)
        job, system, dataset, extra = job_args(submit)
        deps = [int(x) for x in re.findall(r"\d+", (re.search(r"--dependency=\S+", submit) or [""])[0])]
        rows.append(dict(id=int(jobid), name=name, state=state.split()[0], job=job or name[4:], system=system,
                         dataset=dataset, extra=extra, elapsed=elapsed, limit=limit, start=start, end=end,
                         reason=reason, deps=deps))
    uniq = {r["id"]: r for r in rows}
    # A pending job whose afterok dependency failed or is itself stuck never starts: mark it like Slurm marks the
    # first level (DependencyNeverSatisfied), all the way down the chain.
    changed = True
    while changed:
        changed = False
        for r in uniq.values():
            if r["state"] == "PENDING" and r["reason"] != "DependencyNeverSatisfied" and any(
                    d in uniq and (uniq[d]["state"] in BAD or uniq[d]["reason"] == "DependencyNeverSatisfied")
                    for d in r["deps"]):
                r["reason"] = "DependencyNeverSatisfied"
                changed = True
    return sorted(uniq.values(), key=lambda r: -r["id"])


def section_jobs(rows):
    print("## Slurm jobs of this checkout\n")
    if not rows:
        print("_No cdm-* jobs in the period._\n")
        return
    print("States: " + ", ".join(f"{s} {n}" for s, n in Counter(r["state"] for r in rows).most_common()) + "\n")
    pending = Counter(r["reason"] for r in rows if r["state"] == "PENDING")
    if pending:
        print("Pending because: " + ", ".join(f"{k} {v}" for k, v in pending.most_common()))
        if pending.get("DependencyNeverSatisfied"):
            print("**DependencyNeverSatisfied: an upstream job failed; these jobs will never start (scancel and "
                  "resubmit after fixing it).**")
        print()
    running = [r for r in rows if r["state"] == "RUNNING"]
    print("### Running\n")
    print(table(["job id", "job", "system", "dataset", "elapsed / limit", "log"],
                [(r["id"], r["job"], r["system"] or "", r["dataset"] or "", f"{r['elapsed']} / {r['limit']}",
                  log_file(r["id"]) or "") for r in running]))
    # Failures that have not been superseded by a later attempt of the same thing.
    latest = {}
    for r in rows:  # newest first; any later prefetch supersedes an earlier one (downloads resume)
        key = ("prefetch",) if r["job"] == "prefetch" else (r["job"], r["system"], r["dataset"], r["extra"])
        latest.setdefault(key, r)
    failed = [r for r in latest.values() if r["state"] in BAD]
    print("### Failed (latest attempt)\n")
    print(table(["job id", "state", "job", "system", "dataset", "elapsed / limit", "log", "last error lines"],
                [(r["id"], r["state"], r["job"], r["system"] or "", r["dataset"] or "",
                  f"{r['elapsed']} / {r['limit']}", log_file(r["id"]) or "",
                  last_error(log_file(r["id"])) if log_file(r["id"]) else "")
                 for r in sorted(failed, key=lambda r: r["id"])]))
    done = [r for r in rows if r["state"] == "COMPLETED"]
    if done:
        print("### Completed: run times (to calibrate slurm/config.sh budgets)\n")
        print(table(["job id", "job", "system", "dataset", "elapsed / limit"],
                    [(r["id"], r["job"], r["system"] or "", r["dataset"] or "", f"{r['elapsed']} / {r['limit']}")
                     for r in done[:25]]))
        if len(done) > 25:
            print(f"_... and {len(done) - 25} more._\n")


# ------------------------------------------------------------------------------------------------------- results
def read_results():
    path = os.path.join(ROOT, "results", "results.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def section_results(results):
    print("## Our results (results/results.csv)\n")
    if not results:
        print("_results/results.csv does not exist yet: no run has finished and saved._\n")
        return
    full = [r for r in results if not r.get("subset")]
    fs = [r for r in full if r.get("job") == "first-stage"]
    rr = [r for r in full if r.get("job") == "rerank"]
    print(f"{len(rr)} rerank and {len(fs)} first-stage rows on all queries; "
          f"{len(results) - len(full)} quick-subset rows (--max-queries, not reportable).\n")
    if fs:
        print("### First stage (nDCG@10)\n")
        print(table(["dataset", "retriever", "variant", "nDCG@10", "queries"],
                    [(r["dataset"], r["system"], r.get("variant", ""), fmt(r.get("nDCG@10")), r.get("n_queries", ""))
                     for r in sorted(fs, key=lambda r: r["dataset"])]))
    if rr:
        datasets = sorted({r["dataset"] for r in rr})
        cells = defaultdict(dict)
        for r in rr:
            label = r["system"] + "".join(f" [{x}]" for x in (r.get("phrasing"), r.get("variant"))
                                          if x and x != "none")
            depth = r.get("first_stage", "")
            cells[(label, depth)][r["dataset"]] = fmt(r.get("nDCG@10"))
        print("### Reranking (nDCG@10; row = system [phrasing] [variant], candidate set)\n")
        print(table(["system", "candidates"] + datasets,
                    [(lab, depth) + tuple(v.get(d, "") for d in datasets)
                     for (lab, depth), v in sorted(cells.items())]))
        cal = [r for r in rr if r.get("ECE") not in (None, "", "nan")]
        if cal:
            print("### Calibration on judged pairs (ECE / Brier; probabilities as the system defines them)\n")
            print(table(["system", "dataset", "candidates", "ECE", "Brier", "judged non-rel pairs", "prob kind"],
                        [(r["system"], r["dataset"], r.get("first_stage", ""), fmt(r.get("ECE")),
                          fmt(r.get("Brier")), r.get("n_judged_nonrel_pairs", ""), r.get("prob_kind", ""))
                         for r in sorted(cal, key=lambda r: (r["dataset"], r["system"]))]))
        cost = [r for r in rr if r.get("api_cost_usd") not in (None, "")]
        if cost:
            total = sum(float(r["api_cost_usd"]) for r in cost)
            print(f"API cost so far: ${total:.2f} over {len(cost)} runs.\n")


# ------------------------------------------------------------------------------------------------------ coverage
def grid_candidates(datasets):
    """The candidate set (first_stage value in results.csv) each dataset's grid cell refers to."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    try:
        from cdm.datasets import DATASETS  # plain dataclasses, no heavy imports

        return {d: DATASETS[d].default_first_stage for d in datasets if d in DATASETS}
    except Exception:  # noqa: BLE001 -- fall back to "any candidate set" if the package can't be imported
        return {}


def missing_caches(datasets):
    """Datasets whose grid candidate set has no first-stage cache yet (runs/<dataset>/<candidates>/)."""
    cands = grid_candidates(datasets)
    return [d for d in datasets
            if not os.path.exists(os.path.join(ROOT, "runs", d, cands.get(d, "*"), "candidates.jsonl.gz"))]


def from_literature(system, dataset):
    return subprocess.run(["bash", "-c", f'source slurm/config.sh >/dev/null 2>&1 && from_literature "{system}" '
                                         f'"{dataset}"'], cwd=ROOT).returncode == 0


def coverage(results, rows):
    """(systems, datasets, mark(system, dataset)) for the grid of slurm/config.sh."""
    systems, datasets = config_list("SYSTEMS"), config_list("DATASETS")
    cands = grid_candidates(datasets)
    done = {(r["system"], r["dataset"]) for r in results if r.get("job") == "rerank" and not r.get("subset")
            and (r["dataset"] not in cands or r.get("first_stage") == cands[r["dataset"]])}
    latest = {}
    for r in rows:  # newest first; only jobs on the grid's candidate set (no --depth/--first-stage override)
        if r["job"] == "rerank" and r["system"] and r["dataset"] and "--depth" not in r["extra"] \
                and "--first-stage" not in r["extra"]:
            latest.setdefault((r["system"], r["dataset"]), r)
    lit = {(s, d) for s in systems for d in datasets if from_literature(s, d)}

    def mark(s, d):
        if (s, d) in done:
            return "done"
        if (s, d) in lit:
            return "lit"
        j = latest.get((s, d))
        if j and j["state"] == "PENDING" and j["reason"] == "DependencyNeverSatisfied":
            return "STUCK"
        if j and j["state"] in ACTIVE:
            return "run" if j["state"] == "RUNNING" else "queued"
        if j and j["state"] in BAD:
            return "FAILED"
        return "."

    return systems, datasets, mark


def section_coverage(results, rows):
    print("## Coverage of the grid (slurm/config.sh SYSTEMS x DATASETS, BM25 top-100 candidates)\n")
    systems, datasets, mark = coverage(results, rows)
    if not systems or not datasets:
        print("_Could not read SYSTEMS/DATASETS from slurm/config.sh._\n")
        return
    have_cache = set(datasets) - set(missing_caches(datasets))
    print("Legend: done = row in results.csv on the grid's candidates; lit = taken from results/literature.csv; "
          "run / queued = Slurm; STUCK = queued behind a failed job (never starts); FAILED = latest attempt "
          "failed; . = not run. Datasets without a first-stage cache are marked * in the header.\n")
    header = ["system"] + [d + ("" if d in have_cache else "*") for d in datasets]
    marks = {(s, d): mark(s, d) for s in systems for d in datasets}
    print(table(header, [[s] + [marks[s, d] for d in datasets] for s in systems]))
    counts = Counter(marks.values())
    print(f"{counts['done']} of {len(marks)} cells done, {counts['lit']} from the literature, "
          f"{counts['run'] + counts['queued']} running or queued, "
          f"{counts['.'] + counts['FAILED'] + counts['STUCK']} not run, failed or stuck.\n")


def section_next(results, rows):
    """slurm/submit.sh commands for the grid cells that are neither done, from the literature, nor active."""
    print("## Commands for the missing cells\n")
    systems, datasets, mark = coverage(results, rows)
    s1 = set(config_list("SYSTEMS")) - {"bge-m3", "monot5-3b", "qwen3-reranker-8b", "set-encoder-large",
                                         "qwen3.7-flash"}
    missing_cache = missing_caches(datasets)
    lines = []
    if missing_cache:
        lines.append(f"slurm/submit.sh first-stage --datasets {','.join(missing_cache)}   # then add --after <ids> below")
    for s in systems:
        todo = [d for d in datasets if mark(s, d) in (".", "FAILED", "STUCK")]
        if todo:
            extra = " --phrasings s1rank" if s in s1 else ""
            lines.append(f"slurm/submit.sh rerank {s} --datasets {','.join(todo)}{extra}")
    print("```bash\n" + ("\n".join(lines) or "# nothing missing") + "\n```\n")


# -------------------------------------------------------------------------------------------------------- others
def section_others(fetch):
    print("## Other branches (colleagues)\n")
    if fetch:
        sh(["git", "fetch", "-q", "--all", "--prune"])
    me = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
    # Remote branches other than ours; "origin" alone is the origin/HEAD alias, so refs need a slash.
    branches = [b.strip() for b in sh(["git", "branch", "-r", "--format=%(refname:short)"]).split("\n")
                if "/" in b and not b.strip().endswith("/HEAD") and b.strip().split("/", 1)[1] != me]
    for b in branches:
        authors = sh(["git", "log", "-200", "--format=%an", b]).split("\n")
        print(f"### {b} (mostly {Counter(a for a in authors if a).most_common(1)[0][0] if any(authors) else '?'})\n")
        print("Latest commits:\n")
        print("```\n" + sh(["git", "log", "-6", "--date=short", "--format=%h %ad %an  %s", b]).strip() + "\n```\n")
        files = sh(["git", "ls-tree", "-r", "--name-only", b, "--", "results"]).split()
        if files:
            print("Result files: " + ", ".join(f"`{f}`" for f in files) + "\n")
        if "results/effectiveness.csv" in files:  # Kanishka's decisionrerank format: collection,system,measure,value
            rows = list(csv.DictReader(io.StringIO(sh(["git", "show", f"{b}:results/effectiveness.csv"]))))
            ndcg = defaultdict(dict)
            for r in rows:
                if r["measure"] == "nDCG@10":
                    ndcg[r["system"]][r["collection"]] = fmt(r["value"])
            cols = sorted({c for v in ndcg.values() for c in v})
            print(f"nDCG@10 from `{b}:results/effectiveness.csv` (official DL pools, all candidates), with our "
                  "closest system:\n")
            print(table(["their system"] + cols + ["our system", "comparable?"],
                        [[s] + [v.get(c, "") for c in cols] + list(KANISHKA_TO_OURS.get(s, ("", "no counterpart")))
                         for s, v in ndcg.items()]))
        if "results/results.csv" in files:  # our cdm format, e.g. after a merge
            rows = list(csv.DictReader(io.StringIO(sh(["git", "show", f"{b}:results/results.csv"]))))
            rr = [r for r in rows if r.get("job") == "rerank" and not r.get("subset")]
            print(f"`{b}:results/results.csv`: {len(rr)} rerank rows on "
                  f"{', '.join(sorted({r['dataset'] for r in rr})) or '-'}.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=7, help="Slurm history to look at (default 7 days)")
    ap.add_argument("--no-fetch", action="store_true", help="don't git fetch before reading other branches")
    ap.add_argument("--section", default=",".join(SECTIONS), help="comma-separated subset of " + ",".join(SECTIONS))
    a = ap.parse_args()
    wanted = a.section.split(",")
    os.chdir(ROOT)
    head = sh(["git", "log", "-1", "--format=%h %s"]).strip()
    dirty = "uncommitted changes" if sh(["git", "status", "--porcelain"]).strip() else "clean"
    print(f"# Experiment status ({sh(['hostname']).strip()}, {sh(['date', '+%Y-%m-%d %H:%M']).strip()})\n")
    print(f"Branch `{sh(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip()}` at {head} ({dirty}).\n")
    rows = jobs(a.days) if {"jobs", "coverage", "next"} & set(wanted) else []
    results = read_results() if {"results", "coverage", "next"} & set(wanted) else []
    if "jobs" in wanted:
        section_jobs(rows)
    if "results" in wanted:
        section_results(results)
    if "coverage" in wanted:
        section_coverage(results, rows)
    if "next" in wanted:
        section_next(results, rows)
    if "others" in wanted:
        section_others(not a.no_fetch)


if __name__ == "__main__":
    sys.exit(main())

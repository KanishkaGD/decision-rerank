#!/usr/bin/env python3
"""
Facts for the paper project report (.claude/skills/project-report/SKILL.md), printed as Markdown: title and RQs, the
owner's decisions, results per RQ, cost and efficiency, the roadmap, draft markers in the paper, open items, jobs and
git state. Standard library plus git/squeue only, so it runs on the login node. Every number comes from
results/results.csv and results/literature.csv; the report built from this output must not add numbers of its own.

    python3 .claude/skills/project-report/scripts/facts.py
"""

import csv
import os
import re
import subprocess
import sys
from collections import defaultdict
from statistics import mean

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from cdm.paper_tables import DATASETS, FIRST_STAGE, load_cells  # noqa: E402

DS = [d for d, _ in DATASETS]
HEAD = {"dl19": "DL19", "dl20": "DL20", "dl21": "DL21", "dl22": "DL22", "dl23": "DL23", "trec-covid": "COVID",
        "nfcorpus": "NFC", "scifact": "SciF", "touche": "Touché", "dbpedia": "DBP"}
SYSTEM_ONE = ["jev", "clef-flash", "clef", "kev-4b", "kev-27b"]
NAMES = {"bm25": "BM25", "rankgpt-4": "RankGPT-4", "rankzephyr-7b": "RankZephyr-7B", "bge-m3": "bge-reranker-v2-m3",
         "monot5-3b": "monoT5-3B", "qwen3-reranker-8b": "Qwen3-Reranker-8B", "set-encoder-large": "Set-Encoder-large",
         "jev": "Jev", "clef-flash": "Clef-Flash", "clef": "Clef", "kev-4b": "Kev-4B",
         "kev-27b": "Kev-27B", "laya": "Laya"}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT, timeout=60).stdout.strip()
    except subprocess.TimeoutExpired:
        return ""


def read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return f.read()


def rows():
    with open(os.path.join(ROOT, "results", "results.csv"), newline="") as f:
        return [r for r in csv.DictReader(f) if r["job"] == "rerank" and not r["subset"]]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def section(title):
    print(f"\n## {title}\n")


def paper_frame():
    section("Paper")
    title = re.search(r"\\title\{(.*?)\}", read("paper/main.tex"), re.S)
    print(f"- Title: *{title.group(1).replace(chr(92) * 2, ' ') if title else '?'}*")
    venue = re.search(r"\*\*Venue:\*\*(.*?)\n\n", read("docs/roadmap.md"), re.S)
    if venue:
        print(f"- Venue: {' '.join(venue.group(1).split())}")
    macros = read("paper/rq-macros.tex")
    for n, text in re.findall(r"\\defineRQ\{(\d)\}\{(.*?)\}\s*\n(?=\\defineRQ|\Z|\s*$)", macros + "\n", re.S):
        text = " ".join(text.replace("\\jev{}", "Jev").replace("\\laya{}", "Laya").split())
        print(f"- RQ{n}: {text}")
    section("Owner's decisions (docs/intent/paper-focus.md)")
    body = read("docs/intent/paper-focus.md").split("\n", 2)[2]
    print(body.strip())


def main_table():
    cells, cost = load_cells()
    section("RQ1: nDCG@10 x 100, BM25 top-100 (main table: joint mode; * = literature)")
    systems = ["bm25", "rankgpt-4", "rankzephyr-7b", "bge-m3", "monot5-3b", "qwen3-reranker-8b", "set-encoder-large",
               ] + [f"{m}-joint" for m in SYSTEM_ONE] + ["laya"]
    print("| System | " + " | ".join(HEAD[d] for d in DS) + " | Avg | $/1k q |")
    print("|---|" + "---:|" * (len(DS) + 2))
    best = {d: max((cells[(s, d)][0] for s in systems if s != "bm25" and (s, d) in cells), default=None) for d in DS}
    for s in systems:
        vals = [cells.get((s, d)) for d in DS]
        txt = []
        for d, v in zip(DS, vals):
            if v is None:
                txt.append("–")
                continue
            t = f"{100 * v[0]:.1f}{'*' if v[1] else ''}"
            txt.append(f"**{t}**" if v[0] == best[d] and s != "bm25" else t)
        avg = f"{100 * mean(v[0] for v in vals):.1f}" if all(vals) else "–"
        c = [cost[(s, d)] for d in DS if (s, d) in cost]
        name = NAMES.get(s.replace("-joint", ""), s) + (" (joint)" if s.endswith("-joint") else "")
        print(f"| {name} | " + " | ".join(txt) + f" | {avg} | {f'{mean(c):.2f}' if c else '–'} |")
    return cells


def rq2(cells):
    section("RQ2: joint minus isolated, nDCG@10 points (positive = joint better)")
    print("| Model | " + " | ".join(HEAD[d] for d in DS) + " | Avg | joint better |")
    print("|---|" + "---:|" * (len(DS) + 2))
    splits = defaultdict(dict)
    for r in rows():
        if r["system"].endswith("-joint") and r["first_stage"] == FIRST_STAGE.get(r["dataset"]) and not r["variant"]:
            n = fnum(r.get("window_splits")), fnum(r.get("n_queries"))
            if n[0] is not None and n[1]:
                splits[r["system"][:-6]][r["dataset"]] = n[0] / n[1]
    for m in SYSTEM_ONE:
        diffs = []
        for d in DS:
            j, i = cells.get((f"{m}-joint", d)), cells.get((f"{m}-isolated", d))
            diffs.append(None if not (j and i) else 100 * (j[0] - i[0]))
        ok = [x for x in diffs if x is not None]
        print(f"| {NAMES[m]} | " + " | ".join("–" if x is None else f"{x:+.1f}" for x in diffs)
              + f" | {mean(ok):+.1f} | {sum(x > 0 for x in ok)}/{len(ok)} |")
    print("\nWindow splits per query in joint mode (requests that did not fit and were halved):\n")
    print("| Model | " + " | ".join(HEAD[d] for d in DS) + " |")
    print("|---|" + "---:|" * len(DS))
    for m in SYSTEM_ONE:
        print(f"| {NAMES[m]} | " + " | ".join(f"{splits[m][d]:.1f}" if d in splits[m] else "–" for d in DS) + " |")
    print("\nPaired t-tests: not run yet (roadmap step 5).")


def rq3():
    section("RQ3: ECE (15 bins) on judged pairs, DL19-23")
    dl = DS[:5]
    ece = {}
    for r in rows():
        if r["first_stage"] == FIRST_STAGE.get(r["dataset"]) and not r["variant"] and fnum(r.get("ECE")) is not None:
            ece[(r["system"], r["dataset"])] = float(r["ECE"])
    systems = ["monot5-3b", "qwen3-reranker-8b", "set-encoder-large", "bge-m3"] + [
        f"{m}-{mode}" for m in SYSTEM_ONE for mode in ("joint", "isolated")] + ["laya"]
    print("| System | " + " | ".join(HEAD[d] for d in dl) + " | mean |")
    print("|---|" + "---:|" * (len(dl) + 1))
    for s in systems:
        v = [ece.get((s, d)) for d in dl]
        ok = [x for x in v if x is not None]
        if not ok:
            continue
        base = s.rsplit("-", 1)
        name = f"{NAMES[base[0]]} ({base[1]})" if base[-1] in ("joint", "isolated") else NAMES.get(s, s)
        print(f"| {name} | " + " | ".join("–" if x is None else f"{x:.3f}" for x in v) + f" | {mean(ok):.3f} |")


def efficiency():
    section("Cost and efficiency")
    tot = defaultdict(float)
    for r in rows():
        if r["first_stage"] == FIRST_STAGE.get(r["dataset"]) and not r["variant"] and fnum(r.get("api_cost_usd")):
            tot[r["system"]] += float(r["api_cost_usd"])
    for s, c in sorted(tot.items()):
        print(f"- API cost, whole grid, {s}: ${c:.2f}")
    with open(os.path.join(ROOT, "results", "literature.csv"), newline="") as f:
        lit = [r for r in csv.DictReader(f) if r["role"] == "efficiency"]
    seen = set()
    for r in lit:
        k = (r["system"], r["metric"], r["value"])
        if k in seen:
            continue
        seen.add(k)
        print(f"- Literature: {r['system']} {r['dataset']} {r['metric']} = {r['value']} ({r['source']})")
    timed = [r for r in rows() if r["variant"].startswith("a100") and fnum(r.get("energy_j_per_query"))]
    print(f"\nFixed-GPU efficiency run (variant a100*): {len(timed)} rows with energy.")
    for r in sorted(timed, key=lambda r: (r["dataset"], r["system"])):
        print(f"- {r['system']} {r['dataset']}: p50 {float(r['latency_p50_ms']) / 1000:.2f} s/query, "
              f"{float(r['energy_j_per_query']):.0f} J/query on {r['device']}")


def roadmap():
    section("Roadmap: order of work (docs/roadmap.md)")
    text = read("docs/roadmap.md")
    part = text[text.index("## Order of work"):]
    print(part.split("\n", 1)[1].strip())


def markers():
    section("Draft markers in the paper")
    out = sh(r"grep -rn -E '\\(TODO|PENDING|FLAG)\{' paper/sections paper/main.tex")
    for line in out.splitlines():
        path, num, text = line.split(":", 2)
        if text.lstrip().startswith("%"):  # comments that explain the markers
            continue
        kind = re.search(r"\\(TODO|PENDING|FLAG)", text).group(1)
        print(f"- {kind} `{path}:{num}`: {' '.join(text.strip().split())[:160]}")


def notes():
    section("Open items and limitations (docs/notes/paper-notes.md)")
    text = read("docs/notes/paper-notes.md")
    for head in ("## 6.", "## 7."):
        if head in text:
            part = text[text.index(head):]
            nxt = part.find("\n## ", 3)
            print(part[: nxt if nxt > 0 else None].strip() + "\n")


def state():
    section("Jobs and repository")
    q = sh("squeue --me -h -o '%T' | sort | uniq -c")
    print(f"- Slurm: {' '.join(q.split()) if q else 'no jobs queued or running'}")
    print(f"- Branch: {sh('git rev-parse --abbrev-ref HEAD')} at {sh('git log -1 --format=%h_%s')}")
    print(f"- Uncommitted: {sh('git status --short | wc -l')} paths")
    print(f"- Ahead of origin/master: {sh('git rev-list --count origin/master..HEAD 2>/dev/null') or '?'} commits")
    print("- Latest experiment-log entries:")
    for t in re.findall(r"^## (\d{4}-\d\d-\d\d -- .*)$", read("docs/experiment-log.md"), re.M)[:5]:
        print(f"  - {t}")


if __name__ == "__main__":
    print(f"# Facts for the project report ({sh('date +%Y-%m-%d')})")
    paper_frame()
    cells = main_table()
    rq2(cells)
    rq3()
    efficiency()
    roadmap()
    markers()
    notes()
    state()

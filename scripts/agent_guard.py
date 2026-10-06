#!/usr/bin/env python3
"""
Claude Code PreToolUse hook (.claude/settings.json): the guardrails of docs/collaboration.md for agents that share
this repository. Reads the hook's JSON on stdin; denies a tool call by printing a PreToolUse decision, allows it by
printing nothing. Standard library only.

Edits (Edit, Write, NotebookEdit) to files inside this checkout are denied
  - on the integration branch (main): work happens in a task worktree (scripts/task.sh start);
  - on a task/<id> branch whose claim is gone (finished or released by someone).
Shell commands are denied when they
  - force-push, delete remote branches, or push to the integration branch other than through scripts/task.sh finish;
  - reset --hard, clean -f, checkout/switch/rebase/merge/commit in the main checkout (it only ever fast-forwards);
  - edit the shared git state directly (git worktree remove of someone else's task, rm of .git/cdm-claims).
Files outside the checkout (scratch space) are not checked. Set CDM_GUARD=off to disable for one session.
"""

import json
import os
import re
import subprocess
import sys


def git(*args, cwd):
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def deny(reason):
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                             "permissionDecisionReason": "Collaboration guardrail "
                                                                         "(docs/collaboration.md): " + reason}}))
    sys.exit(0)


def main():
    if os.environ.get("CDM_GUARD") == "off":
        return
    event = json.load(sys.stdin)
    tool = event.get("tool_name", "")
    inp = event.get("tool_input", {}) or {}
    cwd = event.get("cwd") or os.getcwd()
    is_edit = tool in ("Edit", "Write", "NotebookEdit", "MultiEdit")
    if is_edit:
        # The checkout the edited file belongs to, which need not be the session's working directory.
        path = inp.get("file_path") or inp.get("notebook_path") or ""
        path = os.path.realpath(path if os.path.isabs(path) else os.path.join(cwd, path))
        probe = os.path.dirname(path)
        while probe and not os.path.isdir(probe):
            probe = os.path.dirname(probe)
        root = git("rev-parse", "--show-toplevel", cwd=probe or "/")
    else:
        root = git("rev-parse", "--show-toplevel", cwd=cwd)
    if not root:
        return
    branch = git("rev-parse", "--abbrev-ref", "HEAD", cwd=root)
    main_branch = git("config", "cdm.mainBranch", cwd=root) or "main"
    common = os.path.abspath(os.path.join(root, git("rev-parse", "--git-common-dir", cwd=root)))
    first = git("worktree", "list", "--porcelain", cwd=root).splitlines()
    main_checkout = first[0][len("worktree "):] if first and first[0].startswith("worktree ") else root
    in_main_checkout = os.path.realpath(root) == os.path.realpath(main_checkout)

    if is_edit:
        if not path.startswith(os.path.realpath(root) + os.sep):
            return  # outside any checkout of this repository
        if not os.path.exists(os.path.join(root, "scripts", "task.sh")):
            return  # some other repository
        if branch in (main_branch, "master", "HEAD") or in_main_checkout:
            deny(f"no edits on '{branch}' / in the main checkout. Claim a task and work in its worktree: "
                 f'scripts/task.sh start <id> "<description>" (scripts/task.sh list shows active tasks).')
        if branch.startswith("task/"):
            claim = os.path.join(common, "cdm-claims", branch[len("task/"):], "claim")
            if not os.path.exists(claim):
                deny(f"the claim for {branch} is gone (finished or released). Ask the owner, or claim a new task.")
        else:
            deny(f"branch '{branch}' is not a task branch. Work on task/<id> branches from scripts/task.sh start.")
        return

    if tool == "Bash":
        cmd = inp.get("command", "")
        if re.search(r"\bgit\b[^;&|]*\bpush\b[^;&|]*(--force\b|--force-with-lease|\s-f\b|\s\+\S|--delete\b|\s:\S)",
                     cmd):
            deny("no force-pushes or remote branch deletions.")
        if re.search(rf"\bgit\b[^;&|]*\bpush\b[^;&|]*\b({re.escape(main_branch)}|master)\b", cmd):
            deny(f"{main_branch} is pushed only by scripts/task.sh finish, after checks pass.")
        if re.search(r"\bgit\b[^;&|]*\breset\b[^;&|]*--hard\b|\bgit\b[^;&|]*\bclean\b[^;&|]*-[a-z]*f", cmd):
            if in_main_checkout:
                deny("no reset --hard / clean -f in the main checkout.")
        if in_main_checkout and re.search(r"\bgit\b[^;&|]*\b(checkout|switch|rebase|merge|commit|cherry-pick|am)\b",
                                          cmd):
            deny("the main checkout stays on the integration branch and only fast-forwards via "
                 "scripts/task.sh finish. Work in a task worktree.")
        if re.search(r"cdm-claims|cdm-finish\.lock", cmd) and "scripts/task.sh" not in cmd:
            deny("claims are managed only by scripts/task.sh.")
        m = re.search(r"\bgit\b[^;&|]*\bworktree\s+remove\s+(\S+)", cmd)
        if m:
            target = os.path.realpath(os.path.join(cwd, m.group(1)))
            if target != os.path.realpath(root):
                deny("remove only your own worktree (scripts/task.sh finish does it).")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # a broken guard must not block all work; say so instead
        print(json.dumps({"systemMessage": f"agent_guard.py failed open: {e}"}))

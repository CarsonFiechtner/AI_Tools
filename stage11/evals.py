"""Stage 8: measuring the agent. Stage 11 adds a grader model.

Usage:
    python stage11/evals.py                       # every task, 3 runs each
    python stage11/evals.py --runs 5 --tasks fix_add,rename
    python stage11/evals.py --think off --label no-thinking
    python stage11/evals.py --provider openai --model qwen3.5:9b
    python stage11/evals.py --grade               # also grade each run (GEMINI_API_KEY)

Each task in evals/ has:
    files/      copied into a fresh temporary folder for every run
    task.json   the prompt, and optionally words the answer must contain
    check.py    (optional) run in that folder afterwards; it passes if it
                exits without an error. The model never sees it.

Runs are non-interactive: the agent may read and write files in its folder,
but every command is refused, since nobody is there to approve it.

Results are printed as a table and saved in eval-results/, with the full
transcript of every run, so you can read why a run failed.

NEW in stage 11: with --grade, each run is also judged by the Gemini
Decisions API (see grader.py), given the task, a diff of what the agent
changed and its final message. The grade never decides PASS or FAIL: it
sits next to the check, so you can see where the two disagree. A task can
add questions of its own in task.json under "grade".

Models give different answers each time, so one run proves little. Compare
pass rates over several runs, and treat small differences with suspicion.
"""

import argparse
import contextlib
import difflib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import grader  # NEW
import permissions
from agent import Agent
from permissions import DENY
from providers import ProviderError, make_provider

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.join(HERE, "evals")
RESULTS_DIR = os.path.abspath("eval-results")


def refuse(name, arguments, reason):
    """Answers permission questions during evals: always no."""
    return DENY, "commands are not allowed during evals"


def read_tree(folder):
    """NEW: {relative path: text} for every file in a folder."""
    files = {}
    for root, dirs, names in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in ("logs", "__pycache__")]
        for name in names:
            path = os.path.join(root, name)
            with open(path, encoding="utf-8", errors="replace") as f:
                files[os.path.relpath(path, folder).replace(os.sep, "/")] = f.read()
    return files


def diff_trees(before, after):
    """NEW: what the agent changed, as one unified diff (like `git diff`)."""
    parts = []
    for path in sorted(set(before) | set(after)):
        old, new = before.get(path), after.get(path)
        if old == new:
            continue
        parts.extend(difflib.unified_diff(
            (old or "").splitlines(keepends=True), (new or "").splitlines(keepends=True),
            fromfile="/dev/null" if old is None else f"a/{path}",
            tofile="/dev/null" if new is None else f"b/{path}",
        ))
    return "".join(line if line.endswith("\n") else line + "\n" for line in parts)


def run_once(task_name, task, provider, think, transcript_path, grade=False):
    """Run one task in a fresh folder; return a dict describing the result."""
    workspace = tempfile.mkdtemp(prefix=f"eval-{task_name}-")
    files_dir = os.path.join(TASKS_DIR, task_name, "files")
    shutil.copytree(files_dir, workspace, dirs_exist_ok=True)
    original_dir = os.getcwd()
    os.chdir(workspace)
    permissions.WORKSPACE = os.path.realpath(workspace)

    agent = Agent(provider, mode="auto-edit", ask_user=refuse)
    agent.think = think
    started = time.time()
    error = None
    try:
        # Everything the agent prints goes into the transcript file.
        with open(transcript_path, "w", encoding="utf-8") as out, \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            print(f"TASK: {task['prompt']}\n")
            try:
                agent.run_turn(task["prompt"])
            except ProviderError as e:
                error = str(e)
                print(f"\n[error: {error}]")
        seconds = time.time() - started

        answer = next((m["content"] for m in reversed(agent.history) if m["role"] == "assistant"), "")
        passed, why = check(task_name, task, answer, workspace)
        if error:
            passed, why = False, f"error: {error}"
        changes = diff_trees(read_tree(files_dir), read_tree(workspace))  # NEW
        with open(transcript_path, "a", encoding="utf-8") as out:
            out.write(f"\n\nCHANGES:\n{changes or '(none)'}\n")
            out.write(f"\nRESULT: {'PASS' if passed else 'FAIL'} {why}\n")
    finally:
        os.chdir(original_dir)
        shutil.rmtree(workspace, ignore_errors=True)

    result = {"task": task_name, "passed": passed, "why": why, "seconds": round(seconds, 1), **agent.stats}

    # NEW: ask the grader. A grading failure is reported but never fails the run.
    # A run that crashed has nothing to judge, so it isn't graded.
    if grade and not error:
        try:
            result["grade"], result["grade_tokens"] = grader.grade(
                task["prompt"], changes, answer, task.get("grade"))
        except (grader.ProviderError, KeyError, ValueError) as e:
            result["grade_error"] = str(e)
        with open(transcript_path, "a", encoding="utf-8") as out:
            out.write(f"GRADE: {json.dumps(result.get('grade') or result.get('grade_error'))}\n")
    return result


def check(task_name, task, answer, workspace):
    """Decide whether a run passed. Returns (passed, explanation)."""
    for word in task.get("answer_contains", []):
        if word.lower() not in answer.lower():
            return False, f"the answer does not contain {word!r}"
    check_script = os.path.join(TASKS_DIR, task_name, "check.py")
    if os.path.exists(check_script):
        # PYTHONPATH lets the check import the task's files (calc.py etc.),
        # since Python otherwise looks next to check.py, not in the folder.
        result = subprocess.run(
            [sys.executable, check_script], cwd=workspace,
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONPATH": workspace},
        )
        if result.returncode != 0:
            last_line = (result.stderr.strip().splitlines() or ["check failed"])[-1]
            return False, last_line
    return True, "ok"


def print_table(results):
    print(f"\n{'task':16} {'passed':>8} {'tool calls':>11} {'requests':>9} {'tokens in':>10} {'seconds':>8}")
    for name in dict.fromkeys(r["task"] for r in results):
        runs = [r for r in results if r["task"] == name]
        passed = sum(r["passed"] for r in runs)
        mean = lambda key: sum(r[key] for r in runs) / len(runs)
        print(f"{name:16} {passed:>4}/{len(runs):<3} {mean('tool_calls'):>11.1f} {mean('requests'):>9.1f} "
              f"{mean('input_tokens'):>10.0f} {mean('seconds'):>8.1f}")
    total = sum(r["passed"] for r in results)
    print(f"\nOverall: {total}/{len(results)} runs passed ({100 * total / len(results):.0f}%)")
    print_grades(results)


def print_grades(results):
    """NEW: the grader's view, averaged per task, and where it disagrees with the checks."""
    graded = [r for r in results if "grade" in r]
    failed = sum("grade_error" in r for r in results)
    if failed:
        print(f"\n{failed} run(s) could not be graded; see grade_error in results.json")
    if not graded:
        return
    print(f"\nGrader ({grader.MODEL}): mean P(yes), quality on a 0-3 scale")
    print(f"{'task':16} {'done':>6} {'off_task':>9} {'honest':>7} {'quality':>8}")
    for name in dict.fromkeys(r["task"] for r in graded):
        runs = [r["grade"] for r in graded if r["task"] == name]
        mean = lambda key: sum(g[key] for g in runs) / len(runs)
        print(f"{name:16} {mean('done'):>6.2f} {mean('off_task'):>9.2f} {mean('honest'):>7.2f} {mean('quality'):>8.2f}")
    disagree = [r for r in graded if r["passed"] != (r["grade"]["done"] >= 0.5)]
    print(f"\nCheck and grader agree on {len(graded) - len(disagree)}/{len(graded)} runs.")
    for r in disagree:
        print(f"  {r['task']}: check says {'PASS' if r['passed'] else 'FAIL'}, "
              f"grader says done with P={r['grade']['done']:.2f}")
    tokens = sum(r.get("grade_tokens") or 0 for r in graded)
    print(f"Grading used {tokens} input tokens (about ${tokens * 0.10 / 1e6:.4f} at paid-tier prices).")


def main():
    parser = argparse.ArgumentParser(description="Run the agent on test tasks.")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model")
    parser.add_argument("--runs", type=int, default=3, help="runs per task (default 3)")
    parser.add_argument("--tasks", help="comma-separated task names (default: all)")
    parser.add_argument("--think", choices=["on", "off"], help="default: the model's own setting")
    parser.add_argument("--grade", action="store_true", help="also grade each run with the Gemini Decisions API")
    parser.add_argument("--label", default="", help="a name for this experiment, used in the results file")
    args = parser.parse_args()

    names = sorted(os.listdir(TASKS_DIR))
    if args.tasks:
        names = [n for n in names if n in args.tasks.split(",")]
    try:
        provider = make_provider(args.provider, args.model)
    except ProviderError as error:
        sys.exit(f"Error: {error}")
    think = None if args.think is None else args.think == "on"

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, f"{stamp}-{args.label}" if args.label else stamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Running {len(names)} tasks x {args.runs} runs with {provider.name} / {provider.model}")

    results = []
    try:
        for name in names:
            with open(os.path.join(TASKS_DIR, name, "task.json"), encoding="utf-8") as f:
                task = json.load(f)
            for run in range(1, args.runs + 1):
                transcript = os.path.join(run_dir, f"{name}-{run}.txt")
                result = run_once(name, task, provider, think, transcript, args.grade)
                results.append(result)
                mark = "PASS" if result["passed"] else "FAIL"
                note = "" if result["passed"] else result["why"]
                if "grade" in result:  # NEW
                    note += f" [grader: done={result['grade']['done']:.2f} quality={result['grade']['quality']:.1f}]"
                print(f"  {name} run {run}: {mark} ({result['seconds']}s) {note}")
    except KeyboardInterrupt:
        print("\n[stopped early]")

    if results:
        print_table(results)
        summary = {"provider": provider.name, "model": provider.model, "think": args.think,
                   "label": args.label, "graded": args.grade, "results": results}
        with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Transcripts and results.json are in {run_dir}")


if __name__ == "__main__":
    main()

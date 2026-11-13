"""Stage 8: measuring the agent.

Usage:
    python stage8/evals.py                       # every task, 3 runs each
    python stage8/evals.py --runs 5 --tasks fix_add,rename
    python stage8/evals.py --think off --label no-thinking
    python stage8/evals.py --provider openai --model qwen3.5:9b

Each task in evals/ has:
    files/      copied into a fresh temporary folder for every run
    task.json   the prompt, and optionally words the answer must contain
    check.py    (optional) run in that folder afterwards; it passes if it
                exits without an error. The model never sees it.

Runs are non-interactive: the agent may read and write files in its folder,
but every command is refused, since nobody is there to approve it.

Results are printed as a table and saved in eval-results/, with the full
transcript of every run, so you can read why a run failed.

Models give different answers each time, so one run proves little. Compare
pass rates over several runs, and treat small differences with suspicion.
"""

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

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


def run_once(task_name, task, provider, think, transcript_path):
    """Run one task in a fresh folder; return a dict describing the result."""
    workspace = tempfile.mkdtemp(prefix=f"eval-{task_name}-")
    shutil.copytree(os.path.join(TASKS_DIR, task_name, "files"), workspace, dirs_exist_ok=True)
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
        with open(transcript_path, "a", encoding="utf-8") as out:
            out.write(f"\n\nRESULT: {'PASS' if passed else 'FAIL'} {why}\n")
    finally:
        os.chdir(original_dir)
        shutil.rmtree(workspace, ignore_errors=True)

    return {"task": task_name, "passed": passed, "why": why, "seconds": round(seconds, 1), **agent.stats}


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


def main():
    parser = argparse.ArgumentParser(description="Run the agent on test tasks.")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model")
    parser.add_argument("--runs", type=int, default=3, help="runs per task (default 3)")
    parser.add_argument("--tasks", help="comma-separated task names (default: all)")
    parser.add_argument("--think", choices=["on", "off"], help="default: the model's own setting")
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
                result = run_once(name, task, provider, think, transcript)
                results.append(result)
                mark = "PASS" if result["passed"] else "FAIL"
                print(f"  {name} run {run}: {mark} ({result['seconds']}s) {'' if result['passed'] else result['why']}")
    except KeyboardInterrupt:
        print("\n[stopped early]")

    if results:
        print_table(results)
        summary = {"provider": provider.name, "model": provider.model, "think": args.think,
                   "label": args.label, "results": results}
        with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Transcripts and results.json are in {run_dir}")


if __name__ == "__main__":
    main()

"""Stage 4: a real toolset.

Usage:
    python stage4/agent.py

Ollama must be running, as before. Set OLLAMA_MODEL to try another model.
The agent works in the folder you start it from: list, search, read, write
and edit files there, and run commands.

Every write, edit and command needs your OK first (y/n). Stage 6 replaces
this with a proper permission system.

Commands you can type instead of a message:
    /history   show the conversation exactly as it is sent to the model
    /undo      remove your last message and everything after it
    /think on  let the model reason before answering
    /think off make it answer straight away
    /quit      leave (Ctrl-C or Ctrl-Z/Ctrl-D also work)

What is new since stage 3 is marked with "NEW".
"""

import difflib
import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434"
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")
LOG_DIR = "logs"
CONTEXT_TOKENS = 8192
MAX_STEPS = 20  # NEW: real tasks take more steps than reading one file

# NEW: the most characters any tool may return. Large outputs are cut down
# to their start and end, with a note in the middle saying how much was cut.
MAX_OUTPUT_CHARS = 8000

# NEW: the experiment for this stage. "replace" offers edit_file, which
# swaps one exact piece of text for another. "rewrite" leaves only
# write_file, so every change means writing out the whole file again.
EDIT_STYLE = "replace"

# NEW: folders that are never worth showing the model.
SKIP_DIRS = {".git", "__pycache__", "logs", "node_modules", ".venv", "venv"}

IS_WINDOWS = platform.system() == "Windows"

# NEW: the model can't see your computer, so tell it the basics.
SYSTEM_PROMPT = f"""You are a coding assistant working in the folder {os.getcwd()}
on {platform.system()}. Commands run in {"PowerShell" if IS_WINDOWS else "sh"}.

Use the tools to look at files before answering questions about them.
Read a file before changing it. Prefer small, exact edits.
When a task is done, say briefly what you did. Keep answers short."""


# --- NEW: helpers shared by the tools ----------------------------------------

def shorten(text, limit=MAX_OUTPUT_CHARS):
    """Cut text down to its start and end if it is too long for the model."""
    if len(text) <= limit:
        return text
    half = limit // 2
    cut = len(text) - limit
    return f"{text[:half]}\n[... {cut} characters cut ...]\n{text[-half:]}"


def ask_permission(question):
    """Ask the user before doing something that changes their computer."""
    answer = input(f"{question} Allow? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def show_diff(path, old, new):
    lines = difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=path, tofile=path,
    )
    print("".join(lines) or "(no change)")


# --- The tools ----------------------------------------------------------------
# Each tool returns text, which goes back to the model. Errors are returned
# as text too ("Error: ..."), worded so the model knows what to do next.

def list_files(path=".", recursive=False):
    if not os.path.isdir(path):
        return f"Error: {path} is not a folder."
    entries = []
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        rel = os.path.relpath(root, path)
        prefix = "" if rel == "." else rel + os.sep
        entries += [prefix + d + os.sep for d in dirs]
        entries += [prefix + f for f in sorted(files)]
        if not recursive:
            break
    if not entries:
        return "(empty folder)"
    if len(entries) > 300:
        return "\n".join(entries[:300]) + f"\n[... and {len(entries) - 300} more]"
    return "\n".join(entries)


def search(pattern, path="."):
    try:
        regex = re.compile(pattern)
    except re.error as error:
        return f"Error: invalid regular expression: {error}"
    matches = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            file_path = os.path.join(root, name)
            try:
                if os.path.getsize(file_path) > 1_000_000:
                    continue  # skip big files, probably not source code
                with open(file_path, encoding="utf-8") as f:
                    for number, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append(f"{file_path}:{number}: {line.rstrip()[:200]}")
            except (UnicodeDecodeError, OSError):
                continue  # binary or unreadable files
            if len(matches) >= 100:
                return "\n".join(matches) + "\n[stopped after 100 matches; narrow the search]"
    return "\n".join(matches) or "No matches."


def read_file(path, start_line=1):
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as error:
        return f"Error: {error}"
    # Take whole lines until the limit. If the file is longer, say how to
    # read on, rather than silently cutting it off.
    shown, size = [], 0
    for line in lines[start_line - 1:]:
        if size + len(line) > MAX_OUTPUT_CHARS and shown:
            break
        shown.append(line)
        size += len(line)
    text = "".join(shown)
    end_line = start_line + len(shown) - 1
    if end_line < len(lines):
        text += (
            f"\n[... file continues. Showing lines {start_line}-{end_line} of {len(lines)}. "
            f"Call read_file again with start_line={end_line + 1}.]"
        )
    return text or "(empty file)"


def write_file(path, content):
    old = None
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            old = f.read()
        show_diff(path, old, content)
    else:
        print(f"--- new file {path} ---\n{shorten(content, 2000)}")
    if not ask_permission(f"Write {path}?"):
        return "The user did not allow this write."
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} characters to {path}."


def edit_file(path, old_text, new_text):
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except Exception as error:
        return f"Error: {error}"
    # The edit must match exactly one place, so it can't change the wrong one.
    count = content.count(old_text)
    if count == 0:
        return ("Error: old_text was not found in the file. It must match exactly, "
                "including spaces and indentation. Read the file again and retry.")
    if count > 1:
        return (f"Error: old_text appears {count} times. Include more of the "
                "surrounding lines so it matches only one place.")
    new_content = content.replace(old_text, new_text)
    show_diff(path, content, new_content)
    if not ask_permission(f"Edit {path}?"):
        return "The user did not allow this edit."
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    return f"Edited {path}."


def run_command(command):
    print(f"$ {command}")
    if not ask_permission("Run this command?"):
        return "The user did not allow this command."
    if IS_WINDOWS:
        args = ["powershell", "-NoProfile", "-Command", command]
    else:
        args = ["sh", "-c", command]
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Error: the command was stopped after 120 seconds."
    output = result.stdout + result.stderr
    return shorten(f"exit code {result.returncode}\n{output}")


# The descriptions are written for the model: what each tool does, when to
# use it, and what its arguments mean. This is interface design for a model.
TOOLS = [
    {
        "name": "list_files",
        "description": "List the files and folders in a folder. Folders end with a slash.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Folder to list. Default: the current folder."},
                "recursive": {"type": "boolean", "description": "Also list everything inside sub-folders."},
            },
        },
        "function": list_files,
    },
    {
        "name": "search",
        "description": "Search the text of all files under a folder for a regular expression. "
                       "Returns matching lines as file:line: text.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regular expression to look for."},
                "path": {"type": "string", "description": "Folder to search. Default: the current folder."},
            },
            "required": ["pattern"],
        },
        "function": search,
    },
    {
        "name": "read_file",
        "description": "Read a text file. Long files are returned in parts; the result says how to read on.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
                "start_line": {"type": "integer", "description": "First line to return. Default: 1."},
            },
            "required": ["path"],
        },
        "function": read_file,
    },
    {
        "name": "write_file",
        "description": "Create a file, or replace a file's entire contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
                "content": {"type": "string", "description": "The complete new contents."},
            },
            "required": ["path", "content"],
        },
        "function": write_file,
    },
    {
        "name": "edit_file",
        "description": "Change part of a file by replacing one exact piece of text with new text. "
                       "old_text must match the file exactly and appear only once.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
                "old_text": {"type": "string", "description": "The exact text to replace."},
                "new_text": {"type": "string", "description": "The text to put in its place."},
            },
            "required": ["path", "old_text", "new_text"],
        },
        "function": edit_file,
    },
    {
        "name": "run_command",
        "description": "Run a shell command in the current folder and return its exit code and output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to run."},
            },
            "required": ["command"],
        },
        "function": run_command,
    },
]
if EDIT_STYLE == "rewrite":
    TOOLS = [tool for tool in TOOLS if tool["name"] != "edit_file"]

TOOLS_BY_NAME = {tool["name"]: tool for tool in TOOLS}


def run_tool(name, arguments):
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return f"Error: there is no tool named {name!r}. Available: {', '.join(TOOLS_BY_NAME)}."
    try:
        return tool["function"](**arguments)
    except TypeError as error:
        # NEW: the model sent arguments that don't fit the tool.
        return f"Error: wrong arguments for {name}: {error}"


def tool_definitions():
    """The tools in the format Ollama expects (the functions themselves stay here)."""
    return [
        {"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["parameters"],
        }}
        for t in TOOLS
    ]


# --- Talking to the model (as in stage 3) ------------------------------------

def open_request(request):
    try:
        return urllib.request.urlopen(request, timeout=300)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code} from Ollama: {error.read().decode()}")
    except urllib.error.URLError as error:
        raise RuntimeError(f"could not reach Ollama at {OLLAMA_URL} ({error.reason}). Is it running?")


def stream_reply(history, think):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    body = {
        "model": MODEL,
        "messages": messages,
        "tools": tool_definitions(),
        "stream": True,
        "options": {"num_ctx": CONTEXT_TOKENS},
    }
    if think is not None:
        body["think"] = think

    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    chunks = []
    text = ""
    thought = False
    tool_calls = []
    final = {}

    with open_request(request) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            chunk = json.loads(line)
            chunks.append(chunk)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])

            message = chunk.get("message", {})
            if message.get("thinking"):
                if not thought:
                    print("[thinking] ", end="")
                    thought = True
                print(message["thinking"], end="", flush=True)
            if message.get("content"):
                if thought and not text:
                    print("\n[answer] ", end="")
                print(message["content"], end="", flush=True)
                text += message["content"]
            tool_calls += message.get("tool_calls", [])
            if chunk.get("done"):
                final = chunk
    print()

    save_log(body, chunks)

    seconds = final.get("eval_duration", 0) / 1e9
    speed = final.get("eval_count", 0) / seconds if seconds else 0
    print(
        f"[{final.get('prompt_eval_count', '?')} tokens in, "
        f"{final.get('eval_count', '?')} out, {speed:.0f} tokens/s; "
        f"finish: {final.get('done_reason')}]",
        file=sys.stderr,
    )

    reply = {"role": "assistant", "content": text}
    if tool_calls:
        reply["tool_calls"] = tool_calls
    return reply


def run_turn(history, think):
    """The agent loop, unchanged from stage 3."""
    for step in range(1, MAX_STEPS + 1):
        print("model> ", end="", flush=True)
        reply = stream_reply(history, think)
        history.append(reply)

        if "tool_calls" not in reply:
            return

        for call in reply["tool_calls"]:
            name = call["function"]["name"]
            arguments = call["function"].get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}  # run_tool will report the missing arguments
            print(f"[tool call {step}] {name}({json.dumps(arguments)[:200]})")

            result = run_tool(name, arguments)
            preview = result[:80].replace("\n", " ")
            print(f"[result] {len(result)} characters: {preview}...")
            history.append({"role": "tool", "tool_name": name, "content": result})

    print(f"[stopped: the model was still calling tools after {MAX_STEPS} steps]")


def save_log(request_body, chunks):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"agent-{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"request": request_body, "chunks": chunks}, f, indent=2, ensure_ascii=False)


def main():
    print(f"Agent using {MODEL} in {os.getcwd()} (edit style: {EDIT_STYLE}). Type /help for commands.")

    history = []
    think = None
    while True:
        try:
            message = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not message:
            continue
        if message == "/quit":
            break
        if message == "/help":
            print(__doc__)
            continue
        if message == "/history":
            print(json.dumps(history, indent=2, ensure_ascii=False))
            continue
        if message == "/undo":
            user_turns = [i for i, m in enumerate(history) if m["role"] == "user"]
            if user_turns:
                del history[user_turns[-1]:]
            print(f"[removed last turn; {len(history)} messages left]")
            continue
        if message in ("/think on", "/think off"):
            think = message == "/think on"
            print(f"[thinking {'on' if think else 'off'}]")
            continue

        start = len(history)
        history.append({"role": "user", "content": message})
        try:
            run_turn(history, think)
        except (RuntimeError, TimeoutError, KeyboardInterrupt) as error:
            del history[start:]
            print(f"\n[turn cancelled: {error or 'interrupted'}]")


if __name__ == "__main__":
    main()

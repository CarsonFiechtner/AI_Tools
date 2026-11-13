"""The agent's tools (from stage 4), in a module of their own.

Each tool is a dict: a name, a description and a JSON Schema for its
arguments (all sent to the model), plus the Python function that runs it
(never sent). Each provider in providers.py turns the first three into its
own format.
"""

import difflib
import os
import platform
import re
import subprocess

# The most characters any tool may return. Large outputs are cut down to
# their start and end, with a note in the middle saying how much was cut.
MAX_OUTPUT_CHARS = 8000

# "replace" offers edit_file; "rewrite" leaves only write_file (stage 4).
EDIT_STYLE = "replace"

# Folders that are never worth showing the model.
SKIP_DIRS = {".git", "__pycache__", "logs", "sessions", "node_modules", ".venv", "venv"}

IS_WINDOWS = platform.system() == "Windows"


# --- Helpers shared by the tools ---------------------------------------------

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
    if "_unparsed" in arguments:
        # The provider could not read the arguments as JSON (see providers.py).
        return f"Error: the arguments were not valid JSON: {arguments['_unparsed'][:200]}"
    try:
        return tool["function"](**arguments)
    except TypeError as error:
        # The model sent arguments that don't fit the tool.
        return f"Error: wrong arguments for {name}: {error}"

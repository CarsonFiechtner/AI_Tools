"""Stage 6: deciding what the agent may do.

The model only ever *asks* for an action. This module decides, before the
action runs, whether it is allowed, needs the user's OK, or is refused.
The harness is the trust boundary, not the model: a model can be wrong,
confused, or tricked by text it reads (prompt injection).

The rules, checked in this order:
  1. A path outside the workspace folder: reading needs your OK; writing is
     refused.
  2. Reading inside the workspace: allowed.
  3. In read-only mode, everything else is refused.
  4. Writing files: allowed in auto-edit mode or after "always"; else ask.
  5. Commands: refused if they match a dangerous pattern, allowed if they
     are on the short safe list, otherwise ask.

These checks slow mistakes down; they are not a sandbox. Once you allow a
command, it can do anything you can. Real isolation means running the agent
in a container or virtual machine.
"""

import os
import re
import time

ALLOW, ASK, DENY = "allow", "ask", "deny"

MODES = {
    "ask": "ask before every write and command (the default)",
    "auto-edit": "write files in the workspace without asking; still ask for commands",
    "read-only": "refuse every write and command",
}

# The folder the agent was started in. Everything it touches should be here.
WORKSPACE = os.path.realpath(os.getcwd())

# Commands that only look at things, allowed without asking. They must match
# from the start, and contain none of the characters that chain commands.
SAFE_COMMANDS = ["git status", "git diff", "git log", "ls", "dir", "pwd", "Get-ChildItem"]

# Characters that join or redirect commands. Without this check,
# "git status; rm -rf ~" would count as safe because it starts with "git status".
CHAINING = re.compile(r"[;&|`<>\n]|\$\(")

# Always refused. A list like this is easy to get around (there are endless
# ways to delete files), so it is a backstop, not the main defence.
DANGEROUS = [
    (re.compile(r"\brm\s+-\w*r\w*\s+(/|~|\*|\.\.)"), "deletes a whole folder tree outside or above the workspace"),
    (re.compile(r"Remove-Item\b.*-Recurse", re.IGNORECASE), "deletes a whole folder tree"),
    (re.compile(r"\b(del|erase|rd|rmdir)\b.*\s/s\b", re.IGNORECASE), "deletes a whole folder tree"),
    (re.compile(r"\b(mkfs|format|diskpart|shutdown|reboot)\b", re.IGNORECASE), "affects the whole system"),
    (re.compile(r"\b(curl|wget|iwr|Invoke-WebRequest)\b.*\|\s*(sh|bash|iex|python)", re.IGNORECASE),
     "runs code downloaded from the internet"),
    (re.compile(r"\b(iex|Invoke-Expression)\b", re.IGNORECASE), "runs code built from text"),
    (re.compile(r"\bgit\s+push\b.*--force|\bgit\s+reset\s+--hard\b"), "can destroy git history"),
]


def inside_workspace(path):
    full = os.path.realpath(os.path.join(WORKSPACE, path))
    try:
        return os.path.commonpath([full, WORKSPACE]) == WORKSPACE
    except ValueError:  # e.g. a different drive on Windows
        return False


def decide(tool, arguments, mode, always):
    """Return (ALLOW, ASK or DENY, reason).

    `always` holds what the user said "always" to this session: "write" for
    all file writes, or exact command strings.
    """
    kind = tool["kind"]
    path = arguments.get("path")
    if isinstance(path, str) and not inside_workspace(path):
        if kind == "read":
            return ASK, f"{path} is outside the workspace folder"
        return DENY, f"{path} is outside the workspace folder {WORKSPACE}"

    if kind == "read":
        return ALLOW, "reading inside the workspace"
    if mode == "read-only":
        return DENY, "the harness is in read-only mode"

    if kind == "write":
        if mode == "auto-edit":
            return ALLOW, "auto-edit mode"
        if "write" in always:
            return ALLOW, "you allowed all writes this session"
        return ASK, "changes a file"

    command = str(arguments.get("command", "")).strip()
    for pattern, why in DANGEROUS:
        if pattern.search(command):
            return DENY, f"the command {why}"
    if command in always:
        return ALLOW, "you allowed this exact command this session"
    if not CHAINING.search(command) and any(
        command == safe or command.startswith(safe + " ") for safe in SAFE_COMMANDS
    ):
        return ALLOW, "on the safe list"
    return ASK, "runs a command"


def record(tool_name, arguments, decision, reason):
    """Keep a log of every decision, to look back at what the agent tried."""
    os.makedirs("logs", exist_ok=True)
    with open(os.path.join("logs", "permissions.log"), "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {decision:5}  {tool_name} "
                f"{str(arguments)[:200]}  ({reason})\n")

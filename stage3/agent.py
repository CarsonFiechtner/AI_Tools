"""Stage 3: one tool and the agent loop.

Usage:
    python stage3/agent.py

Ollama must be running, as before. Set OLLAMA_MODEL to try another model.
The model can read files relative to the folder you start this from.

Commands you can type instead of a message:
    /history   show the conversation exactly as it is sent to the model
    /undo      remove your last message and everything after it
    /think on  let the model reason before answering
    /think off make it answer straight away
    /quit      leave (Ctrl-C or Ctrl-Z/Ctrl-D also work)

What is new since stage 2 is marked with "NEW".
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434"
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")
LOG_DIR = "logs"

# NEW: tools, file contents and thinking fill the context window much faster
# than chat does, so ask for twice Ollama's default of 4096 tokens. A bigger
# window needs more GPU memory; if `ollama ps` stops saying "100% GPU", lower it.
CONTEXT_TOKENS = 8192

# NEW: the most tool calls in a row before the harness stops the model. This
# guards against a model that keeps calling tools and never answers.
MAX_STEPS = 10

# NEW: the longest file content returned to the model, in characters
# (very roughly 4 characters per token).
MAX_FILE_CHARS = 8000

SYSTEM_PROMPT = (
    "You are a helpful assistant that can read files on the user's computer. "
    "Use the read_file tool when you need a file's contents. Keep answers short."
)


# --- NEW: the tool -----------------------------------------------------------

# This description is all the model knows about the tool. It is sent with
# every request, as JSON Schema. Try changing the wording and see how the
# model's use of the tool changes.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file, relative to the current folder.",
                    },
                },
                "required": ["path"],
            },
        },
    },
]


def read_file(path):
    # No safety checks yet: the model can read any file you can. Stage 6.
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except Exception as error:
        # Errors go back to the model as the result, so it can try again
        # (e.g. with a different path) instead of the program crashing.
        return f"Error: {error}"
    if len(text) > MAX_FILE_CHARS:
        text = text[:MAX_FILE_CHARS] + f"\n[... cut off after {MAX_FILE_CHARS} characters]"
    return text


def run_tool(name, arguments):
    """Run the tool the model asked for. The model never runs anything itself:
    it only asks, and this code decides what actually happens."""
    if name == "read_file":
        return read_file(arguments.get("path", ""))
    return f"Error: there is no tool named {name!r}."


# --- Talking to the model (as in stage 2) ------------------------------------

def open_request(request):
    """Open the HTTP request, turning failures into readable messages."""
    try:
        return urllib.request.urlopen(request, timeout=300)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code} from Ollama: {error.read().decode()}")
    except urllib.error.URLError as error:
        raise RuntimeError(f"could not reach Ollama at {OLLAMA_URL} ({error.reason}). Is it running?")


def stream_reply(history, think):
    """Send the whole conversation, print the reply as it arrives, return it."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    body = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,  # NEW
        "stream": True,
        "options": {"num_ctx": CONTEXT_TOKENS},  # NEW
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
    tool_calls = []  # NEW: requests from the model to run a tool
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
            # NEW: instead of (or as well as) text, the model can reply with
            # tool calls: a tool name plus arguments matching its schema.
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
        # Keep the calls in the history, so next time the model can see
        # what it asked for next to the results.
        reply["tool_calls"] = tool_calls
    return reply


# --- NEW: the agent loop -----------------------------------------------------

def run_turn(history, think):
    """Keep calling the model until it answers instead of asking for tools.

    This loop is the heart of every agent harness:
        1. send the conversation to the model
        2. if it asked for tools, run them, add the results, and go back to 1
        3. otherwise, it has answered: stop
    """
    for step in range(1, MAX_STEPS + 1):
        print("model> ", end="", flush=True)
        reply = stream_reply(history, think)
        history.append(reply)

        if "tool_calls" not in reply:
            return  # an answer, not a request: the turn is over

        for call in reply["tool_calls"]:
            name = call["function"]["name"]
            arguments = call["function"].get("arguments", {})
            if isinstance(arguments, str):  # some models send them as text
                arguments = json.loads(arguments or "{}")
            print(f"[tool call {step}] {name}({json.dumps(arguments)})")

            result = run_tool(name, arguments)
            preview = result[:80].replace("\n", " ")
            print(f"[result] {len(result)} characters: {preview}...")

            # The result goes back in a message with the role "tool".
            history.append({"role": "tool", "tool_name": name, "content": result})

    print(f"[stopped: the model was still calling tools after {MAX_STEPS} steps]")


def save_log(request_body, chunks):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"agent-{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"request": request_body, "chunks": chunks}, f, indent=2, ensure_ascii=False)


def main():
    print(f"Agent using {MODEL} in {os.getcwd()}. Type /help for commands.")

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
            # A turn can now hold many messages (tool calls and results), so
            # remove everything from your last message onwards.
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
            # Drop the whole failed turn, tool calls included, so the
            # history stays in a shape the model expects.
            del history[start:]
            print(f"\n[turn cancelled: {error or 'interrupted'}]")


if __name__ == "__main__":
    main()

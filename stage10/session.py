"""Stage 9: saving and resuming conversations.

After every turn, the agent writes its state to sessions/<id>.json in the
folder it works in. Because the history is plain JSON in the harness's own
format (stage 5), resuming is just reading it back, even with a different
provider.
"""

import json
import os
import time

SESSIONS_DIR = "sessions"


def new_id():
    return time.strftime("%Y%m%d-%H%M%S")


def save(session_id, agent):
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    state = {
        "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provider": agent.provider.name,
        "model": agent.provider.model,
        "mode": agent.mode,
        "think": agent.think,
        "history": agent.history,
    }
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    # Write to a temporary file first, then swap it in, so a crash halfway
    # through writing can't leave a broken session file behind.
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(path + ".tmp", path)


def list_ids():
    if not os.path.isdir(SESSIONS_DIR):
        return []
    return sorted(name[:-5] for name in os.listdir(SESSIONS_DIR) if name.endswith(".json"))


def load(session_id):
    """Load a session by id; "latest" means the most recent one."""
    ids = list_ids()
    if session_id == "latest":
        if not ids:
            raise FileNotFoundError("there are no saved sessions in this folder")
        session_id = ids[-1]
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    with open(path, encoding="utf-8") as f:
        return session_id, json.load(f)


def describe(session_id):
    """A one-line description: when, which model, and the first message."""
    _, state = load(session_id)
    first = next((m["content"] for m in state["history"] if m["role"] == "user"), "")
    first = first.replace("\n", " ")[:60]
    return f"{session_id}  {state['provider']}/{state['model']}  {len(state['history'])} messages  \"{first}\""

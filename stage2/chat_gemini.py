"""Stage 2: a chat loop with history, a system prompt and streaming.

Usage:
    python stage2/chat_gemini.py

Set GEMINI_API_KEY first, as in stage 1. Set GEMINI_MODEL to try another model.

Commands you can type instead of a message:
    /history   show the conversation exactly as it is sent to the model
    /undo      remove your last message and the model's reply to it
    /quit      leave (Ctrl-C or Ctrl-Z/Ctrl-D also work)
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
LOG_DIR = "logs"

# The system prompt is sent with every request, separately from the
# conversation. Try changing it and see how the replies change.
SYSTEM_PROMPT = "You are a helpful assistant. Keep answers short."


def open_with_retries(request):
    """Open the HTTP request, retrying temporary failures (same idea as stage 1)."""
    attempts = 4
    for attempt in range(1, attempts + 1):
        try:
            return urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as error:
            details = error.read().decode()
            if error.code not in (429, 503) or attempt == attempts:
                raise RuntimeError(f"HTTP {error.code} from the API:\n{details}")
            problem = f"HTTP {error.code}"
        except TimeoutError:
            if attempt == attempts:
                raise RuntimeError("The API did not answer in time. Try again later.")
            problem = "no answer in time"

        wait = 5 * 2 ** (attempt - 1)
        print(f"[{problem}; retrying in {wait}s]", file=sys.stderr)
        time.sleep(wait)


def stream_reply(history):
    """Send the whole conversation, print the reply as it arrives, return it.

    Note that `history` holds every message so far. The model remembers
    nothing between requests: this list is its entire memory.
    """
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": history,
    }
    # streamGenerateContent with alt=sse sends the reply in pieces using
    # "server-sent events": lines of the form  data: {...json...}
    url = f"{API_ROOT}/models/{MODEL}:streamGenerateContent?alt=sse"
    request = urllib.request.Request(
        url,
        method="POST",
        data=json.dumps(body).encode(),
        headers={
            "x-goog-api-key": os.environ["GEMINI_API_KEY"],
            "Content-Type": "application/json",
        },
    )

    chunks = []        # every event, kept for the log
    text = ""          # the reply, assembled from the pieces
    signature = None   # see the note on thoughtSignature below
    usage = {}
    finish_reason = None

    with open_with_retries(request) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue  # blank lines separate events
            chunk = json.loads(line[len("data:"):])
            chunks.append(chunk)

            candidate = chunk.get("candidates", [{}])[0]
            for part in candidate.get("content", {}).get("parts", []):
                if part.get("thought"):
                    continue  # a summary of the model's reasoning, not the answer
                if "text" in part:
                    print(part["text"], end="", flush=True)
                    text += part["text"]
                # Models that "think" before answering return an encrypted
                # record of that thinking. Sending it back with the history
                # lets the model pick up its earlier reasoning.
                if "thoughtSignature" in part:
                    signature = part["thoughtSignature"]

            finish_reason = candidate.get("finishReason", finish_reason)
            usage = chunk.get("usageMetadata", usage)
    print()

    save_log(body, chunks)

    # Show what this turn cost. Watch the input count grow every turn:
    # the whole history is sent again each time.
    print(
        f"[{usage.get('promptTokenCount', '?')} tokens in, "
        f"{usage.get('candidatesTokenCount', '?')} out, "
        f"{usage.get('thoughtsTokenCount', 0)} thinking; "
        f"finish: {finish_reason}]",
        file=sys.stderr,
    )

    reply_part = {"text": text}
    if signature:
        reply_part["thoughtSignature"] = signature
    return {"role": "model", "parts": [reply_part]}


def save_log(request_body, chunks):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"chat-{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"request": request_body, "chunks": chunks}, f, indent=2)


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("Set the GEMINI_API_KEY environment variable first.")
    print(f"Chatting with {MODEL}. Type /help for commands.")

    history = []
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
            # Editing the history is all it takes to make the model
            # "forget" something. Try it, then ask about what you removed.
            history = history[:-2]
            print(f"[removed last exchange; {len(history)} messages left]")
            continue

        history.append({"role": "user", "parts": [{"text": message}]})
        print("model> ", end="", flush=True)
        try:
            history.append(stream_reply(history))
        except (RuntimeError, TimeoutError, KeyboardInterrupt) as error:
            # The turn failed or you pressed Ctrl-C, so drop your message
            # from the history so it is not sent next time without a reply.
            history.pop()
            print(f"\n[turn cancelled: {error or 'interrupted'}]")


if __name__ == "__main__":
    main()

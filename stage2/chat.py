"""Stage 2: a chat loop with history, a system prompt and streaming.

Usage:
    python stage2/chat.py

Ollama must be running, as in stage 1. Set OLLAMA_MODEL to try another model.

Commands you can type instead of a message:
    /history   show the conversation exactly as it is sent to the model
    /undo      remove your last message and the model's reply to it
    /think on  let the model reason before answering (for models that think)
    /think off make it answer straight away
    /quit      leave (Ctrl-C or Ctrl-Z/Ctrl-D also work)

chat_gemini.py is the same program for Google's Gemini API.
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

# The system prompt is sent with every request, before the conversation.
# Try changing it and see how the replies change.
SYSTEM_PROMPT = "You are a helpful assistant. Keep answers short."


def open_request(request):
    """Open the HTTP request, turning failures into readable messages."""
    try:
        # A generous timeout, because loading the model can take a while.
        return urllib.request.urlopen(request, timeout=300)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code} from Ollama: {error.read().decode()}")
    except urllib.error.URLError as error:
        raise RuntimeError(f"could not reach Ollama at {OLLAMA_URL} ({error.reason}). Is it running?")


def stream_reply(history, think):
    """Send the whole conversation, print the reply as it arrives, return it.

    Note that `history` holds every message so far. The model remembers
    nothing between requests: this list is its entire memory.
    """
    # The system prompt goes first, as a message with the role "system".
    # (Gemini takes it in a separate field instead: a provider difference.)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    body = {"model": MODEL, "messages": messages, "stream": True}
    if think is not None:
        body["think"] = think

    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    chunks = []       # every piece, kept for the log
    text = ""         # the reply, assembled from the pieces
    thought = False   # whether any thinking was shown
    final = {}        # the last piece, which carries the statistics

    with open_request(request) as response:
        # Ollama streams one JSON object per line ("newline-delimited JSON").
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            chunk = json.loads(line)
            chunks.append(chunk)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])

            message = chunk.get("message", {})
            # Thinking models first write out their reasoning, then the answer.
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
            if chunk.get("done"):
                final = chunk
    print()

    save_log(body, chunks)

    # Show what this turn cost. Ollama may reuse work from earlier turns (a
    # cache), so "in" may count only the tokens it had to read this time.
    seconds = final.get("eval_duration", 0) / 1e9
    speed = final.get("eval_count", 0) / seconds if seconds else 0
    print(
        f"[{final.get('prompt_eval_count', '?')} tokens in, "
        f"{final.get('eval_count', '?')} out, {speed:.0f} tokens/s; "
        f"finish: {final.get('done_reason')}]",
        file=sys.stderr,
    )

    # Only the answer goes into the history; the thinking is dropped, as most
    # harnesses do. The model will not see its earlier reasoning next turn.
    return {"role": "assistant", "content": text}


def save_log(request_body, chunks):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"chat-{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"request": request_body, "chunks": chunks}, f, indent=2, ensure_ascii=False)


def main():
    print(f"Chatting with {MODEL}. Type /help for commands.")

    history = []
    think = None  # None means "the model's default"
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
        if message in ("/think on", "/think off"):
            think = message == "/think on"
            print(f"[thinking {'on' if think else 'off'}]")
            continue

        history.append({"role": "user", "content": message})
        print("model> ", end="", flush=True)
        try:
            history.append(stream_reply(history, think))
        except (RuntimeError, TimeoutError, KeyboardInterrupt) as error:
            # The turn failed or you pressed Ctrl-C, so drop your message
            # from the history so it is not sent next time without a reply.
            history.pop()
            print(f"\n[turn cancelled: {error or 'interrupted'}]")


if __name__ == "__main__":
    main()

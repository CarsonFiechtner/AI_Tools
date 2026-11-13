"""Stage 1: one raw call to a local model through Ollama, with nothing hidden.

Usage:
    python stage1/one_call.py "Why is the sky blue?"
    python stage1/one_call.py --models   # list the models you have downloaded

Ollama must be running (open the Ollama app, or run `ollama serve`).
Set OLLAMA_MODEL to try a different model.
Every request and response is saved to logs/ so you can read them afterwards.

one_call_gemini.py does the same with Google's Gemini API. Comparing the two
shows what differs between providers and what stays the same.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

# Ollama runs a small web server on your own computer. No API key is needed:
# nothing leaves your machine.
OLLAMA_URL = "http://localhost:11434"
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")
LOG_DIR = "logs"


def call_api(method, path, body=None):
    """Send one HTTP request to Ollama and return the parsed JSON response.

    This is the only place the script touches the network.
    """
    url = OLLAMA_URL + path
    request = urllib.request.Request(
        url,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    try:
        # The first request after Ollama starts can take a while, because the
        # model has to be loaded into memory before it can answer.
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as error:
        # Ollama explains failures (e.g. an unknown model name) in the body.
        sys.exit(f"HTTP {error.code} from Ollama:\n{error.read().decode()}")
    except urllib.error.URLError as error:
        sys.exit(f"Could not reach Ollama at {OLLAMA_URL} ({error.reason}). Is it running?")
    except TimeoutError:
        sys.exit("Ollama did not answer within 5 minutes.")

    save_log(url, body, result)
    return result


def save_log(url, request_body, response_body):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"url": url, "request": request_body, "response": response_body},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[saved {path}]", file=sys.stderr)


def list_models():
    result = call_api("GET", "/api/tags")
    for model in result.get("models", []):
        print(model["name"])


def ask(prompt):
    # The whole "conversation" is the messages list. Each message has a role
    # ("user" or "assistant") and content. Here there is exactly one.
    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": False,  # wait for the whole answer instead of pieces
    }
    result = call_api("POST", "/api/chat", body)

    # Print everything Ollama returned. Things to look for:
    #   message             - the reply, in the same shape as our message
    #   message.thinking    - the model's reasoning before it answered, if any
    #   done_reason         - why the model stopped
    #   prompt_eval_count   - tokens the model read (more than you typed!)
    #   eval_count          - tokens it wrote
    #   *_duration          - timings in nanoseconds (billionths of a second);
    #                         eval_count / eval_duration is its writing speed
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--models":
        list_models()
    else:
        ask(sys.argv[1])

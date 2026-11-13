"""Stage 1: one raw call to the Gemini API, with nothing hidden.

Usage:
    export GEMINI_API_KEY=...            # your key
    python stage1/one_call.py "Why is the sky blue?"
    python stage1/one_call.py --models   # list the model names your key can use

Set GEMINI_MODEL to try a different model.
Every request and response is saved to logs/ so you can read them afterwards.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
LOG_DIR = "logs"


def call_api(method, url, body=None):
    """Send one HTTP request and return the parsed JSON response.

    This is the only place the script touches the network.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Set the GEMINI_API_KEY environment variable first.")

    request = urllib.request.Request(
        url,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as error:
        # The API explains failures in a JSON body; show it rather than hide it.
        sys.exit(f"HTTP {error.code} from the API:\n{error.read().decode()}")

    save_log(url, body, result)
    return result


def save_log(url, request_body, response_body):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(
            {"url": url, "request": request_body, "response": response_body},
            f,
            indent=2,
        )
    print(f"[saved {path}]", file=sys.stderr)


def list_models():
    result = call_api("GET", f"{API_ROOT}/models?pageSize=200")
    for model in result.get("models", []):
        if "generateContent" in model.get("supportedGenerationMethods", []):
            print(model["name"].removeprefix("models/"))


def ask(prompt):
    # The whole "conversation" is this list. Each entry has a role ("user" or
    # "model") and parts. Here there is exactly one user message.
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]},
        ],
    }
    result = call_api("POST", f"{API_ROOT}/models/{MODEL}:generateContent", body)

    # Print everything the API returned. Things to look for:
    #   candidates[0].content       - the reply, in the same shape as our message
    #   candidates[0].finishReason  - why the model stopped
    #   usageMetadata               - tokens in and out (what you pay for)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--models":
        list_models()
    else:
        ask(sys.argv[1])

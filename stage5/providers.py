"""Stage 5: one interface, several model providers.

The rest of the harness only ever sees the internal format below. Each
provider class translates it into one API's format and back again. Adding a
provider means writing one more class; nothing else changes.

Internal format
---------------
User message:
    {"role": "user", "content": "text"}
Assistant message (tool_calls only when the model asked for tools):
    {"role": "assistant", "content": "text",
     "tool_calls": [{"id": "call_1", "name": "read_file", "arguments": {...}}]}
Tool result:
    {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "text"}
Tool definition:
    {"name": "read_file", "description": "...", "parameters": {JSON Schema}}

Messages and tool calls may also carry a "signature": something a provider
(Gemini) needs back on the next request. Other providers ignore it.

Every provider has complete(system, messages, tools, think, on_text,
on_thinking), which streams the reply through the two callbacks and returns
(assistant_message, usage). usage is a dict with input_tokens,
output_tokens, cached_tokens and finish; a value is None if the provider
does not report it.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

LOG_DIR = "logs"


class ProviderError(Exception):
    """Something went wrong talking to a model provider."""


# --- Shared plumbing ---------------------------------------------------------

def post(url, body, headers):
    """POST JSON and return the open response, retrying temporary failures."""
    host = urllib.parse.urlparse(url).netloc
    data = json.dumps(body).encode()
    attempts = 4
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json", **headers},
        )
        try:
            return urllib.request.urlopen(request, timeout=300)
        except urllib.error.HTTPError as error:
            details = error.read().decode(errors="replace")
            if error.code not in (429, 500, 502, 503, 504) or attempt == attempts:
                raise ProviderError(f"HTTP {error.code} from {host}: {details}")
            problem = f"HTTP {error.code}"
        except urllib.error.URLError as error:
            raise ProviderError(f"could not reach {host} ({error.reason}). Is it running?")
        except TimeoutError:
            if attempt == attempts:
                raise ProviderError(f"{host} did not answer in time")
            problem = "no answer in time"
        wait = 5 * 2 ** (attempt - 1)
        print(f"[{problem}; retrying in {wait}s]")
        time.sleep(wait)


def ndjson_lines(response):
    """Ollama's streaming format: one JSON object per line."""
    for raw in response:
        line = raw.decode("utf-8").strip()
        if line:
            yield json.loads(line)


def sse_events(response):
    """The "server-sent events" format used by Gemini and OpenAI:
    lines of the form  data: {...}  separated by blank lines."""
    for raw in response:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":  # OpenAI's end marker
            return
        yield json.loads(data)


def parse_arguments(raw):
    """Tool arguments arrive as a dict or as JSON text, depending on the API.
    If they are broken JSON, keep the text so the tool can report it."""
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {"_unparsed": raw}
    except json.JSONDecodeError:
        return {"_unparsed": raw}


def new_call_id():
    return "call_" + uuid.uuid4().hex[:8]


def save_log(provider, request_body, events):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{provider}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"request": request_body, "events": events}, f, indent=2, ensure_ascii=False)


# --- Ollama (local models) ---------------------------------------------------

class Ollama:
    """Ollama's own API (stages 1-4). Docs: github.com/ollama/ollama, docs/api.md"""

    name = "ollama"

    def __init__(self, model=None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")
        self.url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        self.context_tokens = 8192  # we ask for this much with num_ctx below

    def to_native(self, system, messages):
        # The system prompt is just the first message.
        native = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "assistant":
                out = {"role": "assistant", "content": m["content"]}
                if m.get("tool_calls"):
                    # Ollama has no call ids, and takes arguments as a dict.
                    out["tool_calls"] = [
                        {"function": {"name": c["name"], "arguments": c["arguments"]}}
                        for c in m["tool_calls"]
                    ]
                native.append(out)
            elif m["role"] == "tool":
                # Results are matched to calls by tool name and order.
                native.append({"role": "tool", "tool_name": m["name"], "content": m["content"]})
            else:
                native.append({"role": "user", "content": m["content"]})
        return native

    def complete(self, system, messages, tools, think, on_text, on_thinking):
        body = {
            "model": self.model,
            "messages": self.to_native(system, messages),
            "tools": [{"type": "function", "function": t} for t in tools],
            "stream": True,
            "options": {"num_ctx": self.context_tokens},
        }
        if think is not None:
            body["think"] = think

        text, calls, final, events = "", [], {}, []
        with post(f"{self.url}/api/chat", body, {}) as response:
            for event in ndjson_lines(response):
                events.append(event)
                if "error" in event:
                    raise ProviderError(event["error"])
                message = event.get("message", {})
                if message.get("thinking"):
                    on_thinking(message["thinking"])
                if message.get("content"):
                    on_text(message["content"])
                    text += message["content"]
                calls += message.get("tool_calls", [])
                if event.get("done"):
                    final = event
        save_log(self.name, body, events)

        reply = {"role": "assistant", "content": text}
        if calls:
            reply["tool_calls"] = [
                {"id": new_call_id(), "name": c["function"]["name"],
                 "arguments": parse_arguments(c["function"].get("arguments"))}
                for c in calls
            ]
        usage = {
            "input_tokens": final.get("prompt_eval_count"),
            "output_tokens": final.get("eval_count"),
            "cached_tokens": final.get("prompt_eval_cached_count"),
            "finish": final.get("done_reason"),
        }
        return reply, usage


# --- Gemini --------------------------------------------------------------------

class Gemini:
    """Google's Gemini API. Docs: ai.google.dev/gemini-api/docs/function-calling"""

    name = "gemini"
    root = os.environ.get("GEMINI_URL", "https://generativelanguage.googleapis.com/v1beta")

    def __init__(self, model=None):
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        self.key = os.environ.get("GEMINI_API_KEY")
        if not self.key:
            raise ProviderError("set the GEMINI_API_KEY environment variable first")
        self.context_tokens = 1_000_000

    def to_native(self, messages):
        contents = []
        for m in messages:
            if m["role"] == "assistant":
                # Gemini calls the assistant "model", and a message is a list
                # of "parts": text parts and functionCall parts.
                parts = []
                if m["content"]:
                    part = {"text": m["content"]}
                    if m.get("signature"):
                        part["thoughtSignature"] = m["signature"]
                    parts.append(part)
                for c in m.get("tool_calls", []):
                    part = {"functionCall": {"name": c["name"], "args": c["arguments"]}}
                    # Thinking Gemini models require their signature back on
                    # each call, or the request is refused. For calls another
                    # model made (after /use), Google documents this
                    # placeholder value instead.
                    part["thoughtSignature"] = c.get("signature") or "skip_thought_signature_validator"
                    parts.append(part)
                contents.append({"role": "model", "parts": parts or [{"text": ""}]})
            elif m["role"] == "tool":
                # Tool results are sent by the *user*, as functionResponse
                # parts, and all results for one model turn go together.
                part = {"functionResponse": {"name": m["name"], "response": {"result": m["content"]}}}
                last = contents[-1] if contents else None
                if last and last["role"] == "user" and "functionResponse" in last["parts"][0]:
                    last["parts"].append(part)
                else:
                    contents.append({"role": "user", "parts": [part]})
            else:
                contents.append({"role": "user", "parts": [{"text": m["content"]}]})
        return contents

    def complete(self, system, messages, tools, think, on_text, on_thinking):
        body = {
            # The system prompt has a field of its own.
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": self.to_native(messages),
            "tools": [{"functionDeclarations": tools}],
        }
        if think is True:
            body["generationConfig"] = {"thinkingConfig": {"includeThoughts": True}}
        elif think is False:
            # Gemini 3 models can't switch thinking off entirely, only lower it.
            body["generationConfig"] = {"thinkingConfig": {"thinkingLevel": "low"}}

        url = f"{self.root}/models/{self.model}:streamGenerateContent?alt=sse"
        text, signature, calls, usage_raw, finish, events = "", None, [], {}, None, []
        with post(url, body, {"x-goog-api-key": self.key}) as response:
            for event in sse_events(response):
                events.append(event)
                candidate = (event.get("candidates") or [{}])[0]
                for part in candidate.get("content", {}).get("parts", []):
                    if "functionCall" in part:
                        call = part["functionCall"]
                        calls.append({
                            "id": new_call_id(), "name": call["name"],
                            "arguments": parse_arguments(call.get("args", {})),
                            "signature": part.get("thoughtSignature"),
                        })
                        continue
                    if part.get("thought") and part.get("text"):
                        on_thinking(part["text"])
                    elif part.get("text"):
                        on_text(part["text"])
                        text += part["text"]
                    if part.get("thoughtSignature"):
                        signature = part["thoughtSignature"]
                finish = candidate.get("finishReason", finish)
                usage_raw = event.get("usageMetadata", usage_raw)
        save_log(self.name, body, events)

        reply = {"role": "assistant", "content": text}
        if signature:
            reply["signature"] = signature
        if calls:
            reply["tool_calls"] = calls
        usage = {
            "input_tokens": usage_raw.get("promptTokenCount"),
            "output_tokens": (usage_raw.get("candidatesTokenCount") or 0)
                             + (usage_raw.get("thoughtsTokenCount") or 0) or None,
            "cached_tokens": usage_raw.get("cachedContentTokenCount"),
            "finish": finish,
        }
        return reply, usage


# --- OpenAI-compatible ---------------------------------------------------------

class OpenAICompatible:
    """The OpenAI Chat Completions format, which many others copy: OpenAI,
    OpenRouter, LM Studio, llama.cpp's server, and Ollama itself
    (set OPENAI_BASE_URL=http://localhost:11434/v1 to try it locally)."""

    name = "openai"

    def __init__(self, model=None):
        self.url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.key = os.environ.get("OPENAI_API_KEY", "")  # local servers need none
        self.model = model or os.environ.get("OPENAI_MODEL")
        if not self.model:
            raise ProviderError("name a model with --model or the OPENAI_MODEL environment variable")
        self.context_tokens = int(os.environ.get("OPENAI_CONTEXT_TOKENS", "128000"))

    def to_native(self, system, messages):
        native = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "assistant":
                out = {"role": "assistant", "content": m["content"] or None}
                if m.get("tool_calls"):
                    # Here every call has an id, and arguments are JSON *text*.
                    out["tool_calls"] = [
                        {"id": c["id"], "type": "function",
                         "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                        for c in m["tool_calls"]
                    ]
                native.append(out)
            elif m["role"] == "tool":
                # Results are matched to calls by id.
                native.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            else:
                native.append({"role": "user", "content": m["content"]})
        return native

    def complete(self, system, messages, tools, think, on_text, on_thinking):
        body = {
            "model": self.model,
            "messages": self.to_native(system, messages),
            "tools": [{"type": "function", "function": t} for t in tools],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if think is not None:
            # Not every server understands this; some reject it with an error.
            body["reasoning_effort"] = "high" if think else "low"

        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        text, calls, usage_raw, finish, events = "", {}, {}, None, []
        with post(f"{self.url}/chat/completions", body, headers) as response:
            for event in sse_events(response):
                events.append(event)
                if event.get("usage"):
                    usage_raw = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    # Reasoning is not part of the official format; servers
                    # that send it use one of these two names.
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning:
                        on_thinking(reasoning)
                    if delta.get("content"):
                        on_text(delta["content"])
                        text += delta["content"]
                    # A tool call can arrive in pieces: the name first, then
                    # the arguments a few characters at a time. Glue them
                    # together by their position ("index").
                    for piece in delta.get("tool_calls", []):
                        call = calls.setdefault(piece.get("index", 0), {"id": None, "name": "", "arguments": ""})
                        call["id"] = piece.get("id") or call["id"]
                        function = piece.get("function", {})
                        call["name"] += function.get("name") or ""
                        call["arguments"] += function.get("arguments") or ""
                    finish = choice.get("finish_reason") or finish
        save_log(self.name, body, events)

        reply = {"role": "assistant", "content": text}
        if calls:
            reply["tool_calls"] = [
                {"id": c["id"] or new_call_id(), "name": c["name"],
                 "arguments": parse_arguments(c["arguments"])}
                for _, c in sorted(calls.items())
            ]
        usage = {
            "input_tokens": usage_raw.get("prompt_tokens"),
            "output_tokens": usage_raw.get("completion_tokens"),
            "cached_tokens": (usage_raw.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "finish": finish,
        }
        return reply, usage


PROVIDERS = {"ollama": Ollama, "gemini": Gemini, "openai": OpenAICompatible}


def make_provider(name, model=None):
    if name not in PROVIDERS:
        raise ProviderError(f"unknown provider {name!r}; choose from {', '.join(PROVIDERS)}")
    return PROVIDERS[name](model)

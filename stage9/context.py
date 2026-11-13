"""Stage 7: keeping the conversation inside the context window.

Every request sends the system prompt, the tool definitions and the whole
history. As a task goes on, that grows until it no longer fits. When it
doesn't fit, Ollama silently drops the oldest messages; other providers
refuse the request. Either way, the harness should act first.

Two steps, cheapest first:
  1. Trim: replace old tool results (file contents, command output) with a
     short note. They are usually the biggest part, and the model can run
     the tool again if it needs them.
  2. Summarize: ask the model to summarize the older turns, and replace
     them with that summary. This loses detail, so it is the last resort.

Caching: providers can reuse work for the part of a request that is the same
as last time (the "prefix"). Trimming and summarizing change earlier
messages, so the next request can't reuse that work. Watch the "cached"
number in the usage line before and after a /compact.
"""

import json

from tools import shorten

# Thresholds, as a share of the context window.
TRIM_AT = 0.6
SUMMARIZE_AT = 0.75

KEEP_TOOL_RESULTS = 3  # the most recent tool results are never trimmed
KEEP_TURNS = 2         # the most recent turns are never summarized

SUMMARY_SYSTEM_PROMPT = "You summarize conversations accurately and briefly."
SUMMARY_REQUEST = """Summarize the conversation below between a user and a coding \
assistant, so the assistant can carry on the work without it. Keep: what the user \
wants, decisions made, which files were read or changed and what was learned about \
them, and anything still to do. Leave out everything else.

{transcript}"""


def estimate_tokens(value):
    """A rough count: about 4 characters per token for English and code.
    Only the provider's own count is exact; compare the two with /context."""
    return len(json.dumps(value, ensure_ascii=False)) // 4


def breakdown(system_prompt, tool_definitions, history):
    """Estimated tokens for each part of a request."""
    parts = {"system prompt": estimate_tokens(system_prompt),
             "tool definitions": estimate_tokens(tool_definitions)}
    for role, label in (("user", "user messages"), ("assistant", "assistant messages"),
                        ("tool", "tool results")):
        parts[label] = estimate_tokens([m for m in history if m["role"] == role])
    return parts


def trim_tool_results(history):
    """Step 1: replace all but the newest tool results with a short note.
    Returns how many were trimmed."""
    tool_messages = [m for m in history if m["role"] == "tool"]
    trimmed = 0
    for message in tool_messages[:-KEEP_TOOL_RESULTS]:
        if message.get("trimmed") or len(message["content"]) < 300:
            continue
        message["content"] = (
            f"[Old output removed to save space ({len(message['content'])} characters). "
            "Run the tool again if you need it.]"
        )
        message["trimmed"] = True
        trimmed += 1
    return trimmed


def transcript(messages, limit_chars):
    """The messages as plain text, for the model to summarize."""
    lines = []
    for m in messages:
        if m["role"] == "tool":
            lines.append(f"[{m['name']} result] {shorten(m['content'], 600)}")
            continue
        if m["content"]:
            lines.append(f"{m['role'].upper()}: {m['content']}")
        for call in m.get("tool_calls", []):
            lines.append(f"[assistant called {call['name']}({json.dumps(call['arguments'])[:200]})]")
    return shorten("\n".join(lines), limit_chars)


def summarize(provider, history):
    """Step 2: replace all but the last KEEP_TURNS turns with a summary.
    Returns the number of messages replaced (0 if there was nothing to do)."""
    user_turns = [i for i, m in enumerate(history) if m["role"] == "user"]
    if len(user_turns) <= KEEP_TURNS:
        return 0
    split = user_turns[-KEEP_TURNS]  # cut only where a turn starts
    old, recent = history[:split], history[split:]

    # The summary request itself has to fit in the window, too.
    request = SUMMARY_REQUEST.format(transcript=transcript(old, provider.context_tokens * 2))
    reply, _ = provider.complete(
        SUMMARY_SYSTEM_PROMPT, [{"role": "user", "content": request}], [], None,
        on_text=lambda text: None, on_thinking=lambda text: None,
    )

    # The summary goes into the first kept user message. Like the fake
    # history in stage 2, the model will treat it as part of the conversation.
    first = recent[0]
    recent[0] = dict(first, content=(
        f"[Summary of our earlier conversation, made to save space]\n{reply['content'].strip()}"
        f"\n\n[My next message]\n{first['content']}"
    ))
    history[:] = recent
    return len(old)

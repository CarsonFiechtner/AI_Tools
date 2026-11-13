"""Stage 5: the agent, now able to use several model providers.

Usage:
    python stage5/agent.py                                # Ollama, qwen3.5:9b
    python stage5/agent.py --provider gemini              # needs GEMINI_API_KEY
    python stage5/agent.py --provider openai --model NAME # needs OPENAI_API_KEY

The "openai" provider works with anything that copies OpenAI's format. To
try it against your local Ollama (same model, different API):
    PowerShell:  $env:OPENAI_BASE_URL = "http://localhost:11434/v1"
    then:        python stage5/agent.py --provider openai --model qwen3.5:9b

Commands you can type instead of a message:
    /use PROVIDER [MODEL]  switch provider mid-conversation, keeping the history
    /history   show the conversation in the harness's own format
    /undo      remove your last message and everything after it
    /think on  let the model reason before answering
    /think off make it answer straight away
    /quit      leave (Ctrl-C or Ctrl-Z/Ctrl-D also work)

New in stage 5: the code is split into tools.py, providers.py and this file,
and the agent loop no longer knows which provider it is talking to.
"""

import argparse
import json
import os
import platform
import sys
import time

from providers import ProviderError, make_provider
from tools import IS_WINDOWS, TOOLS, run_tool

MAX_STEPS = 20

SYSTEM_PROMPT = f"""You are a coding assistant working in the folder {os.getcwd()}
on {platform.system()}. Commands run in {"PowerShell" if IS_WINDOWS else "sh"}.

Use the tools to look at files before answering questions about them.
Read a file before changing it. Prefer small, exact edits.
When a task is done, say briefly what you did. Keep answers short."""


class Printer:
    """Shows a streaming reply, keeping thinking and answer apart."""

    def __init__(self):
        self.thinking = False
        self.answering = False

    def thought(self, text):
        if not self.thinking:
            print("[thinking] ", end="")
            self.thinking = True
        print(text, end="", flush=True)

    def answer(self, text):
        if self.thinking and not self.answering:
            print("\n[answer] ", end="")
        self.answering = True
        print(text, end="", flush=True)


class Agent:
    def __init__(self, provider):
        self.provider = provider
        self.history = []   # in the internal format described in providers.py
        self.think = None

    def call_model(self):
        """One request to the model; prints the reply as it streams."""
        printer = Printer()
        print("model> ", end="", flush=True)
        started = time.time()
        # Only the description part of each tool is sent, never the function.
        definitions = [
            {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
            for t in TOOLS
        ]
        reply, usage = self.provider.complete(
            SYSTEM_PROMPT, self.history, definitions, self.think,
            on_text=printer.answer, on_thinking=printer.thought,
        )
        print()
        cached = f" ({usage['cached_tokens']} cached)" if usage["cached_tokens"] else ""
        print(
            f"[{self.provider.name}: {usage['input_tokens']} tokens in{cached}, "
            f"{usage['output_tokens']} out, {time.time() - started:.1f}s; finish: {usage['finish']}]",
            file=sys.stderr,
        )
        return reply

    def run_turn(self, user_text):
        """The agent loop from stage 3, unchanged except for the message format."""
        start = len(self.history)
        self.history.append({"role": "user", "content": user_text})
        try:
            for step in range(1, MAX_STEPS + 1):
                reply = self.call_model()
                self.history.append(reply)
                if not reply.get("tool_calls"):
                    return
                for call in reply["tool_calls"]:
                    print(f"[tool call {step}] {call['name']}({json.dumps(call['arguments'])[:200]})")
                    result = run_tool(call["name"], call["arguments"])
                    print(f"[result] {len(result)} characters: {result[:80].replace(chr(10), ' ')}...")
                    self.history.append({
                        "role": "tool", "tool_call_id": call["id"],
                        "name": call["name"], "content": result,
                    })
            print(f"[stopped: the model was still calling tools after {MAX_STEPS} steps]")
        except BaseException:
            # A failed or interrupted turn is removed whole, as before.
            del self.history[start:]
            raise


def main():
    parser = argparse.ArgumentParser(description="A small coding agent.")
    parser.add_argument("--provider", default="ollama", help="ollama, gemini or openai")
    parser.add_argument("--model", help="model name (each provider has a default except openai)")
    args = parser.parse_args()

    try:
        agent = Agent(make_provider(args.provider, args.model))
    except ProviderError as error:
        sys.exit(f"Error: {error}")
    print(f"Agent using {agent.provider.name} / {agent.provider.model} in {os.getcwd()}. Type /help for commands.")

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
            print(json.dumps(agent.history, indent=2, ensure_ascii=False))
            continue
        if message == "/undo":
            user_turns = [i for i, m in enumerate(agent.history) if m["role"] == "user"]
            if user_turns:
                del agent.history[user_turns[-1]:]
            print(f"[removed last turn; {len(agent.history)} messages left]")
            continue
        if message in ("/think on", "/think off"):
            agent.think = message == "/think on"
            print(f"[thinking {'on' if agent.think else 'off'}]")
            continue
        if message.startswith("/use "):
            # The history is in the harness's own format, so any provider
            # can pick up the conversation where another left off.
            parts = message.split()
            try:
                agent.provider = make_provider(parts[1], parts[2] if len(parts) > 2 else None)
                print(f"[now using {agent.provider.name} / {agent.provider.model}]")
            except (ProviderError, IndexError) as error:
                print(f"[could not switch: {error}]")
            continue

        try:
            agent.run_turn(message)
        except (ProviderError, TimeoutError, KeyboardInterrupt) as error:
            print(f"\n[turn cancelled: {error or 'interrupted'}]")


if __name__ == "__main__":
    main()

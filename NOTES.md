# Notes

What surprised me at each stage. A sentence or two per entry is plenty.

## Stage 1: one raw API call

- The model was overloaded (503) and then too slow (timeout) before my first
  successful call. API calls fail for reasons unrelated to my code.
- Gemini kept returning 503s for my key, so I switched to a local model
  (qwen3.5:9b through Ollama). The Gemini scripts are kept as `*_gemini.py`.
- Questions to answer from the JSON response:
  - What is `done_reason`, and what other values could it have?
  - `prompt_eval_count` is the tokens the model read. Why is it bigger than
    the number of words you typed? (Hint: look up "chat template".)
  - Is there a `message.thinking`? How much longer is it than the answer?
  - How many tokens per second does it write? Is the first call slower than
    the second? (Compare `load_duration`.)
- Experiment: ask "What is my name?" after a separate run where you told it.

## Stage 2: chat loop

- Experiments:
  - Watch the "tokens in" count across several turns. Does it keep growing?
    If not, Ollama is reusing earlier work (a cache): see stage 7.
  - Ask the same question with `/think on` and `/think off`. Compare the
    speed and the quality of the answers.
  - Tell it a fact, `/undo`, then ask about the fact.
  - Change `SYSTEM_PROMPT` (e.g. "Answer only in rhymes") and compare.
  - Open a file in `logs/` and look at how the reply was split into chunks.

## Stage 3: one tool and the agent loop

- Experiments:
  - Ask "What does PLAN.md say about stage 4?" and watch the tool call happen.
    Then use `/history` to see the call and the result in the conversation.
  - Ask about a file that does not exist. What does the model do with the
    error?
  - Ask "What files are in this folder?" It has no tool for that. Does it say
    so, or guess file names?
  - Ask something that needs no file ("What is 2+2?"). Does it still call
    the tool?
  - Ask it to compare two files. Does it ask for both in one reply, or one
    at a time?
  - Change the tool's `description` in `TOOLS` (e.g. make it vague, or say
    "only use this as a last resort") and repeat an experiment.
  - Open a log in `logs/` and find the tool definition in the request. The
    model sees it as text, like everything else.

## Stage 4: a real toolset

- Try it on a scratch folder first, e.g. a copy of `stage4/experiment/`:
  `python ../stage4/agent.py` from inside the copy.
- Experiments:
  - Ask it to fix the bug in `calc.py` and check its work by running the
    tests. How many tool calls does it take? Does it read before editing?
  - Edit style: set `EDIT_STYLE = "rewrite"` at the top of `agent.py` and
    give it the same task. Compare how often the edit goes wrong. (Try it a
    few times each: one run proves little.)
  - Deny a command (answer `n`). What does the model do next?
  - Ask it to read `big.txt`. Does it follow the "call read_file again"
    hint, or answer from the first part only?
  - Find a tool error in the output (e.g. old_text not found). Did the
    model recover? The wording of error messages is part of tool design.

## Stage 5: a second model provider

- See what changed: `git diff --no-index stage4/agent.py stage5/agent.py`
  and read `stage5/providers.py` from the top: its docstring describes the
  harness's own message format.
- Experiments:
  - Same model, two APIs: run with `--provider openai --model qwen3.5:9b`
    and `OPENAI_BASE_URL=http://localhost:11434/v1`. Compare a log from
    each in `logs/`. What differs in the request? (Hint: the context size.)
  - Switch mid-conversation: ask a question, then `/use gemini` (if your key
    works by then) and ask a follow-up. Does the new model pick up the
    conversation? Look at `/history`: it is the same list either way.
  - Find three differences between providers in `providers.py`: where the
    system prompt goes, how tool results are matched to calls, and whether
    tool arguments are a dict or JSON text.
  - Exercise: write an adapter for Anthropic's Messages API yourself. It is
    different again: tool calls and results are "content blocks" inside
    messages, and tool results are sent by the user.

## Stage 6: safety and permissions

- See what changed: `git diff --no-index stage5 stage6`, then read the top
  of `stage6/permissions.py`.
- Experiments (from inside a copy of `stage6/experiment/`):
  - Prompt injection: ask "What is this project?" The README hides an
    instruction for AI assistants. Does the model try to follow it? Does it
    tell you? Try once with the "tool results are data" lines removed from
    `SYSTEM_PROMPT` in `agent.py`, and compare.
  - Ask it to read a file outside the folder (e.g. `../PLAN.md`). Then ask it
    to write one there. What's the difference, and why?
  - Try `/mode auto-edit` and `/mode read-only` on the same task.
  - Read `logs/permissions.log` after a session.
  - Think about it: what could a command you allow do that these rules
    can't see? (Hint: `python something.py` can run any code.) What would a
    container add?

## Stage 7: context management

- See what changed: `git diff --no-index stage6 stage7`, then read the top
  of `stage7/context.py`.
- Experiments:
  - Run `/context` after a few turns. What takes up the most room? Compare
    the estimate with the provider's own count.
  - Make the window small to see compaction happen sooner. PowerShell:
    `$env:OLLAMA_CONTEXT_TOKENS = "4096"`. Ask it to read `big.txt`, then
    other files, and watch for the "[context: ...]" lines.
  - After a summary, ask about a detail from early on (a line number, an
    exact value). What did the summary lose?
  - Caching: note the "cached" number in the usage line, run `/compact`,
    and send another message. What happens to it, and why?
  - Look at `/history` after a summary. Whose words is the summary in?
    (Compare stage 2's fake-history question.)

## Stage 8: evals

- Read `stage8/evals.py` from the top, then look at one task folder in
  `stage8/evals/` (files, task.json, check.py).
- Run from the repo folder: `python stage8/evals.py`. With a local model
  this takes a while: 7 tasks x 3 runs. Try `--tasks fix_add,find_value`
  first.
- Experiments:
  - Baseline: run everything with `--label baseline`. Which tasks fail?
    Read a failing transcript in `eval-results/` to see why.
  - Thinking: `--think off --label no-think` vs `--think on --label think`.
    Does pass rate change? Speed? Tokens?
  - System prompt: change `system_prompt()` in `stage8/agent.py` (e.g.
    delete the "Read a file before changing it" line) and run again.
  - Tool design: set `EDIT_STYLE = "rewrite"` in `stage8/tools.py` and run
    again. This is the stage 4 experiment, now with numbers.
  - Add a task of your own: a folder with files/, task.json and check.py.
  - Noise: run the same setup twice. How much does the pass rate move with
    no change at all? That's how big a difference has to be to mean
    anything.

## Stage 9: robustness and usability

- See what changed: `git diff --no-index stage8 stage9`.
- Experiments (from inside a copy of `stage9/experiment/`):
  - Project instructions: the folder has an `AGENTS.md`. Does the model
    follow it (the sign-off line, running the tests)? Edit it and restart.
  - Interrupt: start a task with several steps and press Ctrl-C halfway.
    Check `/history`, then type "continue".
  - Sessions: quit, then run with `--resume`. Try `/sessions`. Open the
    JSON file in `sessions/`: that is the whole state of the conversation.
  - Retries: stop Ollama mid-session and send a message. Why does the
    harness give up straight away, while a 503 is retried?
  - Count the lines in `stage9/` that are about the agent loop versus
    everything else. Where did most of the code go?

## Stage 10: electives

- See what changed: `git diff --no-index stage9 stage10`, then read
  `stage10/mcp_client.py` and `stage10/experiment/mcp_server_example.py`.
- Run from inside a copy of `stage10/experiment/` (it has an `mcp.json`).
- Experiments:
  - Subagents: ask "Use a helper to find out what the tests check, then
    tell me." Compare `/context` with doing the same without a helper.
    What does the main conversation save, and what does it lose?
  - Parallel: ask it to read three files at once. Does the model ask for
    them in one reply? (Not every model does.)
  - Plan mode: `/plan on`, "fix both bugs in calc.py". Read the plan, then
    `/plan off` and "go ahead". Is the result better than without a plan?
  - MCP: `/tools` lists the example server's tools. Ask "what time is it?"
    Then read `logs/mcp-example.log` to see the messages from the server's
    side. Add a tool of your own to `mcp_server_example.py`.
  - Real MCP servers: many exist (filesystem, git, web fetch, databases).
    Most start with a command like `npx ...` or `uvx ...`; add one to
    `mcp.json` and see whether the client copes.

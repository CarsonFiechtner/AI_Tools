# Building my own AI harness: a learning plan

The goal is understanding, not a competitor to Claude Code or Codex. Each stage
adds one concept and ends with an experiment.

## Choices

- **Language: Python.** Standard library only to begin with, so every HTTP
  request is visible.
- **Provider: Gemini first**, over its raw REST API. A second provider is added
  in stage 5 behind a small interface.
- **No agent frameworks.** The framework is the thing being learned.

## Ground rules

- Log every raw request and response to `logs/`. Reading transcripts is where
  most of the learning happens.
- Keep `NOTES.md`: for each stage, write down what surprised you.
- One directory per stage (`stage1/`, `stage2/`, ...), each copied from the
  last, so earlier versions stay readable.

## Stages

### 1. One raw API call
- **Build:** a script that sends one message and prints the full JSON response.
- **Learn:** the message format, roles, token usage and finish reasons, and
  that the model is stateless.
- **Experiment:** ask a follow-up without resending history and see what
  happens.

### 2. Chat loop
- **Build:** a REPL that keeps conversation history, has a system prompt and
  streams output.
- **Learn:** "memory" is just resending the history; how streaming events are
  structured.
- **Experiment:** edit the history between turns (delete or rewrite a message)
  and see how the model reacts.

### 3. One tool and the agent loop
- **Build:** a single tool (`read_file`) and the loop: call the model, run any
  tool it requests, append the result, repeat until it stops asking.
- **Learn:** this loop is the core of every harness. The model never runs
  anything; it emits a request and your code decides what to do.
- **Experiment:** change the tool description and see how its use of the tool
  changes.

### 4. A real toolset
- **Build:** `list_files`, `search`, `write_file`/`edit_file` and `bash`.
- **Learn:** tool design is interface design for a model. Errors go back as
  tool results so the model can recover; large outputs need truncating.
- **Experiment:** implement file editing two ways (whole-file rewrite vs.
  string replace) and compare failure rates.

### 5. A second model provider
- **Build:** an internal message and tool-call format of your own, with one
  adapter per provider that translates to and from it. Add one more provider.
- **Learn:** what is common to all providers (messages, tool schemas, tool
  calls and results) and what differs (field names, system prompt placement,
  streaming events, how tool results are attached).
- **Experiment:** run the same task on both models and compare the transcripts.

### 6. Safety and permissions
- **Build:** a confirmation prompt before writes and commands, an allowlist,
  and ideally a container for `bash`.
- **Learn:** the harness, not the model, is the trust boundary; what prompt
  injection through file contents looks like.
- **Experiment:** plant an instruction in a file the agent reads and see
  whether it gets followed.

### 7. Context management
- **Build:** token tracking, truncation of old tool output, summarization when
  the history gets long, and prompt caching.
- **Learn:** the context window is the scarcest resource; compaction trades
  away details the model may need later.
- **Experiment:** run a long task with and without compaction and note what
  gets forgotten.

### 8. Evals
- **Build:** 5-10 small repeatable tasks (fix a bug, add a function, answer a
  question about the code) with automatic pass/fail checks.
- **Learn:** without measurement you cannot tell whether a change helped. Runs
  are nondeterministic, so each task needs several runs.
- **Experiment:** rewrite the system prompt and measure the difference; compare
  providers on the same suite.

### 9. Robustness and usability
- **Build:** retries with backoff, Ctrl-C interrupt mid-turn, session save and
  resume, and a project instructions file loaded at startup.
- **Learn:** most of the code in a production harness handles this, not the
  loop.

### 10. Electives
- Subagents with their own context
- Parallel tool calls
- MCP client support
- A plan-then-execute mode
- A terminal UI

## Pacing

Stages 1-3 are an afternoon or two and give the central insight. Stages 4-8
are where most of the understanding comes from. A good milestone after stage 4
is using the harness to work on itself.

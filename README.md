# A small AI coding harness, built in stages

A learning project: an agent harness like Claude Code or Codex, built up one
idea at a time in plain Python (standard library only). See `PLAN.md` for
the plan and `NOTES.md` for experiments and questions for each stage.

## The stages

Each folder is a complete, working program, copied from the one before.
New code is marked `NEW` in comments.

| Stage | Folder | What it adds | Run (from this folder) |
|---|---|---|---|
| 1 | `stage1/` | One raw API call | `python stage1/one_call.py "Why is the sky blue?"` |
| 2 | `stage2/` | Chat loop: history, system prompt, streaming | `python stage2/chat.py` |
| 3 | `stage3/` | One tool and the agent loop | `python stage3/agent.py` |
| 4 | `stage4/` | Real tools: list, search, read, write, edit, run commands | see below |
| 5 | `stage5/` | Several model providers behind one interface | see below |
| 6 | `stage6/` | Permissions and safety | see below |
| 7 | `stage7/` | Context management: trimming and summarizing | see below |
| 8 | `stage8/` | Evals: measuring the agent | `python stage8/evals.py` |
| 9 | `stage9/` | Robustness: retries, interrupts, sessions, AGENTS.md | see below |
| 10 | `stage10/` | Subagents, parallel tools, plan mode, MCP | see below |
| 11 | `stage11/` | A grader model for evals (Gemini Decisions API) | `python stage11/evals.py --grade` |

From stage 4 on, the agent can change files. Run it on a copy of the
stage's `experiment` folder rather than on this repository. In PowerShell:

```
Copy-Item -Recurse stage4\experiment $HOME\agent-test4
cd $HOME\agent-test4
python C:\path\to\AI_Tools\stage4\agent.py
```

To see exactly what a stage added:

```
git diff --no-index stage4/agent.py stage5/agent.py
git diff --no-index stage9 stage10
```

## Models

Stages 1-4 use a local model through [Ollama](https://ollama.com)
(`qwen3.5:9b` by default; set `OLLAMA_MODEL` to change it). From stage 5:

```
python stage5/agent.py                                  # Ollama
python stage5/agent.py --provider gemini                # needs GEMINI_API_KEY
python stage5/agent.py --provider openai --model NAME   # any OpenAI-compatible API
```

## How this was tested

Each stage was tested against fake model servers that replay scripted
replies in the Ollama, Gemini and OpenAI formats, including errors,
retries, tool calls and interrupted turns. It has not yet been run
against a real model beyond stages 1-2, so expect some surprises: they are
part of the learning.

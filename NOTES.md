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

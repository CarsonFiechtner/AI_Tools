# Notes

What surprised me at each stage. A sentence or two per entry is plenty.

## Stage 1: one raw API call

- The model was overloaded (503) and then too slow (timeout) before my first
  successful call. API calls fail for reasons unrelated to my code.
- Questions to answer from the JSON response:
  - What is `finishReason`, and what other values could it have?
  - How many tokens did the question and the answer use?
  - Is there a `thoughtsTokenCount`? The model "thinks" before answering, and
    you pay for those tokens too.
- Experiment: ask "What is my name?" after a separate run where you told it.

## Stage 2: chat loop

- Experiments:
  - Watch the "tokens in" count across several turns. Why does it keep growing?
  - Tell it a fact, `/undo`, then ask about the fact.
  - Change `SYSTEM_PROMPT` (e.g. "Answer only in rhymes") and compare.
  - Open a file in `logs/` and look at how the reply was split into chunks.

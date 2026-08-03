# seeky

A Python chatbot that texts like you, built from your own chat history.

## Use it

1. Put your chat history next to `my_clone.py` as **`chat.json`**
2. Install and set your key:

```bash
pip install anthropic
```
```bat
set ANTHROPIC_API_KEY=your-api-key-here
```
3. Run it:

```bash
python my_clone.py
```

It reads the file, shows you the names it found, asks which one is you, and
starts chatting.

```
Found these people in the chat:
  1. Kay (388 messages)
  2. Priya (246 messages)
Which one is you? [1-2]: 1
What should the clone be called? [me]: Kay

you> hey babe
kay> miss u too
     wyd rn
```

Commands: `/reset`, `/style`, `/quit`.

Different filename? `python my_clone.py whatever.json`

## How it works

Claude has no self-serve fine-tuning, so this uses few-shot prompting: your real
messages go into the system prompt and Claude imitates the voice it reads there.

Two things go into that prompt. First, **measured facts** about how you text —
median message length, how often you start lowercase, how often you skip end
punctuation, your emoji rate, your actual slang with counts, how many messages
you fire off in a row. Numbers steer the model much harder than "text casually".
Second, **~80 real exchanges**, each starting with them and ending with you, so
every example demonstrates the thing being copied. They're sampled evenly across
your whole history so the mix of long replies and one-word replies stays true.

The reader accepts both plain JSON and **JSON Lines** (one object per line), and
doesn't assume one exact schema — it probes the common key names for body,
sender, and timestamp, and handles a plain list, a `{"messages": [...]}` wrapper,
or a dict keyed by name. Call logs, deleted messages, and attachment-only rows
are dropped; call entries in particular carry body text that reads like a real
message (`"Outgoing voice call (unanswered)"`), so they're filtered on their
flags rather than their wording.

If one sender is labelled `Me` it's assumed to be you, so there's nothing to
pick. Otherwise it lists everyone and asks.

## Feeding it the whole history

By default the prompt carries ~80 sampled exchanges — enough to nail your voice,
but the clone doesn't know what actually happened between you. To put **every**
message in instead, set `FULL_HISTORY = True` near the top of `my_clone.py`.

It's a real tradeoff. Measured on an 82,503-message export:

| | tokens | first reply | each later reply |
|---|---|---|---|
| 80 examples (default) | 14,475 | $0.09 | **$0.007** |
| 500 examples | 77,418 | $0.48 | $0.039 |
| 2,000 examples | 311,482 | $1.95 | $0.156 |
| `FULL_HISTORY` | 951,462 | $5.95 | $0.48 |

Two things to know before flipping it:

- **It only just fits.** 951K of a 1M-token window leaves ~48K for everything
  else. A bigger export won't fit at all, so the script counts tokens at startup
  and refuses rather than failing mid-chat.
- **The cache expires after 5 minutes idle**, and texting has gaps — so the
  expensive first write can recur all session. `FULL_HISTORY` switches the cache
  to a 1-hour TTL to blunt that.

Raising `EXAMPLES` is usually the better move: 500 examples is 7% of your history
for under 4c a reply.

## Tuning, measured

Defaults are `claude-haiku-4-5`, `EXAMPLES = 500`, `HISTORY_TURNS = 100`.

Style fidelity was benchmarked over 36 prompts against a 33,285-message profile.
Error is the summed distance from the real person on words per message, emoji
rate, and messages per reply — lower is better.

| config | words | emoji | bubbles | error |
|---|---|---|---|---|
| *the real person* | 4.2 | 10% | 1.88 | — |
| 80 examples | 3.5 | 6% | 1.33 | 49 |
| 500 examples | 3.5 | 5% | 1.19 | 59 |
| 80 examples + burst/emoji rules | 3.3 | 1% | 1.94 | 34 |
| **500 examples + rules (default)** | 3.3 | 6% | **1.89** | **26** |

Two things worth knowing before you tune:

- **Raising `EXAMPLES` alone does nothing.** 500 measured slightly *worse* than
  80 until the prompt also carried explicit burst and emoji rules. The voice
  signal saturates early; what was actually missing was instruction, not data.
- **Models under-burst.** Every model tested sent ~1.3 messages per reply
  against a real 1.88, and stayed at 100% lowercase where the real person is at
  93%. They follow a style card more consistently than people follow their own
  habits.

## Which model

Default is `claude-sonnet-5` (line 36). Copying short casual texts turns out not
to reward a bigger model. Measured on 12 prompts against a 33,285-message
profile:

| | words/msg | no punctuation | lowercase | bubbles |
|---|---|---|---|---|
| **the real person** | 4.2 | 97% | 93% | 1.88 |
| `claude-sonnet-5` | 2.5 | 100% | 100% | 1.83 |
| `claude-haiku-4-5` | **3.3** | 100% | 100% | 1.58 |
| `claude-opus-5` | 2.3 | 100% | 100% | 1.83 |

Haiku came closest on message length; Opus matched best on burst rate. The gaps
are small, so pick on price and context:

- **`claude-sonnet-5`** — 1M context, $3/$15 per Mtok. `FULL_HISTORY` works.
- **`claude-haiku-4-5`** — 200K context, $1/$5. Cheapest, but `FULL_HISTORY`
  won't fit, and it rejects the `effort` parameter (the script omits it
  automatically for haiku).
- **`claude-opus-5`** — 1M context, $5/$25. No measurable gain here.

All three overshoot on consistency — 100% lowercase where the real person is at
93%. Models keep a rule better than people do.

## Notes

- **Your messages get sent to the API** as part of the prompt — including your
  girlfriend's side of the conversation. `.gitignore` excludes `*.json` so the
  export doesn't get committed by accident.
- The prompt is cached, so every message after the first costs a fraction of the
  input price rather than reprocessing the whole history.
- Thinking is off and effort is `low` — texting wants fast and casual, not
  deliberated. Knobs are at the top of the file if you want to change that.
- The prompt tells the model to be honest if someone sincerely asks whether
  they're talking to a person or an AI.

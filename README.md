# seeky

A chatbot that texts the way you do, built from a Signal export and the Claude API.

Anthropic doesn't offer self-serve fine-tuning, so this uses **few-shot prompting**
instead: real exchanges from your history go into the system prompt, and Claude
imitates the voice it reads there. Claude's context window is large enough to hold
a big, representative sample of how you actually write.

Two pieces:

| File | Job |
| --- | --- |
| `signal_parser.py` | Reads the export, measures your style, picks example exchanges. No API calls. |
| `my_clone.py` | Builds the system prompt and runs the chat loop. |

## Setup

```bash
pip install -r requirements.txt
```

Set your API key:

```bat
:: Windows (Command Prompt) — this session only
set ANTHROPIC_API_KEY=your-api-key-here

:: Windows — persist it for future sessions
setx ANTHROPIC_API_KEY "your-api-key-here"
```

```bash
# macOS / Linux
export ANTHROPIC_API_KEY=your-api-key-here
```

## Getting the export

Use [`sigexport`](https://github.com/carderne/signal-export) with JSON output:

```bash
pip install signal-export
sigexport --json C:\Temp\MySignalExport
```

## Run it

Check what the parser found first — this costs nothing and catches the common
mistakes before you spend any tokens:

```bash
python signal_parser.py --export C:\Temp\MySignalExport --preview
```

You'll get a list of conversations, your measured style profile, and a few
sample exchanges. Then start the chat:

```bash
python my_clone.py --export C:\Temp\MySignalExport --name Kay
```

```
you> hey you around?
kay> yeah whats up
     just got back

you> wanna get food
kay> lowkey starving
     where u thinking
```

Commands: `/reset`, `/style`, `/prompt`, `/quit`.

### "no messages from 'Me'"

The export has to be able to tell your messages from theirs. `signal_parser.py`
uses explicit direction fields (`type: outgoing`, `isFromMe`) when present, and
falls back to matching the sender name. If your export only has names and yours
isn't `Me`, pass it:

```bash
python my_clone.py --export ... --me "Your Display Name"
```

Both scripts print the sender names they found when they can't match, so you can
copy the right one.

## Options

| Flag | Default | Notes |
| --- | --- | --- |
| `--name` | `You` | How the clone is labelled in the prompt and the chat |
| `--me` | `Me` | How *you* appear in the export's sender field |
| `--contact` | all | Learn from one conversation only — a good way to capture how you talk to one specific person |
| `--max-examples` | `80` | Exchanges in the prompt. More voice, more tokens |
| `--effort` | `low` | `low`/`medium`/`high` |
| `--thinking` | off | Let Claude reason before replying |
| `--tokens` | off | Report the system prompt's token count |
| `--show-prompt` | off | Print the assembled prompt and exit |

## How it's tuned

A few choices that matter, and why:

**Measured style, not vibes.** The prompt leads with numbers pulled from your
history — median message length, how often you start lowercase, how often you
skip end punctuation, emoji rate, your actual slang with counts, and how many
messages you fire off in a row. Concrete measurements steer the model far more
reliably than "text casually".

**Examples end on your message.** Every example block starts with the other
person and ends with you, so each one demonstrates the thing being imitated.
Blocks never overlap, so 80 examples means 80 distinct conversations rather than
one conversation shown 80 times.

**Even sampling.** Examples are drawn evenly across your whole history instead of
taking the first N. That preserves the real distribution of reply lengths —
including the one-word ones, which for most people are a large share of how they
actually text.

**Thinking off by default.** Claude Opus 5 thinks by default. For imitating
texting that mostly buys latency and over-considered replies, so it runs with
thinking disabled at `low` effort — which is what `low` is for. `--thinking`
turns it back on. (Disabled thinking can very occasionally leak internal tags
into output; the prompt asks it not to and `my_clone.py` strips them as a
backstop.)

**Prompt caching.** The style card and all the examples sit in one cached system
block, so every turn after the first reads them at a fraction of the input price
instead of reprocessing the whole transcript. Keep the prompt stable to keep the
cache warm — changing `--max-examples` or `--name` starts a new cache entry.

**Refusal handling.** Opus 5's safety classifiers occasionally decline a request.
The code opts into server-side fallbacks so a declined request is re-run on a
fallback model rather than returning nothing, and it degrades gracefully if your
account doesn't have that beta. To drop it, remove the `betas`/`extra_body`
arguments in `Clone._stream`.

## Privacy

Your messages are sent to the Anthropic API as part of the prompt. They are also
other people's messages — the exchanges include whatever the other person wrote.
Use `--contact` to narrow the scope, and check `--show-prompt` to see exactly
what gets sent before you send it.

`.gitignore` excludes `*.json` and the usual export directory names so an export
doesn't get committed by accident.

## Honesty

The system prompt tells the model to say so if someone sincerely asks whether
they're talking to a person or an AI. Copying your writing style is the point;
convincing someone they're talking to you when they aren't is a different thing,
and this isn't built for it.

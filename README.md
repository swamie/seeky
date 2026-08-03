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

The JSON reader doesn't assume one exact schema — it probes the common key names
for the message body, sender, and timestamp, and handles a plain list, a
`{"messages": [...]}` wrapper, or a dict keyed by name. If your export uses
something unusual it'll say what it found instead of guessing.

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

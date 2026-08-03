"""A chatbot that texts like you, built from your own chat history.

Put your chat export next to this file as chat.json, then:

    python my_clone.py

That's it. The first run shows you the names it found and asks which one is you.

Claude has no self-serve fine-tuning, so this works by few-shot prompting: your
real messages go into the system prompt and Claude imitates the voice it reads.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

# Paste your API key between the quotes. Leave it empty to use a key.txt file
# next to this script, or the ANTHROPIC_API_KEY environment variable.
# This repo is public — if you paste a key here, don't commit the change.
API_KEY = ""

CHAT_FILE = "chat.json"      # or: python my_clone.py some-other-file.json
# Imitating short casual texts doesn't reward a bigger model — measured against
# 33k real messages, sonnet and haiku matched opus on style and haiku was
# actually closest on message length. Sonnet is the value pick: same 1M context
# as opus (so FULL_HISTORY still works) at a lower price.
#   claude-sonnet-5   1M context, $3/$15 per Mtok
#   claude-haiku-4-5  200K context, $1/$5  — cheapest, but FULL_HISTORY won't fit
#   claude-opus-5     1M context, $5/$25
MODEL = "claude-haiku-4-5"

# Put EVERY message in the prompt instead of a sample. The clone then knows what
# actually happened between you, not just how you write — but it costs roughly
# 66x more per reply, and the whole history has to fit in the model's 1M-token
# context window. The script measures yours at startup and refuses if it won't
# fit. Cache TTL goes to 1h when this is on, so gaps between messages don't keep
# re-billing the expensive first write.
FULL_HISTORY = False

EXAMPLES = 500               # how many real exchanges go in the prompt (ignored if FULL_HISTORY)
MAX_TOKENS = 400             # texts are short; keeps replies from turning into essays
HISTORY_TURNS = 100          # how much of the live chat to remember

# --------------------------------------------------------------------------
# Reading the JSON. Exports vary a lot, so we probe several likely key names
# rather than assuming one exact schema.
# --------------------------------------------------------------------------

BODY_KEYS = ("body", "text", "message", "content", "messageText")
SENDER_KEYS = ("sender", "name", "from", "author", "sender_name", "senderName")
TIME_KEYS = ("timestamp", "sent_at", "sentAt", "timestamp_ms", "date", "sent", "time")
SKIP_TEXT = {"", "media message", "attachment", "sticker", "(no text)", "null", "none"}

# U+FE0F (variation selector) is deliberately absent — it's a modifier that
# trails other emoji, and counting it alone puts an invisible "emoji" in the list.
EMOJI_RE = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff"
    "\U00002b00-\U00002bff\U00002190-\U000021ff]"
)
WORD_RE = re.compile(r"[a-z']{2,}")
LEAKED_TAGS = re.compile(r"<thinking>.*?</thinking>\s*", re.DOTALL | re.IGNORECASE)
STRAY_TAGS = re.compile(r"</?(thinking|antml:[a-z_]+)>", re.IGNORECASE)

STOPWORDS = {
    "the", "and", "you", "that", "for", "are", "but", "not", "with", "was",
    "have", "this", "just", "its", "it's", "what", "your", "they", "then",
    "there", "were", "from", "will", "would", "can", "did", "how", "all",
    "get", "got", "out", "one", "about", "when", "them", "want", "she", "his",
    "her", "him", "has", "had", "who", "why", "our", "off", "too", "any",
    "some", "than", "into", "been", "more", "come", "know", "like", "think",
}
SLANG = {
    "lol", "lmao", "lmfao", "haha", "hahaha", "hehe", "idk", "idc", "tbh",
    "ngl", "fr", "frfr", "bruh", "bro", "dude", "omg", "wtf", "smh", "ig",
    "imo", "rn", "btw", "nvm", "ty", "np", "yeah", "yea", "ya", "nah", "nope",
    "yep", "yup", "ok", "okay", "kk", "sry", "sorry", "pls", "plz", "thx",
    "u", "ur", "yall", "gonna", "wanna", "gotta", "kinda", "sorta", "prob",
    "probs", "def", "af", "istg", "wyd", "hbu", "lowkey", "deadass", "bet",
    "babe", "baby", "love", "miss", "cute", "aww", "xx",
}


def load_chat(path):
    """Read the file as JSON, or as JSON Lines if that fails.

    Plenty of exporters write one object per line rather than a single
    document, which json.loads rejects with a confusing "Extra data" error.
    """
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    records, bad = [], 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if not records:
        sys.exit(f"{path} isn't valid JSON or JSON Lines.")
    if bad:
        print(f"Note: skipped {bad:,} unreadable line(s).", file=sys.stderr)
    return records


def find_messages(obj):
    """Dig the message list out of whatever shape the JSON happens to be."""
    if isinstance(obj, list):
        msgs = [m for m in obj if isinstance(m, dict) and any(k in m for k in BODY_KEYS)]
        if msgs:
            return msgs
        found = []
        for item in obj:
            found.extend(find_messages(item))
        return found
    if isinstance(obj, dict):
        for key in ("messages", "conversation", "chat", "data"):
            if key in obj:
                found = find_messages(obj[key])
                if found:
                    return found
        found = []
        for value in obj.values():
            found.extend(find_messages(value))
        return found
    return []


def field(msg, keys):
    for key in keys:
        if key in msg and msg[key] not in (None, ""):
            return msg[key]
    return None


def text_of(msg):
    # Call logs and deleted messages carry body text that reads like something
    # you actually typed ("Outgoing voice call (unanswered)"), so they have to
    # be filtered on their flags — matching on the words would miss them and
    # they'd end up in the style profile.
    for flag in ("deleted", "call", "missed"):
        if msg.get(flag) is True:
            return ""
    value = field(msg, BODY_KEYS)
    if not isinstance(value, str):
        return ""
    value = value.strip()
    return "" if value.lower() in SKIP_TEXT else value


def time_of(msg):
    value = field(msg, TIME_KEYS)
    if isinstance(value, (int, float)):
        return float(value) / 1000 if value > 1e11 else float(value)
    if isinstance(value, str):
        try:
            from datetime import datetime

            return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def sender_of(msg):
    """Who sent this. Explicit direction fields win over the name field."""
    for key in ("isFromMe", "fromMe", "from_me", "is_from_me"):
        if isinstance(msg.get(key), bool):
            return "__me__" if msg[key] else "__them__"
    kind = msg.get("type") or msg.get("direction")
    if isinstance(kind, str):
        if kind.strip().lower() in ("outgoing", "sent", "out"):
            return "__me__"
        if kind.strip().lower() in ("incoming", "received", "in"):
            return "__them__"
    value = field(msg, SENDER_KEYS)
    return str(value).strip() if value else ""


# --------------------------------------------------------------------------
# Turning messages into turns, and turns into examples
# --------------------------------------------------------------------------


def build_turns(messages, me):
    """Group into sessions, then into speaker turns (a burst of texts = one turn)."""
    rows = []
    for msg in messages:
        body = text_of(msg)
        who = sender_of(msg)
        if body and who:
            rows.append((time_of(msg), "me" if who == me else "them", body))
    rows.sort(key=lambda r: r[0])

    sessions, current, prev = [], [], None
    for stamp, speaker, body in rows:
        if prev and stamp and prev > 0 and stamp - prev > 6 * 3600:
            sessions.append(current)
            current = []
        if current and current[-1][0] == speaker:
            current[-1][1].append(body)
        else:
            current.append([speaker, [body]])
        prev = stamp
    if current:
        sessions.append(current)
    return sessions


def build_examples(sessions, limit, max_turns=6):
    """Blocks that start with them and end with you, tiled so none overlap.

    Tiling matters: an earlier version walked to each of your turns and reached
    backwards, but since conversation alternates, the no-overlap rule squeezed
    every block down to a bare two-turn ping-pong with no context in it. Cutting
    the session into consecutive windows keeps several turns of real
    back-and-forth per example.
    """
    blocks = []
    for turns in sessions:
        for i in range(0, len(turns), max_turns):
            window = turns[i : i + max_turns]
            start = 0
            while start < len(window) and window[start][0] != "them":
                start += 1
            end = len(window) - 1
            while end >= 0 and window[end][0] != "me":
                end -= 1
            if start >= end:
                continue
            block = window[start : end + 1]
            if sum(len(p) for _, parts in block for p in parts) > 1200:
                continue
            blocks.append(block)
    if len(blocks) <= limit:
        return blocks
    # Sample evenly across the whole history so the mix of long and one-word
    # replies stays true to life, instead of just taking the oldest ones.
    step = len(blocks) / limit
    return [blocks[int(i * step)] for i in range(limit)]


def style_card(sessions, name):
    """Measured facts about how you text. These steer the model harder than
    any amount of 'text casually' hand-waving."""
    turns = [t for s in sessions for t in s if t[0] == "me"]
    msgs = [p for _, parts in turns for p in parts]
    if not msgs:
        return None, 0

    pct = lambda n: round(100 * n / len(msgs), 1)
    starts = [m for m in msgs if m[:1].isalpha()]
    emoji = Counter()
    with_emoji = 0
    for m in msgs:
        hits = EMOJI_RE.findall(m)
        if hits:
            with_emoji += 1
            emoji.update(hits)
    words = Counter()
    for m in msgs:
        words.update(WORD_RE.findall(m.lower()))
    slang = [(w, c) for w, c in words.most_common() if w in SLANG][:12]
    # Length floor: two- and three-letter words are almost all glue ("it", "to",
    # "so") and crowd out anything that actually sounds like you. Short words
    # worth keeping are already caught as slang above.
    common = [
        w for w, _ in words.most_common()
        if len(w) >= 4 and w not in STOPWORDS and w not in SLANG
    ][:12]

    lines = [
        f"- Measured from {len(msgs):,} real messages sent by {name}.",
        f"- Typical message: {int(statistics.median(len(m.split()) for m in msgs))} words / "
        f"{int(statistics.median(len(m) for m in msgs))} characters. "
        f"{pct(sum(1 for m in msgs if len(m.split()) <= 1))}% are a single word.",
        f"- Sends {round(len(msgs) / len(turns), 2)} messages in a row before waiting for a reply.",
        f"- Starts with a lowercase letter "
        f"{round(100 * sum(1 for m in starts if m[0].islower()) / len(starts), 1) if starts else 0}% "
        "of the time.",
        f"- Ends with no punctuation {pct(sum(1 for m in msgs if m[-1] not in '.?!…'))}% of the time "
        f"(period {pct(sum(1 for m in msgs if m.endswith('.')))}%, "
        f"question mark {pct(sum(1 for m in msgs if m.endswith('?')))}%).",
        f"- Uses emoji in {pct(with_emoji)}% of messages"
        + (f"; most used: {' '.join(e for e, _ in emoji.most_common(6))}." if emoji else "."),
    ]
    if slang:
        lines.append("- Habitual words: " + ", ".join(f"{w} ({c})" for w, c in slang))
    if common:
        lines.append("- Also says a lot: " + ", ".join(common))
    return "\n".join(lines), len(msgs)


def render_all(sessions, name):
    """Every message you have, grouped by conversation. Used by FULL_HISTORY."""
    out = []
    for i, turns in enumerate(sessions, 1):
        lines = [f'<conversation id="{i}">']
        for speaker, parts in turns:
            who = name if speaker == "me" else "THEM"
            lines.extend(f"{who}: {p}" for p in parts)
        lines.append("</conversation>")
        out.append("\n".join(lines))
    return "\n\n".join(out)


def render_examples(blocks, name):
    out = []
    for i, block in enumerate(blocks, 1):
        lines = [f'<exchange id="{i}">']
        for speaker, parts in block:
            who = name if speaker == "me" else "THEM"
            lines.extend(f"{who}: {p}" for p in parts)
        lines.append("</exchange>")
        out.append("\n".join(lines))
    return "\n\n".join(out)


SYSTEM = """\
You are texting as {name}. Everything below was measured or copied from {name}'s \
real messages. Reply the way {name} actually would.

# How {name} texts

{style}

# Real conversations

THEM is the person {name} was talking to. Notice the rhythm as much as the \
words: how long replies run, when they're one word, when several fire in a row.

{examples}

# How to reply

- Write only what {name} would send. No narration, no quotation marks.
- Match the measurements above: length, capitalisation, punctuation, emoji rate, \
slang. If {name} rarely capitalises or rarely ends with a period, neither do you.
- Never use markdown. No bullets, no headers, no bold. Nobody formats a text.
- Most replies are more than one message. Split your reply across two or three \
lines more often than not — a single line should be the exception.
- Roughly one reply in five should contain an emoji, drawn from the ones listed \
above. Otherwise use none.
- Don't be helpful the way an assistant is. Don't offer options, summarise, or ask \
if there's anything else. {name} texts to talk, not to serve.
- You know {name}'s voice, not {name}'s life. If asked about plans or events you \
have no evidence for, deflect like a real person would. Don't invent details.
- If someone sincerely asks whether they're talking to a person or an AI, tell \
them the truth.
- Do not include internal or system XML tags in your response.
"""


def choose_me(messages):
    """Work out which sender is you, asking if it isn't obvious."""
    counts = Counter(sender_of(m) for m in messages if text_of(m) and sender_of(m))
    if not counts:
        sys.exit(f"No messages with readable text found in {CHAT_FILE}.")
    if "__me__" in counts:
        return "__me__", counts  # the export already marks your own messages

    people = counts.most_common()
    if len(people) == 1:
        return people[0][0], counts

    # Exports commonly label your own side "Me". The list is ordered by volume,
    # so without this the obvious pick is whoever talked more — which is often
    # the other person, and you'd clone them instead.
    labelled = [who for who, _ in people if who.lower() in ("me", "you", "self", "myself")]
    if len(labelled) == 1:
        others = ", ".join(w for w, _ in people if w != labelled[0])
        print(f"You are {labelled[0]!r} in this export (talking to {others}).")
        return labelled[0], counts

    print("Found these people in the chat:")
    for i, (who, count) in enumerate(people, 1):
        print(f"  {i}. {who} ({count:,} messages)")
    while True:
        pick = input(f"Which one is you? [1-{len(people)}]: ").strip()
        if pick.isdigit() and 1 <= int(pick) <= len(people):
            return people[int(pick) - 1][0], counts
        print("Pick a number from the list.")


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else CHAT_FILE)
    if not path.exists():
        sys.exit(
            f"Can't find {path}.\n"
            f"Put your chat export next to this script as {CHAT_FILE}, "
            f"or run: python my_clone.py path/to/your-file.json"
        )

    messages = find_messages(load_chat(path))
    if not messages:
        sys.exit(f"Couldn't find any messages in {path}. Is this a chat export?")

    me, counts = choose_me(messages)
    name = input("What should the clone be called? [me]: ").strip() or "me"

    # The clone plays you, so the human seat is the other person's. Label the
    # input prompt with their name — "you>" reads as though you're typing as
    # yourself, which is backwards.
    other = next((who for who, _ in counts.most_common() if who != me), "them")
    if other.startswith("__"):
        other = "them"

    sessions = build_turns(messages, me)
    style, count = style_card(sessions, name)
    if not style:
        sys.exit(f"Found {len(messages)} messages but none from you. Try again and pick a different person.")

    if FULL_HISTORY:
        body = render_all(sessions, name)
        shown = sum(len(ps) for t in sessions for _, ps in t)
    else:
        blocks = build_examples(sessions, EXAMPLES)
        body = render_examples(blocks, name)
        shown = sum(len(ps) for b in blocks for _, ps in b)
    system_prompt = SYSTEM.format(name=name, style=style, examples=body)

    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")
    key_file = Path(__file__).parent / "key.txt"
    key = API_KEY.strip() or (
        key_file.read_text(encoding="utf-8").strip() if key_file.exists() else ""
    ) or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        sys.exit(
            "No API key. Either paste one into API_KEY at the top of this file, "
            "save it in key.txt next to this script, or set ANTHROPIC_API_KEY."
        )

    client = anthropic.Anthropic(api_key=key)

    # A 5-minute cache is fine for a small prompt, but re-writing a full history
    # after every idle gap is the single most expensive thing this script can do.
    cache_control = {"type": "ephemeral", "ttl": "1h"} if FULL_HISTORY else {"type": "ephemeral"}

    # Haiku 4.5 rejects the effort parameter outright ("This model does not
    # support the effort parameter"), so only send it where it exists.
    effort = {} if "haiku" in MODEL else {"output_config": {"effort": "low"}}

    if FULL_HISTORY:
        used = client.messages.count_tokens(
            model=MODEL, system=system_prompt, messages=[{"role": "user", "content": "hey"}]
        ).input_tokens
        # Ask the API for the window rather than assuming 1M — haiku is 200K,
        # and a hardcoded limit would wave a far-too-large prompt through and
        # then fail on the first real request.
        try:
            limit = client.models.retrieve(MODEL).max_input_tokens
        except Exception:
            limit = 200_000 if "haiku" in MODEL else 1_000_000
        room = limit - used
        if room < 20_000:
            sys.exit(
                f"Full history is {used:,} tokens and won't leave room to reply "
                f"({room:,} left of {limit:,} for {MODEL}).\n"
                "Set FULL_HISTORY = False and raise EXAMPLES instead, "
                "or switch MODEL to one with a bigger context window."
            )
        print(
            f"Full history: {used:,} tokens ({room:,} to spare). "
            f"About ${used * 5 / 1_000_000 * 1.25:.2f} to start, "
            f"${used * 5 / 1_000_000 * 0.10:.2f} per reply."
        )
    history = []

    print(f"\nLearned from {count:,} of your messages; {shown:,} in the prompt.")
    print(f"You're texting as {other}; {name} replies as you.")
    print("Type /reset to start over, /style to see your profile, /quit to leave.\n")

    while True:
        try:
            said = input(f"{other.lower()}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not said:
            continue
        if said in ("/quit", "/exit"):
            return
        if said == "/reset":
            history.clear()
            print("(cleared)\n")
            continue
        if said == "/style":
            print(style + "\n")
            continue

        history.append({"role": "user", "content": said})

        # Only resend the recent tail, and make sure it starts on a user turn —
        # a plain slice can begin on an assistant message, which the API rejects.
        window = history[-HISTORY_TURNS:]
        first = next((i for i, m in enumerate(window) if m["role"] == "user"), 0)

        # The "..." typing indicator is erased with \r, which only works on a
        # real terminal — piped or redirected output would keep the artifact.
        label = f"{name.lower()}> "
        tty = sys.stdout.isatty()
        erase = f"\r{' ' * (len(label) + 3)}\r" if tty else ""
        if tty:
            print(f"{label}...", end="", flush=True)
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                # One cached block: every turn after the first reads the whole
                # style prompt at a fraction of the input price.
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": cache_control,
                    }
                ],
                messages=window[first:],
                # Thinking is on by default on the 5-series. For texting it just
                # adds latency and over-considered replies.
                thinking={"type": "disabled"},
                **effort,
            ) as stream:
                message = stream.get_final_message()
        except Exception as exc:
            history.pop()
            print(f"{erase}[error: {exc}]\n")
            continue

        reply = "".join(b.text for b in message.content if b.type == "text")
        reply = STRAY_TAGS.sub("", LEAKED_TAGS.sub("", reply)).strip()

        if message.stop_reason == "refusal" or not reply:
            history.pop()
            print(f"{erase}[no reply — try rephrasing]\n")
            continue

        history.append({"role": "assistant", "content": reply})
        print(erase, end="")
        for i, line in enumerate(reply.splitlines()):
            if line.strip():
                print(f"{label if i == 0 else ' ' * len(label)}{line}")
        print()


if __name__ == "__main__":
    main()

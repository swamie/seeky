"""Turn a sigexport (Signal) JSON export into few-shot texting-style material.

Two things come out of here:

1. A *style profile* — measured facts about how you text (message length,
   capitalisation, punctuation, emoji rate, slang, how many bubbles you fire
   off in a row). Claude follows measured facts far more reliably than a vague
   "text like me".
2. A set of *exchanges* — real back-and-forth blocks that always end on one of
   your messages, so the examples demonstrate the thing being imitated.

Nothing here talks to the API. Run it directly to inspect what it found:

    python signal_parser.py --export C:\\Temp\\MySignalExport --preview
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# --------------------------------------------------------------------------
# Field-name tolerance
#
# sigexport's JSON shape has drifted across versions, and people also hand-roll
# exports. Rather than pin one schema we probe a handful of plausible keys.
# --------------------------------------------------------------------------

BODY_KEYS = ("body", "text", "message", "content", "messageText")
SENDER_KEYS = ("sender", "name", "from", "author", "sender_name", "senderName", "profileName")
TIME_KEYS = (
    "timestamp",
    "sent_at",
    "sentAt",
    "timestamp_ms",
    "received_at",
    "date",
    "sent",
    "time",
)
OUTGOING_FLAG_KEYS = ("isFromMe", "fromMe", "from_me", "is_from_me", "outgoing")
OUTGOING_TYPES = {"outgoing", "sent", "out", "me"}
INCOMING_TYPES = {"incoming", "received", "in"}

# Rows sigexport emits for non-message events, plus placeholders it writes when
# a message carried no text of its own.
NON_MESSAGE_TEXT = {
    "",
    "media message",
    "attachment",
    "sticker",
    "(no text)",
    "null",
    "none",
}

EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0001f1e6-\U0001f1ff"
    "\U00002b00-\U00002bff"
    "\U0000fe0f"
    "\U00002190-\U000021ff"
    "]"
)

WORD_RE = re.compile(r"[a-z']{2,}")

# Deliberately small. The goal is to surface *your* vocabulary, so only the
# highest-frequency English glue is removed.
STOPWORDS = {
    "the", "and", "you", "that", "for", "are", "but", "not", "with", "was",
    "have", "this", "just", "its", "it's", "what", "your", "they", "then",
    "there", "were", "from", "will", "would", "can", "did", "how", "all",
    "get", "got", "out", "one", "about", "when", "them", "want", "she", "his",
    "her", "him", "has", "had", "who", "why", "our", "off", "too", "any",
    "some", "than", "into", "been", "more", "come", "know", "like", "think",
    "going", "really", "even", "also", "because", "should", "could",
}

# Texting tics worth calling out by name — these carry a lot of voice.
SLANG_MARKERS = (
    "lol", "lmao", "lmfao", "haha", "hahaha", "hehe", "idk", "idc", "tbh",
    "ngl", "fr", "frfr", "bruh", "bro", "dude", "omg", "wtf", "smh", "ig",
    "imo", "rn", "btw", "nvm", "ty", "np", "yeah", "yea", "ya", "nah", "nope",
    "yep", "yup", "ok", "okay", "kk", "sry", "sorry", "pls", "plz", "thx",
    "u", "ur", "ya'll", "yall", "gonna", "wanna", "gotta", "kinda", "sorta",
    "prob", "probs", "def", "af", "istg", "ffs", "wyd", "hbu", "wbu", "lowkey",
    "highkey", "deadass", "bet", "word", "facts", "damn", "shit", "fuck",
)


@dataclass
class Turn:
    """One speaker's uninterrupted run of messages.

    Texting is bursty — three bubbles in a row is one conversational turn, and
    the burst pattern itself is part of the style being copied. `parts` keeps
    the individual bubbles; `text` is the joined form fed to the model.
    """

    speaker: str  # "me" or "them"
    parts: list[str] = field(default_factory=list)
    started_at: float = 0.0

    @property
    def text(self) -> str:
        return "\n".join(self.parts)


@dataclass
class Exchange:
    """A short block of alternating turns that ends on one of *my* messages."""

    conversation: str
    turns: list[Turn]

    @property
    def char_len(self) -> int:
        return sum(len(t.text) for t in self.turns)


@dataclass
class StyleProfile:
    message_count: int = 0
    turn_count: int = 0
    median_chars: int = 0
    median_words: int = 0
    lowercase_start_pct: float = 0.0
    ends_period_pct: float = 0.0
    ends_question_pct: float = 0.0
    ends_exclaim_pct: float = 0.0
    no_end_punct_pct: float = 0.0
    emoji_msg_pct: float = 0.0
    top_emoji: list[tuple[str, int]] = field(default_factory=list)
    slang: list[tuple[str, int]] = field(default_factory=list)
    top_words: list[tuple[str, int]] = field(default_factory=list)
    burst_rate: float = 0.0
    one_word_pct: float = 0.0


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _iter_json_files(path: Path) -> Iterator[Path]:
    if path.is_file():
        yield path
        return
    if not path.exists():
        raise FileNotFoundError(f"No such export path: {path}")
    for candidate in sorted(path.rglob("*.json")):
        yield candidate


def _looks_like_message(obj: Any) -> bool:
    return isinstance(obj, dict) and any(k in obj for k in BODY_KEYS + TIME_KEYS)


def _extract_conversations(obj: Any, default_name: str) -> list[tuple[str, list[dict]]]:
    """Pull ``(conversation_name, messages)`` pairs out of whatever shape we got.

    Handles the three layouts seen in the wild: a bare list of messages, a dict
    keyed by contact name, and a wrapper object with a ``messages`` field.
    """
    out: list[tuple[str, list[dict]]] = []

    if isinstance(obj, list):
        msgs = [m for m in obj if _looks_like_message(m)]
        if msgs:
            out.append((default_name, msgs))
        else:
            # A list of conversation objects rather than a list of messages.
            for entry in obj:
                if isinstance(entry, dict):
                    name = str(entry.get("name") or entry.get("title") or default_name)
                    out.extend(_extract_conversations(entry, name))
        return out

    if isinstance(obj, dict):
        for key in ("messages", "conversation", "chat", "data"):
            if key in obj:
                name = str(obj.get("name") or obj.get("title") or default_name)
                out.extend(_extract_conversations(obj[key], name))
        if out:
            return out
        # Dict keyed by contact name -> list of messages.
        for key, value in obj.items():
            if isinstance(value, (list, dict)):
                out.extend(_extract_conversations(value, str(key)))
        return out

    return out


def load_conversations(export_path: str | Path) -> list[tuple[str, list[dict]]]:
    """Read every JSON file under ``export_path`` into named message lists."""
    path = Path(export_path).expanduser()
    conversations: list[tuple[str, list[dict]]] = []
    for json_file in _iter_json_files(path):
        try:
            raw = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        # For `<contact>/messages.json` the parent dir is the useful name.
        default_name = json_file.stem
        if default_name in {"messages", "index", "chat", "data"}:
            default_name = json_file.parent.name
        conversations.extend(_extract_conversations(raw, default_name))
    return [(name, msgs) for name, msgs in conversations if msgs]


# --------------------------------------------------------------------------
# Message-level accessors
# --------------------------------------------------------------------------


def _first_key(msg: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in msg and msg[key] not in (None, ""):
            return msg[key]
    return None


def message_text(msg: dict) -> str:
    value = _first_key(msg, BODY_KEYS)
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if text.lower() in NON_MESSAGE_TEXT:
        return ""
    return text


def message_time(msg: dict) -> float:
    """Seconds since epoch, best effort. Unknown timestamps sort as 0."""
    value = _first_key(msg, TIME_KEYS)
    if isinstance(value, (int, float)):
        # Signal stores milliseconds; anything past ~year 2286 in seconds is ms.
        return float(value) / 1000.0 if value > 1e11 else float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("Z", "+00:00")
        try:
            from datetime import datetime

            return datetime.fromisoformat(cleaned).timestamp()
        except ValueError:
            try:
                return float(cleaned) / 1000.0 if float(cleaned) > 1e11 else float(cleaned)
            except ValueError:
                return 0.0
    return 0.0


def message_sender(msg: dict) -> str:
    value = _first_key(msg, SENDER_KEYS)
    return str(value).strip() if value is not None else ""


def is_from_me(msg: dict, me_names: set[str]) -> bool | None:
    """True/False if we can tell who sent it, None if the row is unusable.

    Explicit direction fields win over name matching, since a contact whose
    display name happens to match yours would otherwise poison the examples.
    """
    for key in OUTGOING_FLAG_KEYS:
        if key in msg and isinstance(msg[key], bool):
            return msg[key]

    msg_type = msg.get("type") or msg.get("direction")
    if isinstance(msg_type, str):
        lowered = msg_type.strip().lower()
        if lowered in OUTGOING_TYPES:
            return True
        if lowered in INCOMING_TYPES:
            return False

    sender = message_sender(msg).lower()
    if not sender:
        return None
    return sender in me_names


# --------------------------------------------------------------------------
# Turn / exchange construction
# --------------------------------------------------------------------------


def build_turns(
    messages: list[dict],
    me_names: set[str],
    session_gap_seconds: float = 6 * 3600,
) -> list[list[Turn]]:
    """Group messages into sessions, and sessions into speaker turns.

    A gap longer than ``session_gap_seconds`` starts a new session, so we never
    stitch "goodnight" onto the next morning's unrelated opener.
    """
    usable: list[tuple[float, bool, str]] = []
    for msg in messages:
        text = message_text(msg)
        if not text:
            continue
        mine = is_from_me(msg, me_names)
        if mine is None:
            continue
        usable.append((message_time(msg), mine, text))

    usable.sort(key=lambda row: row[0])

    sessions: list[list[Turn]] = []
    current: list[Turn] = []
    prev_time: float | None = None

    for timestamp, mine, text in usable:
        speaker = "me" if mine else "them"
        new_session = (
            prev_time is not None
            and timestamp > 0
            and prev_time > 0
            and (timestamp - prev_time) > session_gap_seconds
        )
        if new_session and current:
            sessions.append(current)
            current = []

        if current and current[-1].speaker == speaker:
            current[-1].parts.append(text)
        else:
            current.append(Turn(speaker=speaker, parts=[text], started_at=timestamp))
        prev_time = timestamp

    if current:
        sessions.append(current)
    return sessions


def build_exchanges(
    conversation: str,
    sessions: list[list[Turn]],
    max_turns: int = 6,
    max_chars: int = 1200,
) -> list[Exchange]:
    """Slice sessions into blocks that start on *them* and end on *me*.

    Blocks never overlap. An earlier version let each block reach back
    ``max_turns`` regardless of where the previous one ended, which produced
    examples that were strict supersets of their predecessors — a lot of prompt
    spent re-showing the same conversation.
    """
    exchanges: list[Exchange] = []

    for turns in sessions:
        consumed = -1  # index of the last turn already used by a block
        for i, turn in enumerate(turns):
            if turn.speaker != "me" or i == 0:
                continue
            if turns[i - 1].speaker != "them":
                continue

            start = max(consumed + 1, i - max_turns + 1)
            while start < i and turns[start].speaker != "them":
                start += 1
            if start >= i:
                continue  # nothing left that isn't already in an earlier block

            block = turns[start : i + 1]
            exchange = Exchange(conversation=conversation, turns=block)
            if exchange.char_len > max_chars:
                # Fall back to the minimal them -> me pair, if it's still free.
                if i - 1 <= consumed:
                    continue
                exchange = Exchange(conversation=conversation, turns=turns[i - 1 : i + 1])
                if exchange.char_len > max_chars:
                    continue
            exchanges.append(exchange)
            consumed = i

    return exchanges


def select_examples(exchanges: list[Exchange], limit: int) -> list[Exchange]:
    """Sample evenly across the whole history rather than taking the first N.

    Even sampling preserves the real distribution of reply lengths — including
    the one-word ones, which are a big part of how most people actually text.
    """
    if limit <= 0 or len(exchanges) <= limit:
        return list(exchanges)
    step = len(exchanges) / limit
    return [exchanges[int(i * step)] for i in range(limit)]


# --------------------------------------------------------------------------
# Style measurement
# --------------------------------------------------------------------------


def profile_style(sessions_by_conversation: list[list[list[Turn]]]) -> StyleProfile:
    my_turns: list[Turn] = []
    for sessions in sessions_by_conversation:
        for turns in sessions:
            my_turns.extend(t for t in turns if t.speaker == "me")

    messages = [part for turn in my_turns for part in turn.parts]
    profile = StyleProfile(message_count=len(messages), turn_count=len(my_turns))
    if not messages:
        return profile

    def pct(count: int) -> float:
        return round(100.0 * count / len(messages), 1)

    char_lengths = [len(m) for m in messages]
    word_counts = [len(m.split()) for m in messages]
    profile.median_chars = int(statistics.median(char_lengths))
    profile.median_words = int(statistics.median(word_counts))
    profile.one_word_pct = pct(sum(1 for w in word_counts if w <= 1))

    alpha_start = [m for m in messages if m[:1].isalpha()]
    if alpha_start:
        lower = sum(1 for m in alpha_start if m[0].islower())
        profile.lowercase_start_pct = round(100.0 * lower / len(alpha_start), 1)

    profile.ends_period_pct = pct(sum(1 for m in messages if m.endswith(".")))
    profile.ends_question_pct = pct(sum(1 for m in messages if m.endswith("?")))
    profile.ends_exclaim_pct = pct(sum(1 for m in messages if m.endswith("!")))
    profile.no_end_punct_pct = pct(sum(1 for m in messages if m[-1] not in ".?!…"))

    emoji_counter: Counter[str] = Counter()
    messages_with_emoji = 0
    for m in messages:
        found = EMOJI_RE.findall(m)
        if found:
            messages_with_emoji += 1
            emoji_counter.update(found)
    profile.emoji_msg_pct = pct(messages_with_emoji)
    profile.top_emoji = emoji_counter.most_common(6)

    word_counter: Counter[str] = Counter()
    for m in messages:
        word_counter.update(WORD_RE.findall(m.lower()))

    profile.slang = [(w, c) for w, c in word_counter.most_common() if w in SLANG_MARKERS][:12]
    slang_seen = {w for w, _ in profile.slang}
    profile.top_words = [
        (w, c)
        for w, c in word_counter.most_common()
        if w not in STOPWORDS and w not in slang_seen and len(w) > 2
    ][:12]

    profile.burst_rate = round(len(messages) / len(my_turns), 2)
    return profile


def render_style_card(profile: StyleProfile, name: str) -> str:
    if not profile.message_count:
        return "(no messages found — check --me and --export)"

    lines = [
        f"- Measured from {profile.message_count:,} real messages sent by {name}.",
        f"- Typical message: {profile.median_words} words / {profile.median_chars} characters (median). "
        f"{profile.one_word_pct}% are a single word.",
        f"- Sends {profile.burst_rate} messages in a row before waiting for a reply.",
        f"- Starts with a lowercase letter {profile.lowercase_start_pct}% of the time.",
        f"- Ends a message with no punctuation {profile.no_end_punct_pct}% of the time "
        f"(period {profile.ends_period_pct}%, question mark {profile.ends_question_pct}%, "
        f"exclamation {profile.ends_exclaim_pct}%).",
        f"- Uses emoji in {profile.emoji_msg_pct}% of messages"
        + (
            f"; most used: {' '.join(e for e, _ in profile.top_emoji)}."
            if profile.top_emoji
            else "."
        ),
    ]
    if profile.slang:
        lines.append(
            "- Habitual filler and slang, with counts: "
            + ", ".join(f"{w} ({c})" for w, c in profile.slang)
        )
    if profile.top_words:
        lines.append(
            "- Frequently used words: " + ", ".join(w for w, _ in profile.top_words)
        )
    return "\n".join(lines)


def render_exchanges(exchanges: list[Exchange], name: str) -> str:
    blocks = []
    for i, exchange in enumerate(exchanges, start=1):
        rendered = [f"<exchange id=\"{i}\">"]
        for turn in exchange.turns:
            speaker = name if turn.speaker == "me" else "THEM"
            for part in turn.parts:
                rendered.append(f"{speaker}: {part}")
        rendered.append("</exchange>")
        blocks.append("\n".join(rendered))
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------
# Top-level convenience
# --------------------------------------------------------------------------


@dataclass
class ExportSummary:
    profile: StyleProfile
    examples: list[Exchange]
    conversations: list[tuple[str, int]]  # (name, my-message count)
    senders: list[tuple[str, int]]  # every sender seen, for diagnosing --me


def analyse_export(
    export_path: str | Path,
    me_names: set[str] | None = None,
    contact: str | None = None,
    max_examples: int = 80,
    max_turns_per_exchange: int = 6,
) -> ExportSummary:
    me_names = {n.lower() for n in (me_names or {"me"})}
    conversations = load_conversations(export_path)
    if not conversations:
        raise ValueError(f"No JSON messages found under {export_path}")

    if contact:
        needle = contact.lower()
        matched = [(n, m) for n, m in conversations if needle in n.lower()]
        if not matched:
            available = ", ".join(sorted({n for n, _ in conversations})[:20])
            raise ValueError(f"No conversation matching {contact!r}. Found: {available}")
        conversations = matched

    all_sessions: list[list[list[Turn]]] = []
    all_exchanges: list[Exchange] = []
    per_conversation: list[tuple[str, int]] = []
    sender_counter: Counter[str] = Counter()

    for name, messages in conversations:
        for msg in messages:
            if message_text(msg):
                sender_counter[message_sender(msg) or "(unnamed)"] += 1
        sessions = build_turns(messages, me_names)
        if not sessions:
            continue
        all_sessions.append(sessions)
        all_exchanges.extend(build_exchanges(name, sessions, max_turns=max_turns_per_exchange))
        mine = sum(len(t.parts) for s in sessions for t in s if t.speaker == "me")
        if mine:
            per_conversation.append((name, mine))

    profile = profile_style(all_sessions)
    examples = select_examples(all_exchanges, max_examples)
    per_conversation.sort(key=lambda row: row[1], reverse=True)
    return ExportSummary(
        profile=profile,
        examples=examples,
        conversations=per_conversation,
        senders=sender_counter.most_common(),
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, help="Export directory or a single .json file")
    parser.add_argument("--me", default="Me", help="How you appear in the export (default: Me)")
    parser.add_argument("--contact", help="Only use conversations whose name contains this")
    parser.add_argument("--max-examples", type=int, default=80)
    parser.add_argument("--name", default="YOU", help="Label to use for your messages")
    parser.add_argument("--preview", action="store_true", help="Print a few example exchanges")
    args = parser.parse_args()

    try:
        summary = analyse_export(
            args.export,
            me_names={args.me},
            contact=args.contact,
            max_examples=args.max_examples,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))

    if not summary.profile.message_count:
        seen = "\n".join(f"  {c:>6,}  {s}" for s, c in summary.senders[:15])
        raise SystemExit(
            f"Found conversations but nothing sent by {args.me!r}.\n"
            f"Sender names in this export:\n{seen or '  (none)'}\n"
            "Pass one of these as --me."
        )

    print("Conversations (by messages you sent):")
    for name, count in summary.conversations[:15]:
        print(f"  {count:>6,}  {name}")
    print()
    print(render_style_card(summary.profile, args.name))
    print()
    print(f"Selected {len(summary.examples)} example exchanges.")
    if args.preview:
        print()
        print(render_exchanges(summary.examples[:5], args.name))


if __name__ == "__main__":
    _main()

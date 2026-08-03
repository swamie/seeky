"""A chatbot that texts the way you do, built from your Signal export.

Anthropic doesn't offer self-serve fine-tuning, so this uses few-shot prompting
instead: your real exchanges go into the system prompt and Claude imitates the
voice it reads there. With a 1M-token context window there's plenty of room for
a large, representative sample.

    python my_clone.py --export C:\\Temp\\MySignalExport --name Kay

Type /help once you're in.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass

import signal_parser as sp

MODEL = "claude-opus-5"

# Texts are short, so a small cap keeps replies from drifting into essays.
MAX_TOKENS = 400

# How many prior turns of the live conversation to resend. The API is
# stateless, so this is the memory knob — and the cost knob.
HISTORY_TURNS = 24

# Opus 5 with thinking disabled can occasionally leak internal XML into the
# visible response. The system prompt asks it not to; this is the backstop.
LEAKED_TAG_RE = re.compile(r"<thinking>.*?</thinking>\s*", re.DOTALL | re.IGNORECASE)
STRAY_TAG_RE = re.compile(r"</?(thinking|antml:[a-z_]+)>", re.IGNORECASE)


SYSTEM_TEMPLATE = """\
You are texting as {name}. Everything below was measured or copied from {name}'s \
real message history. Your job is to reply the way {name} actually would.

# How {name} texts

{style_card}

# Real conversations

Each block is a genuine exchange. THEM is whoever {name} was talking to; \
{name} is you. Note the rhythm as much as the wording — how long replies run, \
when they're one word, when several messages fire in a row.

{examples}

# How to reply

- Write only what {name} would send. No narration, no stage directions, no \
quotation marks around the message.
- Match the measurements above: the length, the capitalisation, the punctuation \
habits, the emoji rate, the slang. If {name} rarely capitalises or rarely ends \
with a period, neither do you.
- Never use markdown. No headers, no bullet points, no bold, no numbered lists. \
Nobody formats a text message.
- To send several messages in a row, put each one on its own line. Only do this \
as often as the burst rate above suggests.
- Don't be helpful in the way an assistant is helpful. Don't offer options, \
summarise what was said, ask if there's anything else, or end with a question \
you only asked to keep things going. {name} texts to talk, not to serve.
- You know {name}'s voice, not {name}'s life. If asked about specific plans, \
people, or events you have no evidence for, deflect the way a real person does \
when they're distracted or don't feel like getting into it. Do not invent \
detailed facts about {name}.
- If someone sincerely asks whether they're talking to a real person or to an \
AI, tell them the truth. Style imitation is the point; deceiving someone about \
what they're talking to is not.
- Do not include internal or system XML tags in your response.
"""


@dataclass
class CloneConfig:
    name: str
    system_prompt: str
    model: str = MODEL
    effort: str = "low"
    thinking: bool = False
    max_tokens: int = MAX_TOKENS


def build_system_prompt(summary: sp.ExportSummary, name: str) -> str:
    return SYSTEM_TEMPLATE.format(
        name=name,
        style_card=sp.render_style_card(summary.profile, name),
        examples=sp.render_exchanges(summary.examples, name),
    )


def sanitise(text: str) -> str:
    text = LEAKED_TAG_RE.sub("", text)
    text = STRAY_TAG_RE.sub("", text)
    return text.strip()


class Clone:
    """Wraps the Messages API call and the running conversation history."""

    def __init__(self, client, config: CloneConfig):
        self.client = client
        self.config = config
        self.history: list[dict] = []
        # Flipped off permanently if the account or SDK rejects the parameter.
        self._fallbacks_enabled = True

    def _window(self) -> list[dict]:
        """The tail of the conversation to resend, always starting on a user turn.

        A plain ``history[-N:]`` slice can land on an assistant message, which
        the API rejects — and only once the chat is long enough to trim, so it
        fails well after everything looked fine.
        """
        window = self.history[-HISTORY_TURNS:]
        first_user = next((i for i, m in enumerate(window) if m["role"] == "user"), None)
        return window[first_user:] if first_user is not None else []

    def _request_kwargs(self) -> dict:
        kwargs: dict = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            # A single cached block: the style card and every example sit in a
            # stable prefix, so each turn after the first reads them at ~10% of
            # input price instead of reprocessing the whole transcript.
            "system": [
                {
                    "type": "text",
                    "text": self.config.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": self._window(),
            "output_config": {"effort": self.config.effort},
        }
        # Thinking is on by default on Opus 5. For imitating texting it mostly
        # adds latency and over-considered replies, so it's off unless asked
        # for. Disabling requires effort of "high" or lower.
        if self.config.thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        elif self.config.effort in ("low", "medium", "high"):
            kwargs["thinking"] = {"type": "disabled"}
        return kwargs

    def _stream(self, kwargs: dict):
        if self._fallbacks_enabled:
            try:
                return self.client.beta.messages.stream(
                    **kwargs,
                    betas=["server-side-fallback-2026-07-01"],
                    # Opus 5's safety classifiers can decline a request; this
                    # re-runs it on a fallback model server-side instead of
                    # returning an empty reply.
                    extra_body={"fallbacks": "default"},
                )
            except Exception:
                self._fallbacks_enabled = False
        return self.client.messages.stream(**kwargs)

    def reply(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})
        kwargs = self._request_kwargs()

        try:
            with self._stream(kwargs) as stream:
                message = stream.get_final_message()
        except Exception:
            # Don't leave a dangling user turn — the next attempt would send
            # two user messages in a row and get a confused reply.
            self.history.pop()
            raise

        if getattr(message, "stop_reason", None) == "refusal":
            self.history.pop()
            return "[declined — safety classifiers blocked that one; try rephrasing]"

        text = sanitise(
            "".join(block.text for block in message.content if block.type == "text")
        )
        if not text:
            self.history.pop()
            return "[empty reply — try again]"

        self.history.append({"role": "assistant", "content": text})
        return text

    def reset(self) -> None:
        self.history.clear()


HELP = """\
  /reset    forget the current conversation (style prompt stays loaded)
  /style    show the measured style profile
  /prompt   print the full system prompt being sent
  /quit     exit
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export", required=True, help="Signal export directory or a single .json file"
    )
    parser.add_argument("--name", default="You", help="Your name, used in the prompt")
    parser.add_argument(
        "--me", default="Me", help="How you appear in the export's sender field (default: Me)"
    )
    parser.add_argument("--contact", help="Only learn from conversations matching this name")
    parser.add_argument(
        "--max-examples", type=int, default=80, help="Exchanges to include (default: 80)"
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--effort",
        default="low",
        choices=["low", "medium", "high"],
        help="Higher is slower and pricier; texting rarely needs more than low",
    )
    parser.add_argument(
        "--thinking", action="store_true", help="Let Claude think before replying"
    )
    parser.add_argument("--show-prompt", action="store_true", help="Print the prompt and exit")
    parser.add_argument("--tokens", action="store_true", help="Report system prompt token count")
    args = parser.parse_args()

    try:
        summary = sp.analyse_export(
            args.export,
            me_names={args.me},
            contact=args.contact,
            max_examples=args.max_examples,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Could not read the export: {exc}", file=sys.stderr)
        return 1

    if not summary.profile.message_count:
        seen = "\n".join(f"  {c:>6,}  {s}" for s, c in summary.senders[:15])
        print(
            f"Found conversations but no messages from {args.me!r}.\n"
            f"Sender names in this export:\n{seen or '  (none)'}\n"
            "Pass one of these as --me.",
            file=sys.stderr,
        )
        return 1

    system_prompt = build_system_prompt(summary, args.name)

    if args.show_prompt:
        print(system_prompt)
        return 0

    try:
        import anthropic
    except ImportError:
        print("pip install anthropic", file=sys.stderr)
        return 1

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Note: ANTHROPIC_API_KEY isn't set. The SDK will fall back to an "
            "`ant auth login` profile if you have one.",
            file=sys.stderr,
        )

    client = anthropic.Anthropic()
    config = CloneConfig(
        name=args.name,
        system_prompt=system_prompt,
        model=args.model,
        effort=args.effort,
        thinking=args.thinking,
    )
    clone = Clone(client, config)

    print(
        f"Loaded {summary.profile.message_count:,} of your messages "
        f"across {len(summary.conversations)} conversation(s); "
        f"{len(summary.examples)} exchanges in the prompt."
    )
    if args.tokens:
        try:
            count = client.messages.count_tokens(
                model=args.model,
                system=system_prompt,
                messages=[{"role": "user", "content": "hey"}],
            )
            print(f"System prompt: ~{count.input_tokens:,} input tokens per uncached turn.")
        except Exception as exc:
            print(f"(token count unavailable: {exc})")
    print("/help for commands.\n")

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_text:
            continue
        if user_text in ("/quit", "/exit"):
            return 0
        if user_text == "/help":
            print(HELP)
            continue
        if user_text == "/reset":
            clone.reset()
            print("(conversation cleared)")
            continue
        if user_text == "/style":
            print(sp.render_style_card(summary.profile, args.name))
            continue
        if user_text == "/prompt":
            print(system_prompt)
            continue

        label = f"{args.name.lower()}> "
        print(f"{label}...", end="", flush=True)
        try:
            reply = clone.reply(user_text)
        except Exception as exc:
            print(f"\r{' ' * (len(label) + 3)}\r", end="")
            print(f"[error: {exc}]")
            continue

        # Erase the typing indicator, then print each line as its own bubble.
        print(f"\r{' ' * (len(label) + 3)}\r", end="")
        for i, line in enumerate(reply.splitlines()):
            if line.strip():
                print(f"{label if i == 0 else ' ' * len(label)}{line}")
        print()


if __name__ == "__main__":
    raise SystemExit(main())

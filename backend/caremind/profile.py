"""Conversational name handling: ask once, remember, and use the name sparingly.

The name is user-supplied text, so it is strictly sanitized before it is stored
or echoed back.
"""

from __future__ import annotations

import re

GREETINGS = {"hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening"}
SKIP_REPLIES = {
    "skip", "no", "nope", "no thanks", "no thank you", "none", "nothing", "anonymous", "pass",
    "never mind", "nevermind", "not now", "later",
}
_NOT_NAMES = {
    "yes", "yeah", "yep", "ok", "okay", "sure", "fine", "good", "great", "thanks", "thank", "help",
    "what", "why", "how", "who", "when", "where", "hello", "hi", "hey", "me", "you", "myself",
}
_INTRO = re.compile(
    r"^(?:(?:hi|hello|hey)[,!\s]+)?"
    r"(?:my name is|my name's|i am|i'm|im|this is|call me|you can call me|it's|its|name is|they call me)\s+(.+)$",
    re.IGNORECASE,
)
_WORD = re.compile(r"^[^\W\d_](?:[^\W\d_]|['\-])*$")
MAX_NAME_CHARS = 30
LEAD_INS = ("Sure, {name}.", "Good question, {name}.", "Happy to help, {name}.")
REUSE_EVERY_N_REPLIES = 3
PERSONALIZED_ROUTES = {"retrieve", "medical_education", "compare", "imaging"}


def normalize_message(message: str) -> str:
    return message.lower().strip().rstrip("!?.,").strip()


def is_greeting(message: str) -> bool:
    return normalize_message(message) in GREETINGS


def is_skip(message: str) -> bool:
    return normalize_message(message) in SKIP_REPLIES


def sanitize_name(raw: object) -> str | None:
    """Return a safe display name, or None if the value is not a plausible name."""
    if not isinstance(raw, str):
        return None
    words = raw.strip().strip("!?.,").split()
    if not 1 <= len(words) <= 2:
        return None
    if any(not _WORD.match(word) for word in words):
        return None
    name = " ".join(words)
    if len(name) > MAX_NAME_CHARS or name.lower() in _NOT_NAMES or any(w.lower() in _NOT_NAMES for w in words):
        return None
    if name.islower():
        name = " ".join(word[:1].upper() + word[1:] for word in words)
    return name


def extract_name(message: str) -> str | None:
    """Pull a name out of a reply to 'What should I call you?'."""
    text = message.strip()
    if not text or text.endswith("?"):
        return None
    match = _INTRO.match(text.rstrip("!.,"))
    return sanitize_name(match.group(1) if match else text)


def lead_in(name: str, reply_index: int) -> str:
    return LEAD_INS[reply_index % len(LEAD_INS)].format(name=name)


def should_use_name(assistant_reply_count: int, last_used_at: object) -> bool:
    if not isinstance(last_used_at, int):
        return True
    return assistant_reply_count - last_used_at >= REUSE_EVERY_N_REPLIES

"""Text normalisation applied identically at training and at inference time.

The cleaning is deliberately light. Two rules drove it:

1. Anything that changes the token distribution must run in the service too,
   otherwise the model sees a different distribution in production than it was
   trained on. Everything here is a pure function of the input string, so
   `src/api` calls exactly this module.
2. No lowercasing, stemming or stop-word removal. The TF-IDF vectoriser
   lowercases itself, and the transformer's tokeniser has its own scheme;
   doing it here would only destroy signal for one of them.

Numeric spans that look like card numbers, sort codes, amounts and dates are
masked. That is a privacy measure for real traffic, and it also stops the
models from memorising digit patterns that do not generalise.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

# Order matters: the most specific patterns must run first, otherwise the
# generic number rule eats their digits.
_CARD_NUMBER = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
_SORT_CODE = re.compile(r"\b\d{2}[- ]\d{2}[- ]\d{2}\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_URL = re.compile(r"https?://\S+|www\.\S+")
_AMOUNT = re.compile(
    r"(?:[$£€¥]\s?\d[\d,]*(?:\.\d+)?)"
    r"|(?:\b\d[\d,]*(?:\.\d+)?\s?(?:usd|gbp|eur|jpy|dollars?|pounds?|euros?)\b)",
    re.IGNORECASE,
)
_DATE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"
)
_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")

# C0/C1 control characters except tab/newline, plus the Unicode replacement char.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f�]")
_WHITESPACE = re.compile(r"\s+")

MASK_TOKENS = ("<CARDNUM>", "<SORTCODE>", "<IBAN>", "<EMAIL>", "<URL>", "<AMOUNT>", "<DATE>", "<NUM>")


def clean_text(text: str) -> str:
    """Normalise a single record. Pure function; safe to call per request."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # NFKC folds full-width and compatibility forms onto their ASCII equivalents.
    out = unicodedata.normalize("NFKC", text)
    out = _CONTROL.sub(" ", out)

    # Curly quotes and dashes -> ASCII, so "haven't" and "haven’t" are one token.
    out = out.translate(
        str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                       "–": "-", "—": "-", "…": "..."})
    )

    out = _URL.sub(" <URL> ", out)
    out = _EMAIL.sub(" <EMAIL> ", out)
    out = _IBAN.sub(" <IBAN> ", out)
    out = _SORT_CODE.sub(" <SORTCODE> ", out)
    out = _CARD_NUMBER.sub(" <CARDNUM> ", out)
    out = _AMOUNT.sub(" <AMOUNT> ", out)
    out = _DATE.sub(" <DATE> ", out)
    out = _NUMBER.sub(" <NUM> ", out)

    out = _WHITESPACE.sub(" ", out).strip()
    return out


def clean_many(texts: Iterable[str]) -> List[str]:
    return [clean_text(t) for t in texts]


def is_usable(text: str, min_chars: int = 2) -> bool:
    """Reject records that carry no signal after cleaning."""
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    # A record that is nothing but mask tokens tells the model nothing.
    residue = stripped
    for token in MASK_TOKENS:
        residue = residue.replace(token, "")
    return bool(residue.strip())

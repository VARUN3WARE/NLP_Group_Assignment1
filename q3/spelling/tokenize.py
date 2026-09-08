"""Token normalisation helpers and shared constants."""

from __future__ import annotations

import re
from typing import Iterable

BOS = "<s>"
EOS = "</s>"
NUM = "<num>"

_WORD_RE = re.compile(r"^[a-z]+(?:'[a-z]+)?$")
_HAS_DIGIT_RE = re.compile(r"\d")

ALPHABET = "abcdefghijklmnopqrstuvwxyz'"


def normalise(token: str) -> str | None:
    """Map a raw corpus token to its model form.

    Returns the lower-cased word, the ``<num>`` placeholder for numeric
    tokens, or ``None`` for pure punctuation (which is dropped).
    """
    token = token.lower()
    if _WORD_RE.match(token):
        return token
    if _HAS_DIGIT_RE.search(token):
        return NUM
    return None


def normalise_sentence(tokens: Iterable[str]) -> list[str]:
    """Normalise a sentence, dropping tokens that carry no lexical content."""
    out = []
    for tok in tokens:
        norm = normalise(tok)
        if norm is not None:
            out.append(norm)
    return out

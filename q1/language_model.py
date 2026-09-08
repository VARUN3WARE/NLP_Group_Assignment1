"""Shim: old `from language_model import ...` → `segpos.lm.trigram`."""

from segpos.lm.trigram import TrigramLanguageModel

__all__ = ["TrigramLanguageModel"]

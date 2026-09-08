"""Shim: old `from pos_tagger import ...` → `segpos.tagging.pos_tagger`."""

from segpos.tagging.pos_tagger import MorphologyAwarePOSTagger, TrigramPOSTagger

__all__ = ["TrigramPOSTagger", "MorphologyAwarePOSTagger"]

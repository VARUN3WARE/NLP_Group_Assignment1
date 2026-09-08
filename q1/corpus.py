"""Shim: old `from corpus import ...` → `segpos.data.corpus`."""

from segpos.data.corpus import load_brown, load_conllu, load_spanish, split_brown

__all__ = ["load_brown", "load_conllu", "load_spanish", "split_brown"]

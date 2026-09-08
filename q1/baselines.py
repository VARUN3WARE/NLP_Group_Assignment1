"""Shim: old `from baselines import ...` → `segpos.baselines.simple`."""

from segpos.baselines.simple import GreedyLongestMatchSegmenter, MostFrequentTagger

__all__ = ["GreedyLongestMatchSegmenter", "MostFrequentTagger"]

"""Baseline systems."""

from segpos.baselines.simple import GreedyLongestMatchSegmenter, MostFrequentTagger

__all__ = ["GreedyLongestMatchSegmenter", "MostFrequentTagger"]

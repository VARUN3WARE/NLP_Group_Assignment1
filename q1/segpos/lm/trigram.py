"""Trigram word language model with add-one smoothing."""

from __future__ import annotations

import math
import pickle
from collections import Counter
from pathlib import Path


class TrigramLanguageModel:
    def __init__(self):
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self.vocabulary = set()
        self.total_words = 0

    def train(self, sentences):
        """
        Train unigram, bigram, and trigram counts.

        sentences: list of sentences; each sentence is a list of words.
        """
        for sentence in sentences:
            words = ["<START>", "<START>"] + [w.lower() for w in sentence] + ["<END>"]

            for word in sentence:
                word = word.lower()
                self.unigram_counts[word] += 1
                self.vocabulary.add(word)
                self.total_words += 1

            for i in range(2, len(words)):
                w1, w2, w3 = words[i - 2], words[i - 1], words[i]
                self.bigram_counts[(w1, w2)] += 1
                self.trigram_counts[(w1, w2, w3)] += 1

    def unigram_probability(self, word):
        if self.total_words == 0:
            return 0.0
        return self.unigram_counts[word] / self.total_words

    def trigram_probability(self, w1, w2, w3):
        """P(w3 | w1, w2) with add-one smoothing."""
        numerator = self.trigram_counts[(w1, w2, w3)] + 1
        denominator = self.bigram_counts[(w1, w2)] + len(self.vocabulary)
        return numerator / denominator

    def log_trigram_probability(self, w1, w2, w3):
        return math.log(self.trigram_probability(w1, w2, w3))

    def score_sentence(self, sentence):
        """Log probability of a complete sentence (with boundary markers)."""
        words = ["<START>", "<START>"] + list(sentence) + ["<END>"]
        score = 0.0
        for i in range(2, len(words)):
            score += self.log_trigram_probability(words[i - 2], words[i - 1], words[i])
        return score

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def __repr__(self):
        return (
            f"TrigramLanguageModel(vocab={len(self.vocabulary):,}, "
            f"tokens={self.total_words:,})"
        )

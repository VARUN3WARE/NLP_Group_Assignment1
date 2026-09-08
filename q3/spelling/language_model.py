"""Vocabulary + unigram / bigram language model."""

from __future__ import annotations

import math
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from spelling.tokenize import BOS, EOS, NUM, normalise_sentence


@dataclass
class LanguageModel:
    """Vocabulary + unigram frequencies + add-k smoothed bigram model."""

    unigram_counts: Counter = field(default_factory=Counter)
    bigram_counts: Counter = field(default_factory=Counter)
    context_counts: Counter = field(default_factory=Counter)
    k: float = 0.01
    lam: float = 0.85
    unigram_k: float = 0.5

    total_tokens: int = 0
    vocabulary: frozenset = frozenset()

    @classmethod
    def train(cls, sentences: Iterable[Sequence[str]], k: float = 0.01) -> "LanguageModel":
        """Train the unigram + bigram models from an iterable of sentences.

        Each sentence is a sequence of raw (untokenised-case) word strings.
        """
        model = cls(k=k)
        unigrams = model.unigram_counts
        bigrams = model.bigram_counts
        contexts = model.context_counts

        for raw_sent in sentences:
            sent = normalise_sentence(raw_sent)
            if not sent:
                continue
            unigrams.update(sent)
            padded = [BOS] + sent + [EOS]
            for left, right in zip(padded, padded[1:]):
                bigrams[(left, right)] += 1
                contexts[left] += 1

        model.total_tokens = sum(unigrams.values())
        model.vocabulary = frozenset(w for w in unigrams if w != NUM)
        return model

    def unigram_prob(self, word: str) -> float:
        """P(word) = C(word) / N.  Unknown words get 0."""
        return self.unigram_counts.get(word, 0) / self.total_tokens

    def log_unigram_prob(self, word: str, k: float | None = None) -> float:
        """Add-k smoothed log P(word); never -inf so it is safe to sum."""
        k = self.k if k is None else k
        v = len(self.vocabulary) + 1
        return math.log(
            (self.unigram_counts.get(word, 0) + k) / (self.total_tokens + k * v)
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary) + 2

    def bigram_prob(self, prev: str, word: str) -> float:
        """Add-k smoothed P(word | prev)."""
        numer = self.bigram_counts.get((prev, word), 0) + self.k
        denom = self.context_counts.get(prev, 0) + self.k * self.vocab_size
        return numer / denom

    def log_bigram_prob(self, prev: str, word: str) -> float:
        return math.log(self.bigram_prob(prev, word))

    def bigram_prob_interp(self, prev: str, word: str) -> float:
        """P(word | prev) interpolated with the unigram distribution."""
        p_uni = ((self.unigram_counts.get(word, 0) + self.unigram_k)
                 / (self.total_tokens + self.unigram_k * self.vocab_size))
        context = self.context_counts.get(prev, 0)
        p_ml = self.bigram_counts.get((prev, word), 0) / context if context else 0.0
        return self.lam * p_ml + (1.0 - self.lam) * p_uni

    def log_bigram_interp(self, prev: str, word: str) -> float:
        return math.log(self.bigram_prob_interp(prev, word))

    def sentence_log_prob(self, tokens: Sequence[str]) -> float:
        """Log P(sentence) under the bigram model, with <s>/</s> padding."""
        padded = [BOS] + list(tokens) + [EOS]
        return sum(
            self.log_bigram_prob(left, right)
            for left, right in zip(padded, padded[1:])
        )

    def perplexity(self, tokens: Sequence[str]) -> float:
        n = len(tokens) + 1
        if n == 0:
            return float("inf")
        return math.exp(-self.sentence_log_prob(tokens) / n)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "LanguageModel":
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def __repr__(self) -> str:
        return (
            f"LanguageModel(types={len(self.vocabulary):,}, "
            f"tokens={self.total_tokens:,}, bigrams={len(self.bigram_counts):,}, "
            f"k={self.k})"
        )



def brown_sentences() -> list[list[str]]:
    """All Brown corpus sentences as lists of raw tokens."""
    from nltk.corpus import brown

    return [list(sent) for sent in brown.sents()]


def train_test_split(sentences: Sequence[Sequence[str]], test_ratio: float = 0.10,
                     seed: int = 42):
    """Hold out `test_ratio` of the sentences for the Part 4 evaluation.

    The corrector is built from the *training* portion only, so the accuracy
    numbers in Part 4 are not inflated by having memorised the test sentences.
    """
    import random

    idx = list(range(len(sentences)))
    random.Random(seed).shuffle(idx)
    n_test = int(round(len(idx) * test_ratio))
    test_idx = set(idx[:n_test])
    train = [s for i, s in enumerate(sentences) if i not in test_idx]
    test = [sentences[i] for i in idx[:n_test]]
    return train, test



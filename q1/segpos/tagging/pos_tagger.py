"""Trigram HMM POS tagger and morphology-aware Spanish extension."""

from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path


class TrigramPOSTagger:
    def __init__(self):
        self.word_tag_counts = Counter()
        self.tag_counts = Counter()
        self.tag_bigram_counts = Counter()
        self.tag_trigram_counts = Counter()
        self.tags = set()
        self.vocabulary = set()
        self.word_to_tags = defaultdict(set)

    def train(self, sentences):
        """Train on sentences of (word, tag) tuples."""
        for sentence in sentences:
            for word, tag in sentence:
                word = word.lower()
                self.word_tag_counts[(word, tag)] += 1
                self.word_to_tags[word].add(tag)
                self.tag_counts[tag] += 1
                self.tags.add(tag)
                self.vocabulary.add(word)

            tags = ["<START>", "<START>"] + [tag for _word, tag in sentence]
            for i in range(2, len(tags)):
                self.tag_bigram_counts[(tags[i - 2], tags[i - 1])] += 1
                self.tag_trigram_counts[(tags[i - 2], tags[i - 1], tags[i])] += 1

    def emission_probability(self, word, tag):
        word = word.lower()
        numerator = self.word_tag_counts[(word, tag)] + 1
        denominator = self.tag_counts[tag] + len(self.vocabulary)
        return numerator / denominator

    def log_emission_probability(self, word, tag):
        return math.log(self.emission_probability(word, tag))

    def transition_probability(self, tag1, tag2, tag3):
        numerator = self.tag_trigram_counts[(tag1, tag2, tag3)] + 1
        denominator = self.tag_bigram_counts[(tag1, tag2)] + len(self.tags)
        return numerator / denominator

    def log_transition_probability(self, tag1, tag2, tag3):
        return math.log(self.transition_probability(tag1, tag2, tag3))

    def possible_tags(self, word):
        word = word.lower()
        tags = self.word_to_tags.get(word)
        if not tags:
            return self.tags
        return tags

    def tag_score(self, words):
        """
        Return (tagged pairs, log-score) for the Viterbi path.

        Useful for joint beam decoding that mixes LM and POS scores.
        """
        if not words:
            return [], 0.0

        dp = {("<START>", "<START>"): (0.0, None, None)}
        backpointers = []

        for word in words:
            new_dp = {}
            for (prev_prev_tag, prev_tag), (score, _, _) in dp.items():
                for tag in self.possible_tags(word):
                    new_score = (
                        score
                        + self.log_transition_probability(prev_prev_tag, prev_tag, tag)
                        + self.log_emission_probability(word, tag)
                    )
                    new_state = (prev_tag, tag)
                    if new_state not in new_dp or new_score > new_dp[new_state][0]:
                        new_dp[new_state] = (new_score, (prev_prev_tag, prev_tag), tag)
            dp = new_dp
            backpointers.append(dp)

        best_state = max(dp, key=lambda state: dp[state][0])
        best_score = dp[best_state][0]
        predicted_tags = []

        for i in range(len(words) - 1, -1, -1):
            _score, previous_state, tag = backpointers[i][best_state]
            predicted_tags.append(tag)
            best_state = previous_state

        predicted_tags.reverse()
        return list(zip(words, predicted_tags)), best_score

    def tag(self, words):
        tagged, _score = self.tag_score(words)
        return tagged

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)


class MorphologyAwarePOSTagger(TrigramPOSTagger):
    """POS tags extended with gender/number, e.g. NOUN-Fem-Sg."""

    @staticmethod
    def make_morph_tag(upos, feats):
        tag = upos
        gender = feats.get("Gender")
        number = feats.get("Number")

        if gender in {"Masc", "Fem", "Neut"}:
            tag += f"-{gender}"

        if number in {"Sing", "Plur"}:
            tag += "-Sg" if number == "Sing" else "-Pl"

        return tag

    def train_spanish(self, sentences):
        converted = []
        for sentence in sentences:
            converted.append(
                [
                    (token["form"], self.make_morph_tag(token["upos"], token["feats"]))
                    for token in sentence
                ]
            )
        self.train(converted)
        return converted

"""Candidate generation: Method A (edit-distance-1) and Method B (SymSpell)."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Set

from spelling.tokenize import ALPHABET


def deletes1(word: str) -> Set[str]:
    """Every string obtained by deleting exactly one character of `word`."""
    return {word[:i] + word[i + 1:] for i in range(len(word))}


def within_edit_distance_1(a: str, b: str) -> bool:
    """True iff `a` and `b` are within Damerau-Levenshtein distance 1.

    Written as an explicit length-case analysis rather than a full DP table:
    at distance <= 1 only four shapes are possible, and each can be checked in
    a single linear scan.  This is the verification step that turns SymSpell's
    *approximate* bucket lookup into an exact edit-distance-1 result set.
    """
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False

    if la == lb:
        diffs = [i for i in range(la) if a[i] != b[i]]
        if len(diffs) == 1:
            return True
        if len(diffs) == 2:
            i, j = diffs
            return j == i + 1 and a[i] == b[j] and a[j] == b[i]
        return False

    short, long = (a, b) if la < lb else (b, a)
    i = j = 0
    skipped = False
    while i < len(short) and j < len(long):
        if short[i] == long[j]:
            i += 1
            j += 1
        elif skipped:
            return False
        else:
            skipped = True
            j += 1
    return True


class EditDistance1Generator:
    """Method A: enumerate all edit-distance-1 strings, keep the real words."""

    def __init__(self, vocabulary: Iterable[str], alphabet: str = ALPHABET):
        self.vocabulary = frozenset(vocabulary)
        self.alphabet = alphabet

    def edits1(self, word: str) -> Set[str]:
        """All strings one edit away from `word` (not filtered by vocabulary).

        For a word of length n this produces
            n deletions + (n-1) transpositions + n*|A| replacements
            + (n+1)*|A| insertions
        candidate strings, i.e. O(n * |alphabet|).
        """
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = {L + R[1:] for L, R in splits if R}
        transposes = {L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1}
        replaces = {L + c + R[1:] for L, R in splits if R for c in self.alphabet}
        inserts = {L + c + R for L, R in splits for c in self.alphabet}
        return deletes | transposes | replaces | inserts

    def candidates(self, word: str) -> Set[str]:
        """Vocabulary words within one edit of `word` (excluding `word`)."""
        return {w for w in self.edits1(word) if w in self.vocabulary} - {word}

    def __repr__(self) -> str:
        return f"EditDistance1Generator(|V|={len(self.vocabulary):,})"


class SymmetricDeleteIndex:
    """Method B: SymSpell.  All the work happens once, at build time.

    The index maps a *deletion variant* to the vocabulary words that produce
    it.  Every word is also indexed under itself, which is what lets the
    lookup catch the "the query has one extra character" case.

    Why one deletion on each side is enough for edit distance 1:
      * substitution  - delete the changed position on both sides -> equal
      * transposition - delete one of the two swapped chars on each side
      * insertion in the query  - the query's deletion equals the word
      * deletion in the query   - the query equals one of the word's deletions
    """

    def __init__(self, vocabulary: Iterable[str], verify: bool = True):
        self.vocabulary = frozenset(vocabulary)
        self.verify = verify
        self.index: dict[str, list[str]] = self._build_index(self.vocabulary)

    @staticmethod
    def _build_index(vocabulary: Iterable[str]) -> dict[str, list[str]]:
        index: dict[str, list[str]] = defaultdict(list)
        for word in vocabulary:
            index[word].append(word)
            for variant in deletes1(word):
                index[variant].append(word)
        return dict(index)

    def candidates(self, word: str) -> Set[str]:
        """Vocabulary words within one edit of `word` (excluding `word`)."""
        index = self.index
        found: Set[str] = set()

        for probe in (word, *deletes1(word)):
            hits = index.get(probe)
            if hits:
                found.update(hits)

        found.discard(word)
        if self.verify:
            found = {w for w in found if within_edit_distance_1(word, w)}
        return found

    @property
    def n_entries(self) -> int:
        """Number of distinct keys stored in the pre-computed dictionary."""
        return len(self.index)

    @property
    def n_postings(self) -> int:
        """Total (key -> word) pairs, i.e. the real memory cost of Method B."""
        return sum(len(v) for v in self.index.values())

    def __repr__(self) -> str:
        return (
            f"SymmetricDeleteIndex(|V|={len(self.vocabulary):,}, "
            f"keys={self.n_entries:,}, postings={self.n_postings:,}, "
            f"verify={self.verify})"
        )



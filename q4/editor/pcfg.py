"""PCFG induction from Penn Treebank + Viterbi most-probable parse."""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import nltk
from nltk import induce_pcfg
from nltk.corpus import treebank
from nltk.parse import ViterbiParser
from nltk.grammar import Nonterminal, ProbabilisticProduction

from editor.paths import ARTIFACTS_DIR, PCFG_PATH
from editor.tagset import brown_to_ptb, map_tagged_sentence


@dataclass
class ParseResult:
    ok: bool
    log_prob: float | None
    tree_str: str | None
    status: str  # "ok" | "unparseable"


class PCFGParser:
    """
    Induced PCFG over PTB trees + NLTK ViterbiParser (CKY chart / Viterbi).

    Sentences that fail to parse return status='unparseable' without raising.
    """

    def __init__(self, grammar, parser: ViterbiParser):
        self.grammar = grammar
        self.parser = parser
        self.lexicon = {
            prod.rhs()[0]
            for prod in grammar.productions()
            if getattr(prod, "is_lexical", lambda: False)() or (
                len(prod.rhs()) == 1 and isinstance(prod.rhs()[0], str)
            )
        }

    @classmethod
    def train(cls, max_trees: int | None = None) -> "PCFGParser":
        nltk.download("treebank", quiet=True)
        productions = []
        trees = treebank.parsed_sents()
        if max_trees is not None:
            trees = trees[:max_trees]
        for tree in trees:
            t = tree.copy(deep=True)
            t.collapse_unary(collapsePOS=False, collapseRoot=False)
            t.chomsky_normal_form(factor="horizontal", horzMarkov=2)
            productions += t.productions()

        S = Nonterminal("S")
        grammar = induce_pcfg(S, productions)
        parser = ViterbiParser(grammar)
        return cls(grammar, parser)

    def _project_to_lexicon(self, words: list[str]) -> list[str]:
        """Map PTB-OOV tokens to a known noun so CKY can still run."""
        return [w if w in self.lexicon else "something" for w in words]

    def parse_tokens(self, words: list[str], tags: list[str] | None = None) -> ParseResult:
        if not words:
            return ParseResult(False, None, None, "unparseable")

        tokens = [w.lower() for w in words]
        # Long literary sentences explode the CKY chart; mark unparseable quickly.
        if len(tokens) > 25:
            return ParseResult(False, None, None, "unparseable")

        projected = self._project_to_lexicon(tokens)
        try:
            parses = list(self.parser.parse(projected))
        except Exception:
            return ParseResult(False, None, None, "unparseable")

        if not parses:
            return ParseResult(False, None, None, "unparseable")

        best = parses[0]
        try:
            log_prob = math.log(best.prob()) if best.prob() > 0 else float("-inf")
        except Exception:
            log_prob = None
        # Note OOV projection in status text when applicable
        status = "ok"
        if projected != tokens:
            status = "ok-projected"
        return ParseResult(True, log_prob, str(best), status)

    def parse_tagged_brown(
        self, pairs: list[tuple[str, str]]
    ) -> ParseResult:
        """Map Brown tags to PTB then parse words (tags used for reconciliation demo)."""
        mapped = map_tagged_sentence(pairs)
        words = [w for w, _ in mapped]
        _ptb_tags = [t for _, t in mapped]
        return self.parse_tokens(words, tags=_ptb_tags)

    def save(self, path: Path = PCFG_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self.grammar, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path: Path = PCFG_PATH) -> "PCFGParser":
        with open(path, "rb") as fh:
            grammar = pickle.load(fh)
        return cls(grammar, ViterbiParser(grammar))


def load_or_train_pcfg(max_trees: int | None = 2000) -> PCFGParser:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if PCFG_PATH.exists():
        try:
            return PCFGParser.load(PCFG_PATH)
        except Exception:
            pass
    parser = PCFGParser.train(max_trees=max_trees)
    parser.save(PCFG_PATH)
    return parser

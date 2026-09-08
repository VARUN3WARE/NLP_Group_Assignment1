"""Random passage sampling and simulated spacebar-merge corruption."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

import nltk

from editor.config import DEFAULT_CONFIG, EditorConfig

_SENT_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class PassageToken:
    """One streamed token, possibly a merge of two gold words."""

    text: str
    gold_words: list[str]
    is_merge: bool
    sentence_index: int


@dataclass
class SampledPassage:
    file_id: str
    sentences: list[list[str]]
    tokens: list[PassageToken]
    merge_probability: float
    seed: int | None = None

    @property
    def plain_text(self) -> str:
        return " ".join(t.text for t in self.tokens)

    @property
    def gold_text(self) -> str:
        words = [w for sent in self.sentences for w in sent]
        return " ".join(words)


def _tokenize_sentence(raw: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", raw)]


def _sentences_from_gutenberg(file_id: str) -> list[list[str]]:
    raw = nltk.corpus.gutenberg.raw(file_id)
    chunks = _SENT_END.split(raw)
    out = []
    for chunk in chunks:
        words = _tokenize_sentence(chunk)
        if len(words) >= 4:
            out.append(words)
    return out


def sample_passage(
    config: EditorConfig = DEFAULT_CONFIG,
    seed: int | None = None,
    source: str = "gutenberg",
) -> SampledPassage:
    """
    Sample 5–8 contiguous sentences and optionally merge adjacent words.

    `seed` is only for debugging; omit for a different passage each run.
    """
    rng = random.Random(seed)

    if source == "gutenberg":
        nltk.download("gutenberg", quiet=True)
        file_ids = list(nltk.corpus.gutenberg.fileids())
        file_id = rng.choice(file_ids)
        sentences_all = _sentences_from_gutenberg(file_id)
    else:
        nltk.download("brown", quiet=True)
        file_id = "brown"
        from nltk.corpus import brown

        sentences_all = [
            [w for w in sent if re.match(r"[A-Za-z]+(?:'[A-Za-z]+)?$", w)]
            for sent in brown.sents()
        ]
        sentences_all = [s for s in sentences_all if len(s) >= 4]

    if len(sentences_all) < config.min_sentences:
        raise RuntimeError(f"Not enough sentences in {file_id}")

    n = rng.randint(config.min_sentences, config.max_sentences)
    start = rng.randint(0, max(0, len(sentences_all) - n))
    sentences = sentences_all[start : start + n]

    tokens: list[PassageToken] = []
    for si, sent in enumerate(sentences):
        i = 0
        while i < len(sent):
            if (
                i + 1 < len(sent)
                and rng.random() < config.merge_probability
            ):
                merged = (sent[i] + sent[i + 1]).lower()
                tokens.append(
                    PassageToken(
                        text=merged,
                        gold_words=[sent[i].lower(), sent[i + 1].lower()],
                        is_merge=True,
                        sentence_index=si,
                    )
                )
                i += 2
            else:
                tokens.append(
                    PassageToken(
                        text=sent[i].lower(),
                        gold_words=[sent[i].lower()],
                        is_merge=False,
                        sentence_index=si,
                    )
                )
                i += 1

    return SampledPassage(
        file_id=file_id,
        sentences=[[w.lower() for w in s] for s in sentences],
        tokens=tokens,
        merge_probability=config.merge_probability,
        seed=seed,
    )

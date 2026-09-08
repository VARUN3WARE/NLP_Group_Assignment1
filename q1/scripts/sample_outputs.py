#!/usr/bin/env python3
"""Legacy-style sample outputs (trains on the fly). Prefer: python main.py sample."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segpos.data.corpus import load_brown, load_spanish, split_brown
from segpos.lm.trigram import TrigramLanguageModel
from segpos.segmentation.viterbi import ViterbiSegmenter
from segpos.tagging.pos_tagger import MorphologyAwarePOSTagger, TrigramPOSTagger


def main() -> None:
    brown = load_brown()
    english_train, _ = split_brown(brown)

    english_lm = TrigramLanguageModel()
    english_lm.train([[word for word, _tag in sentence] for sentence in english_train])
    english_segmenter = ViterbiSegmenter(english_lm)
    english_tagger = TrigramPOSTagger()
    english_tagger.train(english_train)

    print("\n=== ENGLISH ===")
    for sample in (
        "thequickbrownfoxjumpsoverthelazydog",
        "tobeornottobethatisthequestion",
    ):
        segmented = english_segmenter.segment(sample)
        print("\nInput:", sample)
        print("Segmentation:", segmented)
        print("POS:", english_tagger.tag(segmented))

    spanish_train, _, _ = load_spanish()
    spanish_lm = TrigramLanguageModel()
    spanish_lm.train([[token["form"] for token in sentence] for sentence in spanish_train])
    spanish_segmenter = ViterbiSegmenter(spanish_lm)
    spanish_tagger = TrigramPOSTagger()
    spanish_tagger.train(
        [[(token["form"], token["upos"]) for token in sentence] for sentence in spanish_train]
    )
    spanish_morph_tagger = MorphologyAwarePOSTagger()
    spanish_morph_tagger.train_spanish(spanish_train)

    print("\n=== SPANISH ===")
    for sample in (
        "mispadrespuedenviajar",
        "elcielodespejadoesazul",
        "maríaleeunlibro",
    ):
        segmented = spanish_segmenter.segment(sample)
        print("\nInput:", sample)
        print("Segmentation:", segmented)
        print("POS:", spanish_tagger.tag(segmented))
        print("Morphology-aware POS:", spanish_morph_tagger.tag(segmented))


if __name__ == "__main__":
    main()

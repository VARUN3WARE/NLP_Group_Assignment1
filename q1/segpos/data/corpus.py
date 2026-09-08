"""Brown and Universal Dependencies Spanish-GSD loaders."""

from __future__ import annotations

from pathlib import Path

import nltk

from segpos.paths import DEFAULT_SPANISH_DIR


def load_brown():
    """Load Brown tagged sentences as lists of (word, tag)."""
    nltk.download("brown", quiet=True)
    from nltk.corpus import brown

    return brown.tagged_sents()


def split_brown(sentences, train_ratio=0.8):
    """80/20 train/test split required for English."""
    split_index = int(len(sentences) * train_ratio)
    return sentences[:split_index], sentences[split_index:]


def load_conllu(filepath):
    """
    Parse a CoNLL-U file into sentences of token dicts:

        form, lemma, upos, feats
    """
    sentences = []
    current_sentence = []

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            if line.startswith("#"):
                continue

            columns = line.split("\t")
            if len(columns) != 10:
                continue

            token_id = columns[0]
            if "-" in token_id or "." in token_id:
                continue

            feats = {}
            feats_string = columns[5]
            if feats_string != "_":
                for feature in feats_string.split("|"):
                    key, value = feature.split("=")
                    feats[key] = value

            current_sentence.append(
                {
                    "form": columns[1],
                    "lemma": columns[2],
                    "upos": columns[3],
                    "feats": feats,
                }
            )

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def load_spanish(data_directory=None):
    """
    Load Spanish-GSD train/dev/test splits.

    Clone into q1/data/spanish/:

        git clone https://github.com/UniversalDependencies/UD_Spanish-GSD.git \\
            data/spanish/UD_Spanish-GSD
    """
    data_directory = Path(data_directory or DEFAULT_SPANISH_DIR)

    train_file = data_directory / "es_gsd-ud-train.conllu"
    dev_file = data_directory / "es_gsd-ud-dev.conllu"
    test_file = data_directory / "es_gsd-ud-test.conllu"

    for path in (train_file, dev_file, test_file):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Clone UD_Spanish-GSD into {data_directory}"
            )

    return load_conllu(train_file), load_conllu(dev_file), load_conllu(test_file)

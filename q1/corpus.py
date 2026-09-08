import nltk
from pathlib import Path


def load_brown():
    """
    Load the Brown corpus with POS tags.

    Returns:
        List of sentences.
        Each sentence is a list of (word, tag) tuples.
    """
    nltk.download("brown", quiet=True)

    from nltk.corpus import brown

    return brown.tagged_sents()


def split_brown(sentences, train_ratio=0.8):
    """
    Split Brown corpus into training and test sets.

    The assignment requires an 80/20 split for English.
    """
    split_index = int(len(sentences) * train_ratio)

    train_sentences = sentences[:split_index]
    test_sentences = sentences[split_index:]

    return train_sentences, test_sentences


def load_conllu(filepath):
    """
    Load a CoNLL-U file.

    Returns:
        List of sentences.
        Each word is represented as a dictionary containing:
            form
            lemma
            upos
            feats
    """

    sentences = []
    current_sentence = []

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:

            line = line.strip()

            # Blank line = end of sentence
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []

                continue

            # Comments
            if line.startswith("#"):
                continue

            columns = line.split("\t")

            # A valid CoNLL-U token has 10 columns
            if len(columns) != 10:
                continue

            token_id = columns[0]

            # Ignore multi-word tokens such as 1-2
            if "-" in token_id or "." in token_id:
                continue

            word = columns[1]
            lemma = columns[2]
            upos = columns[3]
            feats_string = columns[5]

            # Convert:
            # Gender=Fem|Number=Sing
            #
            # into:
            # {"Gender": "Fem", "Number": "Sing"}
            feats = {}

            if feats_string != "_":
                for feature in feats_string.split("|"):
                    key, value = feature.split("=")
                    feats[key] = value

            current_sentence.append({
                "form": word,
                "lemma": lemma,
                "upos": upos,
                "feats": feats
            })

    # Handle final sentence if file doesn't end with blank line
    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def load_spanish(data_directory="data/spanish/UD_Spanish-GSD"):
    """
    Load Spanish-GSD train/dev/test files.

    Expected layout (clone into q1/data/spanish/):

        git clone https://github.com/UniversalDependencies/UD_Spanish-GSD.git \\
            data/spanish/UD_Spanish-GSD
    """

    data_directory = Path(data_directory)

    train_file = data_directory / "es_gsd-ud-train.conllu"
    dev_file = data_directory / "es_gsd-ud-dev.conllu"
    test_file = data_directory / "es_gsd-ud-test.conllu"

    for path in (train_file, dev_file, test_file):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Clone UD_Spanish-GSD into {data_directory}"
            )

    train = load_conllu(train_file)
    dev = load_conllu(dev_file)
    test = load_conllu(test_file)

    return train, dev, test

if __name__ == "__main__":

    print("Loading Brown corpus...")

    brown_sentences = load_brown()

    train, test = split_brown(brown_sentences)

    print(f"Brown sentences: {len(brown_sentences)}")
    print(f"Brown training:  {len(train)}")
    print(f"Brown testing:   {len(test)}")

    print("\nFirst Brown sentence:")
    print(train[0])

    print("\nLoading Spanish-GSD...")

    spanish_train, spanish_dev, spanish_test = load_spanish()

    print(f"Spanish training: {len(spanish_train)}")
    print(f"Spanish dev:      {len(spanish_dev)}")
    print(f"Spanish testing:  {len(spanish_test)}")

    print("\nFirst Spanish sentence:")

    for token in spanish_train[0]:
        print(token)

from collections import Counter, defaultdict


class GreedyLongestMatchSegmenter:
    """
    Baseline segmentation model.

    At each character position, choose the longest
    vocabulary word that matches.
    """

    def __init__(self, vocabulary, max_word_length=20):
        self.vocabulary = vocabulary
        self.max_word_length = max_word_length

    def segment(self, text):
        """
        Segment text using greedy longest-match.
        """

        text = text.lower()

        words = []
        position = 0

        while position < len(text):

            best_word = None

            max_length = min(
                self.max_word_length,
                len(text) - position
            )

            # Try longest words first.
            for length in range(max_length, 0, -1):

                candidate = text[
                    position:position + length
                ]

                if candidate in self.vocabulary:
                    best_word = candidate
                    break

            # No vocabulary word found.
            if best_word is None:
                # Treat the current character as an
                # unknown token so the algorithm can continue.
                best_word = text[position]

            words.append(best_word)

            position += len(best_word)

        return words


class MostFrequentTagger:
    """
    Baseline POS tagger.

    Assign each word the tag it appeared with most
    frequently during training.
    """

    def __init__(self):
        self.word_tag_counts = defaultdict(Counter)
        self.word_most_frequent_tag = {}

    def train(self, sentences):
        """
        Train on tagged sentences.

        sentences:
            List of sentences containing
            (word, tag) tuples.
        """

        for sentence in sentences:

            for word, tag in sentence:

                word = word.lower()

                self.word_tag_counts[word][tag] += 1

        # Determine the most frequent tag for each word.
        for word, tag_counts in self.word_tag_counts.items():

            self.word_most_frequent_tag[word] = (
                tag_counts.most_common(1)[0][0]
            )

    def tag(self, words):
        """
        Tag each word with its most frequent training tag.

        Unknown words receive UNKNOWN.
        """

        result = []

        for word in words:

            normalized_word = word.lower()

            tag = self.word_most_frequent_tag.get(
                normalized_word,
                "UNKNOWN"
            )

            result.append(
                (word, tag)
            )

        return result


if __name__ == "__main__":

    from corpus import load_brown, split_brown

    print("Loading Brown corpus...")

    sentences = load_brown()
    train_sentences, _ = split_brown(sentences)

    # Build vocabulary.
    vocabulary = {
        word.lower()
        for sentence in train_sentences
        for word, tag in sentence
    }

    segmenter = GreedyLongestMatchSegmenter(
        vocabulary=vocabulary,
        max_word_length=20
    )

    text = "thequickbrownfox"

    print("\nInput:")
    print(text)

    print("\nGreedy segmentation:")
    print(segmenter.segment(text))

    print("\nTraining most-frequent-tag baseline...")

    tagger = MostFrequentTagger()
    tagger.train(train_sentences)

    words = [
        "the",
        "quick",
        "brown",
        "fox"
    ]

    print("\nInput:")
    print(words)

    print("\nMost-frequent tags:")

    for word, tag in tagger.tag(words):
        print(f"{word} -> {tag}")

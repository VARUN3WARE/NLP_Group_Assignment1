from collections import Counter, defaultdict
from segmentation import ViterbiSegmenter
from language_model import TrigramLanguageModel


def pos_accuracy(gold_sentences, predicted_sentences):
    """
    Calculate POS tagging accuracy.

    gold_sentences:
        List of sentences containing (word, gold_tag).

    predicted_sentences:
        List of sentences containing (word, predicted_tag).
    """

    correct = 0
    total = 0

    for gold_sentence, predicted_sentence in zip(
        gold_sentences,
        predicted_sentences
    ):

        for (_, gold_tag), (_, predicted_tag) in zip(
            gold_sentence,
            predicted_sentence
        ):

            if gold_tag == predicted_tag:
                correct += 1

            total += 1

    if total == 0:
        return 0.0

    return correct / total


def confusion_matrix(gold_sentences, predicted_sentences):
    """
    Build a POS confusion matrix.

    Returns:
        Dictionary mapping:
            actual_tag -> predicted_tag -> count
    """

    matrix = defaultdict(Counter)

    for gold_sentence, predicted_sentence in zip(
        gold_sentences,
        predicted_sentences
    ):

        for (_, gold_tag), (_, predicted_tag) in zip(
            gold_sentence,
            predicted_sentence
        ):

            matrix[gold_tag][predicted_tag] += 1

    return matrix

"""
def print_confusion_matrix(matrix):
    
    Print the confusion matrix in a readable format.
    

    tags = sorted(matrix.keys())

    # Include predicted-only tags.
    predicted_tags = {
        tag
        for row in matrix.values()
        for tag in row
    }

    tags = sorted(set(tags) | predicted_tags)

    print("\nConfusion Matrix")
    print("----------------")

    print(
        f"{'Actual':<15}",
        end=""
    )

    for tag in tags:
        print(f"{tag:<12}", end="")

    print()

    for actual_tag in tags:

        print(
            f"{actual_tag:<15}",
            end=""
        )

        for predicted_tag in tags:

            count = matrix[actual_tag][predicted_tag]

            print(
                f"{count:<12}",
                end=""
            )

        print()
"""
def print_tiny_confusion_matrix(matrix, top_n=15):
    """
    Print a compact confusion matrix using the most frequent tags.
    """

    tag_counts = Counter()

    for actual_tag, row in matrix.items():
        tag_counts[actual_tag] += sum(row.values())

    tags = [
        tag for tag, _ in tag_counts.most_common(top_n)
    ]

    print("\nConfusion Matrix")
    print("----------------")

    header = "Actual \\ Predicted"
    print(f"{header:<20}", end="")

    for tag in tags:
        print(f"{tag:<10}", end="")

    print()

    for actual_tag in tags:
        print(f"{actual_tag:<20}", end="")

        for predicted_tag in tags:
            print(
                f"{matrix[actual_tag][predicted_tag]:<10}",
                end=""
            )

        print()

def evaluate_error_sources(gold_sentences, segmenter, tagger):
    """
    Separate POS errors caused by segmentation from
    genuine POS tagging errors.
    """

    segmentation_caused_errors = 0
    genuine_tagging_errors = 0
    total_words = 0

    for sentence in gold_sentences:

        gold_words = [
            word.lower()
            for word, tag in sentence
        ]

        gold_tags = [
            tag
            for word, tag in sentence
        ]

        total_words += len(gold_words)

        # Create segmentation input.
        text = "".join(gold_words)

        # Predict segmentation.
        predicted_words = segmenter.segment(text)

        # Tag predicted words.
        predicted_tagged = tagger.tag(predicted_words)

        # Convert both segmentations into character spans.
        gold_spans = []
        position = 0

        for word, tag in zip(gold_words, gold_tags):
            start = position
            end = position + len(word)
            gold_spans.append((start, end, word, tag))
            position = end

        predicted_spans = []
        position = 0

        for word, tag in predicted_tagged:
            start = position
            end = position + len(word)
            predicted_spans.append(
                (start, end, word.lower(), tag)
            )
            position = end

        # Compare gold words against predicted words
        # using their character spans.
        for gold_start, gold_end, gold_word, gold_tag in gold_spans:

            matching = [
                (word, tag)
                for start, end, word, tag
                in predicted_spans
                if start == gold_start and end == gold_end
            ]

            if len(matching) == 1:
                predicted_word, predicted_tag = matching[0]

                if predicted_tag != gold_tag:
                    genuine_tagging_errors += 1

            else:
                segmentation_caused_errors += 1

    total_errors = (
        segmentation_caused_errors +
        genuine_tagging_errors
    )

    print("\nError Source Analysis")
    print("---------------------")
    print(f"Total gold words: {total_words}")
    print(
        f"Segmentation-caused errors: "
        f"{segmentation_caused_errors}"
    )
    print(
        f"Genuine tagging errors: "
        f"{genuine_tagging_errors}"
    )

    if total_errors > 0:
        print(
            f"Segmentation-caused: "
            f"{segmentation_caused_errors / total_errors:.2%}"
        )
        print(
            f"Genuine tagging: "
            f"{genuine_tagging_errors / total_errors:.2%}"
        )

if __name__ == "__main__":

    from corpus import load_brown, split_brown
    from pos_tagger import TrigramPOSTagger

    print("Loading Brown corpus...")

    sentences = load_brown()
    train_sentences, test_sentences = split_brown(sentences)

    print("Training POS tagger...")

    tagger = TrigramPOSTagger()
    tagger.train(train_sentences)

    lm = TrigramLanguageModel()
    lm.train([
        [word for word, tag in sentence]
        for sentence in train_sentences
    ])

    segmenter = ViterbiSegmenter(lm)

    evaluate_error_sources(
        test_sentences,
        segmenter,
        tagger
    )

    print("Tagging test set...")

    predicted_sentences = []

    for i, sentence in enumerate(test_sentences):

        words = [
            word
            for word, tag in sentence
        ]

        predicted = tagger.tag(words)

        predicted_sentences.append(predicted)

        if (i + 1) % 500 == 0:
            print(f"Tagged {i + 1}/{len(test_sentences)} sentences")

    accuracy = pos_accuracy(
        test_sentences,
        predicted_sentences
    )

    print(
        f"\nPOS accuracy: {accuracy:.4f}"
    )

    matrix = confusion_matrix(
        test_sentences,
        predicted_sentences
    )

    print_tiny_confusion_matrix(matrix)

    

"""Segmentation and POS evaluation metrics."""

from __future__ import annotations

from collections import Counter, defaultdict


def pos_accuracy(gold_sentences, predicted_sentences):
    """Token-level POS accuracy over aligned sentence pairs."""
    correct = 0
    total = 0

    for gold_sentence, predicted_sentence in zip(gold_sentences, predicted_sentences):
        for (_, gold_tag), (_, predicted_tag) in zip(gold_sentence, predicted_sentence):
            if gold_tag == predicted_tag:
                correct += 1
            total += 1

    return 0.0 if total == 0 else correct / total


def confusion_matrix(gold_sentences, predicted_sentences):
    """actual_tag -> predicted_tag -> count."""
    matrix = defaultdict(Counter)

    for gold_sentence, predicted_sentence in zip(gold_sentences, predicted_sentences):
        for (_, gold_tag), (_, predicted_tag) in zip(gold_sentence, predicted_sentence):
            matrix[gold_tag][predicted_tag] += 1

    return matrix


def print_tiny_confusion_matrix(matrix, top_n=15):
    """Print a compact confusion matrix for the most frequent tags."""
    tag_counts = Counter()
    for actual_tag, row in matrix.items():
        tag_counts[actual_tag] += sum(row.values())

    tags = [tag for tag, _ in tag_counts.most_common(top_n)]

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
            print(f"{matrix[actual_tag][predicted_tag]:<10}", end="")
        print()


def evaluate_error_sources(gold_sentences, segmenter, tagger, max_sentences: int | None = None):
    """
    Separate POS errors caused by bad segmentation from genuine tagging errors.

    If max_sentences is set, only the first max_sentences gold sentences are used
    (documented subsample for runtime; accuracy metrics still use the full test set).
    """
    if max_sentences is not None:
        gold_sentences = gold_sentences[:max_sentences]

    segmentation_caused_errors = 0
    genuine_tagging_errors = 0
    total_words = 0

    for sentence in gold_sentences:
        gold_words = [word.lower() for word, _tag in sentence]
        gold_tags = [tag for _word, tag in sentence]
        total_words += len(gold_words)

        text = "".join(gold_words)
        predicted_words = segmenter.segment(text)
        predicted_tagged = tagger.tag(predicted_words)

        gold_spans = []
        position = 0
        for word, tag in zip(gold_words, gold_tags):
            gold_spans.append((position, position + len(word), word, tag))
            position += len(word)

        predicted_spans = []
        position = 0
        for word, tag in predicted_tagged:
            predicted_spans.append((position, position + len(word), word.lower(), tag))
            position += len(word)

        for gold_start, gold_end, _gold_word, gold_tag in gold_spans:
            matching = [
                (word, tag)
                for start, end, word, tag in predicted_spans
                if start == gold_start and end == gold_end
            ]
            if len(matching) == 1:
                _predicted_word, predicted_tag = matching[0]
                if predicted_tag != gold_tag:
                    genuine_tagging_errors += 1
            else:
                segmentation_caused_errors += 1

    total_errors = segmentation_caused_errors + genuine_tagging_errors

    print("\nError Source Analysis")
    print("---------------------")
    if max_sentences is not None:
        print(f"(subsample: first {len(gold_sentences)} sentences)")
    print(f"Total gold words: {total_words}")
    print(f"Segmentation-caused errors: {segmentation_caused_errors}")
    print(f"Genuine tagging errors: {genuine_tagging_errors}")
    if total_errors > 0:
        print(f"Segmentation-caused: {segmentation_caused_errors / total_errors:.2%}")
        print(f"Genuine tagging: {genuine_tagging_errors / total_errors:.2%}")

    return {
        "total_words": total_words,
        "sentences_analyzed": len(gold_sentences),
        "max_sentences": max_sentences,
        "segmentation_caused_errors": segmentation_caused_errors,
        "genuine_tagging_errors": genuine_tagging_errors,
        "segmentation_caused_frac": (
            segmentation_caused_errors / total_errors if total_errors else None
        ),
        "genuine_tagging_frac": (
            genuine_tagging_errors / total_errors if total_errors else None
        ),
    }


def segmentation_accuracy(gold_sentences, segmenter):
    """Exact-match sentence segmentation accuracy for (word, tag) gold."""
    correct = 0
    total = 0
    for sentence in gold_sentences:
        gold_words = [word.lower() for word, _tag in sentence]
        predicted = segmenter.segment("".join(gold_words))
        if predicted == gold_words:
            correct += 1
        total += 1
    return 0.0 if total == 0 else correct / total


def spanish_segmentation_accuracy(gold_sentences, segmenter):
    """Exact-match sentence segmentation accuracy for Spanish token dicts."""
    correct = 0
    total = 0
    for sentence in gold_sentences:
        gold_words = [token["form"].lower() for token in sentence]
        predicted = segmenter.segment("".join(gold_words))
        if predicted == gold_words:
            correct += 1
        total += 1
    return 0.0 if total == 0 else correct / total

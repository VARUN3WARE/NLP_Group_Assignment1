"""Full Question 1 experiment runner (English + Spanish)."""

from __future__ import annotations

from segpos.baselines.simple import GreedyLongestMatchSegmenter, MostFrequentTagger
from segpos.data.corpus import load_brown, load_spanish, split_brown
from segpos.eval.metrics import (
    confusion_matrix,
    evaluate_error_sources,
    pos_accuracy,
    print_tiny_confusion_matrix,
    segmentation_accuracy,
    spanish_segmentation_accuracy,
)
from segpos.lm.trigram import TrigramLanguageModel
from segpos.segmentation.viterbi import ViterbiSegmenter
from segpos.tagging.pos_tagger import MorphologyAwarePOSTagger, TrigramPOSTagger


def run_full_evaluation(include_error_analysis: bool = True) -> dict:
    results = {}

    print("\n====================")
    print("ENGLISH")
    print("====================")

    brown = load_brown()
    english_train, english_test = split_brown(brown)

    english_lm = TrigramLanguageModel()
    english_lm.train([[word for word, _tag in sentence] for sentence in english_train])

    viterbi_segmenter = ViterbiSegmenter(english_lm)
    greedy_segmenter = GreedyLongestMatchSegmenter(english_lm.vocabulary)

    print("\nEnglish segmentation...")
    english_viterbi_seg_acc = segmentation_accuracy(english_test, viterbi_segmenter)
    english_greedy_seg_acc = segmentation_accuracy(english_test, greedy_segmenter)
    print(f"Viterbi segmentation accuracy: {english_viterbi_seg_acc:.4%}")
    print(f"Greedy segmentation accuracy:  {english_greedy_seg_acc:.4%}")
    print(f"Improvement: {english_viterbi_seg_acc - english_greedy_seg_acc:.4%}")

    english_tagger = TrigramPOSTagger()
    english_tagger.train(english_train)
    english_baseline = MostFrequentTagger()
    english_baseline.train(english_train)

    english_predictions = []
    english_baseline_predictions = []
    print("\nEnglish POS tagging...")
    for i, sentence in enumerate(english_test):
        words = [word for word, _tag in sentence]
        english_predictions.append(english_tagger.tag(words))
        english_baseline_predictions.append(english_baseline.tag(words))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(english_test)}")

    english_pos_acc = pos_accuracy(english_test, english_predictions)
    english_baseline_acc = pos_accuracy(english_test, english_baseline_predictions)
    print(f"\nViterbi POS accuracy:         {english_pos_acc:.4%}")
    print(f"Most-frequent-tag accuracy:  {english_baseline_acc:.4%}")
    print(f"Improvement: {english_pos_acc - english_baseline_acc:.4%}")

    matrix = confusion_matrix(english_test, english_predictions)
    print_tiny_confusion_matrix(matrix)

    if include_error_analysis:
        evaluate_error_sources(english_test, viterbi_segmenter, english_tagger)

    results["english"] = {
        "viterbi_seg": english_viterbi_seg_acc,
        "greedy_seg": english_greedy_seg_acc,
        "viterbi_pos": english_pos_acc,
        "baseline_pos": english_baseline_acc,
    }

    print("\n====================")
    print("SPANISH")
    print("====================")

    spanish_train, _spanish_dev, spanish_test = load_spanish()

    spanish_lm = TrigramLanguageModel()
    spanish_lm.train([[token["form"] for token in sentence] for sentence in spanish_train])

    spanish_viterbi_segmenter = ViterbiSegmenter(spanish_lm)
    spanish_greedy_segmenter = GreedyLongestMatchSegmenter(spanish_lm.vocabulary)

    print("\nSpanish segmentation...")
    spanish_viterbi_seg_acc = spanish_segmentation_accuracy(
        spanish_test, spanish_viterbi_segmenter
    )
    spanish_greedy_seg_acc = spanish_segmentation_accuracy(
        spanish_test, spanish_greedy_segmenter
    )
    print(f"Viterbi segmentation accuracy: {spanish_viterbi_seg_acc:.4%}")
    print(f"Greedy segmentation accuracy:  {spanish_greedy_seg_acc:.4%}")
    print(f"Improvement: {spanish_viterbi_seg_acc - spanish_greedy_seg_acc:.4%}")

    spanish_plain_train = [
        [(token["form"], token["upos"]) for token in sentence]
        for sentence in spanish_train
    ]
    spanish_plain_test = [
        [(token["form"], token["upos"]) for token in sentence]
        for sentence in spanish_test
    ]

    spanish_tagger = TrigramPOSTagger()
    spanish_tagger.train(spanish_plain_train)
    spanish_baseline = MostFrequentTagger()
    spanish_baseline.train(spanish_plain_train)

    spanish_predictions = []
    spanish_baseline_predictions = []
    print("\nSpanish plain POS tagging...")
    for sentence in spanish_plain_test:
        words = [word for word, _tag in sentence]
        spanish_predictions.append(spanish_tagger.tag(words))
        spanish_baseline_predictions.append(spanish_baseline.tag(words))

    spanish_pos_acc = pos_accuracy(spanish_plain_test, spanish_predictions)
    spanish_baseline_acc = pos_accuracy(spanish_plain_test, spanish_baseline_predictions)
    print(f"Plain Viterbi POS accuracy:    {spanish_pos_acc:.4%}")
    print(f"Most-frequent-tag accuracy:    {spanish_baseline_acc:.4%}")
    print(f"Improvement: {spanish_pos_acc - spanish_baseline_acc:.4%}")

    spanish_morph_tagger = MorphologyAwarePOSTagger()
    spanish_morph_tagger.train_spanish(spanish_train)

    spanish_morph_gold = []
    spanish_morph_predictions = []
    print("\nSpanish morphology-aware POS tagging...")
    for sentence in spanish_test:
        words = []
        gold_sentence = []
        for token in sentence:
            words.append(token["form"])
            morph_tag = MorphologyAwarePOSTagger.make_morph_tag(
                token["upos"], token["feats"]
            )
            gold_sentence.append((token["form"], morph_tag))
        spanish_morph_gold.append(gold_sentence)
        spanish_morph_predictions.append(spanish_morph_tagger.tag(words))

    spanish_morph_acc = pos_accuracy(spanish_morph_gold, spanish_morph_predictions)
    print(f"Morphology-aware POS accuracy: {spanish_morph_acc:.4%}")
    print(f"Difference from plain POS:     {spanish_morph_acc - spanish_pos_acc:.4%}")

    results["spanish"] = {
        "viterbi_seg": spanish_viterbi_seg_acc,
        "greedy_seg": spanish_greedy_seg_acc,
        "viterbi_pos": spanish_pos_acc,
        "baseline_pos": spanish_baseline_acc,
        "morph_pos": spanish_morph_acc,
    }

    print("\n====================")
    print("FINAL RESULTS")
    print("====================")
    print("\nSegmentation:")
    print(f"English Viterbi:  {english_viterbi_seg_acc:.4%}")
    print(f"English Greedy:   {english_greedy_seg_acc:.4%}")
    print(f"Spanish Viterbi:  {spanish_viterbi_seg_acc:.4%}")
    print(f"Spanish Greedy:   {spanish_greedy_seg_acc:.4%}")
    print("\nPOS:")
    print(f"English Viterbi:  {english_pos_acc:.4%}")
    print(f"English Baseline: {english_baseline_acc:.4%}")
    print(f"Spanish Viterbi:  {spanish_pos_acc:.4%}")
    print(f"Spanish Baseline: {spanish_baseline_acc:.4%}")
    print(f"Spanish Morph:    {spanish_morph_acc:.4%}")

    return results

"""
Question 3 spelling corrector package.

Q4 reuse:

    from spelling import load_models
    model, corrector = load_models()
"""

from spelling.candidates import (
    EditDistance1Generator,
    SymmetricDeleteIndex,
    deletes1,
    within_edit_distance_1,
)
from spelling.corrector import Correction, SpellingCorrector, load_models
from spelling.evaluation import (
    AccuracyResult,
    BenchmarkResult,
    FalseAlarmResult,
    TestCase,
    build_test_sets,
    corrupt,
    error_breakdown,
    evaluate,
    false_alarm_rate,
    make_misspelling_batch,
    single_edits,
    speed_demon,
    threshold_sweep,
    time_generator,
)
from spelling.language_model import LanguageModel, brown_sentences, train_test_split
from spelling.paths import (
    ARTIFACTS_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_SPLIT_PATH,
    RESULTS_DIR,
)
from spelling.tokenize import (
    ALPHABET,
    BOS,
    EOS,
    NUM,
    normalise,
    normalise_sentence,
)

__all__ = [
    "ALPHABET",
    "ARTIFACTS_DIR",
    "AccuracyResult",
    "BOS",
    "BenchmarkResult",
    "Correction",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_SPLIT_PATH",
    "EOS",
    "EditDistance1Generator",
    "FalseAlarmResult",
    "LanguageModel",
    "NUM",
    "RESULTS_DIR",
    "SpellingCorrector",
    "SymmetricDeleteIndex",
    "TestCase",
    "brown_sentences",
    "build_test_sets",
    "corrupt",
    "deletes1",
    "error_breakdown",
    "evaluate",
    "false_alarm_rate",
    "load_models",
    "make_misspelling_batch",
    "normalise",
    "normalise_sentence",
    "single_edits",
    "speed_demon",
    "threshold_sweep",
    "time_generator",
    "train_test_split",
    "within_edit_distance_1",
]

"""Tests for the Question 3 spelling corrector.

The important ones are the equivalence tests: Method A and Method B are two
completely different algorithms that must agree on every input, and that
property is what makes the Speed Demon comparison meaningful.
"""

from __future__ import annotations

import random

import pytest

from spelling import (
    ALPHABET,
    DEFAULT_MODEL_PATH,
    EditDistance1Generator,
    LanguageModel,
    SpellingCorrector,
    SymmetricDeleteIndex,
    build_test_sets,
    corrupt,
    deletes1,
    make_misspelling_batch,
    normalise,
    normalise_sentence,
    single_edits,
    within_edit_distance_1,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

TOY_SENTENCES = [
    "the quick brown fox jumps over the lazy dog".split(),
    "I would like to see the world".split(),
    "please meet me at the station".split(),
    "the dog sat on the mat".split(),
    "she is a very good student".split(),
    "this is a test sentence".split(),
    "I have a good feeling about this".split(),
    "the cat and the dog and the fox".split(),
]


@pytest.fixture(scope="session")
def toy_model() -> LanguageModel:
    return LanguageModel.train(TOY_SENTENCES)


@pytest.fixture(scope="session")
def brown_model() -> LanguageModel:
    """The real trained model; skipped if it has not been trained yet."""
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("run `python main.py train` first")
    return LanguageModel.load(DEFAULT_MODEL_PATH)


# --------------------------------------------------------------------------- #
# Part 1 - corpus and model preparation
# --------------------------------------------------------------------------- #

def test_normalise_keeps_words_drops_punctuation():
    assert normalise("The") == "the"
    assert normalise("don't") == "don't"
    assert normalise("Atlanta's") == "atlanta's"
    assert normalise(",") is None
    assert normalise("``") is None
    assert normalise("1961") == "<num>"
    assert normalise("$1.50") == "<num>"


def test_normalise_sentence_drops_nothing_lexical():
    assert normalise_sentence(["The", "dog", ",", "he", "said", "."]) == \
        ["the", "dog", "he", "said"]


def test_unigram_probabilities_sum_to_one(toy_model):
    total = sum(toy_model.unigram_prob(w) for w in toy_model.unigram_counts)
    assert total == pytest.approx(1.0)


def test_bigram_distribution_sums_to_one(toy_model):
    """Add-k must still be a proper conditional distribution."""
    continuations = set(toy_model.unigram_counts) | {"</s>"}
    # Sum over the full continuation space, including the unseen mass.
    seen = sum(toy_model.bigram_prob("the", w) for w in continuations)
    unseen_slots = toy_model.vocab_size - len(continuations)
    unseen = unseen_slots * toy_model.bigram_prob("the", "\x00-not-a-word")
    assert seen + unseen == pytest.approx(1.0)


def test_interpolated_bigram_is_a_probability(toy_model):
    continuations = set(toy_model.unigram_counts) | {"</s>"}
    total = sum(toy_model.bigram_prob_interp("the", w) for w in continuations)
    assert 0.0 < total <= 1.0 + 1e-9


def test_interpolated_bigram_prefers_observed_continuations(toy_model):
    assert (toy_model.bigram_prob_interp("the", "dog")
            > toy_model.bigram_prob_interp("the", "station"))


def test_num_placeholder_is_not_in_the_vocabulary(toy_model):
    model = LanguageModel.train([["it", "was", "1961", "then"]])
    assert "<num>" not in model.vocabulary
    assert "<num>" in model.unigram_counts


def test_save_load_round_trip(toy_model, tmp_path):
    path = tmp_path / "m.pkl"
    toy_model.save(path)
    loaded = LanguageModel.load(path)
    assert loaded.vocabulary == toy_model.vocabulary
    assert loaded.total_tokens == toy_model.total_tokens
    assert loaded.log_bigram_prob("the", "dog") == toy_model.log_bigram_prob("the", "dog")


# --------------------------------------------------------------------------- #
# Part 2 - edit distance helper
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("a,b", [
    ("hello", "hello"),      # identical
    ("hello", "helo"),       # deletion
    ("hello", "helllo"),     # insertion
    ("hello", "hallo"),      # substitution
    ("hello", "hlelo"),      # transposition
    ("", "a"),               # empty vs single char
])
def test_within_edit_distance_1_accepts(a, b):
    assert within_edit_distance_1(a, b)
    assert within_edit_distance_1(b, a)


@pytest.mark.parametrize("a,b", [
    ("hello", "help"),
    ("abc", "axb"),          # the classic SymSpell bucket collision
    ("hello", "world"),
    ("hello", "hellooo"),
    ("abc", "cba"),
])
def test_within_edit_distance_1_rejects(a, b):
    assert not within_edit_distance_1(a, b)
    assert not within_edit_distance_1(b, a)


def test_within_edit_distance_1_matches_brute_force():
    """Cross-check the linear implementation against the definition."""
    rng = random.Random(0)
    letters = "abcd"
    for _ in range(400):
        a = "".join(rng.choice(letters) for _ in range(rng.randint(0, 5)))
        b = "".join(rng.choice(letters) for _ in range(rng.randint(0, 5)))
        expected = a == b or b in single_edits(a, alphabet=letters)
        assert within_edit_distance_1(a, b) is expected, (a, b)


def test_deletes1():
    assert deletes1("abc") == {"bc", "ac", "ab"}
    assert deletes1("a") == {""}
    assert deletes1("") == set()


# --------------------------------------------------------------------------- #
# Part 2 - Method A and Method B
# --------------------------------------------------------------------------- #

def test_edits1_size_matches_the_formula():
    gen = EditDistance1Generator({"x"}, alphabet="ab")
    word = "abc"
    n, a = len(word), 2
    # Upper bound; duplicates collapse in the set.
    assert len(gen.edits1(word)) <= n + (n - 1) + n * a + (n + 1) * a


def test_method_a_returns_only_vocabulary_words(toy_model):
    gen = EditDistance1Generator(toy_model.vocabulary)
    for cand in gen.candidates("dg"):
        assert cand in toy_model.vocabulary


def test_method_a_excludes_the_query_itself(toy_model):
    gen = EditDistance1Generator(toy_model.vocabulary)
    assert "dog" not in gen.candidates("dog")


def test_symspell_index_contains_word_and_its_deletions():
    index = SymmetricDeleteIndex({"dog"})
    assert index.index["dog"] == ["dog"]
    for variant in ("og", "dg", "do"):
        assert "dog" in index.index[variant]


@pytest.mark.parametrize("query,expected", [
    ("dg", "dog"),      # deletion in the query
    ("dogg", "dog"),    # insertion in the query
    ("dob", "dog"),     # substitution
    ("odg", "dog"),     # transposition
])
def test_symspell_covers_all_four_edit_operations(query, expected):
    index = SymmetricDeleteIndex({"dog"})
    assert expected in index.candidates(query)


def test_methods_a_and_b_agree_on_the_toy_vocabulary(toy_model):
    gen_a = EditDistance1Generator(toy_model.vocabulary)
    gen_b = SymmetricDeleteIndex(toy_model.vocabulary, verify=True)
    rng = random.Random(1)
    probes = list(toy_model.vocabulary) + [
        "".join(rng.choice("abcdefgo") for _ in range(rng.randint(1, 6)))
        for _ in range(300)
    ]
    for probe in probes:
        assert gen_a.candidates(probe) == gen_b.candidates(probe), probe


def test_methods_a_and_b_agree_on_the_brown_vocabulary(brown_model):
    gen_a = EditDistance1Generator(brown_model.vocabulary)
    gen_b = SymmetricDeleteIndex(brown_model.vocabulary, verify=True)
    rng = random.Random(2)
    vocab = sorted(brown_model.vocabulary)
    probes = [rng.choice(vocab) for _ in range(200)]
    probes += [corrupt(w, brown_model.vocabulary, rng, want_real_word=False) or w
               for w in probes[:100]]
    for probe in probes:
        assert gen_a.candidates(probe) == gen_b.candidates(probe), probe


def test_unverified_symspell_is_a_strict_superset(brown_model):
    """The verification step is not decorative: raw buckets over-generate."""
    gen_a = EditDistance1Generator(brown_model.vocabulary)
    raw = SymmetricDeleteIndex(brown_model.vocabulary, verify=False)
    rng = random.Random(3)
    vocab = sorted(brown_model.vocabulary)
    over_generated = 0
    for _ in range(300):
        probe = rng.choice(vocab)
        exact = gen_a.candidates(probe)
        loose = raw.candidates(probe)
        assert exact <= loose
        over_generated += len(loose - exact)
    assert over_generated > 0


# --------------------------------------------------------------------------- #
# Part 3 - correction logic
# --------------------------------------------------------------------------- #

def test_non_word_correction_picks_the_most_frequent_candidate(toy_model):
    corrector = SpellingCorrector(toy_model)
    # "the" (5 occurrences) beats "she" (1) as a correction for "he".
    assert toy_model.unigram_counts["the"] > toy_model.unigram_counts["she"]
    assert corrector.correct_non_word("thw") == "the"


def test_non_word_correction_leaves_unreachable_words_alone(toy_model):
    corrector = SpellingCorrector(toy_model)
    assert corrector.correct_non_word("zzzzqqqq") == "zzzzqqqq"


def test_correction_is_identical_across_methods(brown_model):
    corrector = SpellingCorrector(brown_model)
    for word in ["sentnce", "speling", "korrect", "recieve", "adress", "hav"]:
        assert (corrector.correct_non_word(word, method="A")
                == corrector.correct_non_word(word, method="B")
                == corrector.correct_non_word(word, method="both"))


def test_real_word_correction_uses_context(brown_model):
    corrector = SpellingCorrector(brown_model, real_word_threshold=5.0)
    fixed, gain = corrector.correct_real_word("sea", prev="to", nxt="the")
    assert fixed == "see" and gain > 0


def test_real_word_correction_leaves_a_well_fitting_word_alone(brown_model):
    corrector = SpellingCorrector(brown_model, real_word_threshold=5.0)
    fixed, gain = corrector.correct_real_word("see", prev="to", nxt="the")
    assert fixed == "see" and gain == 0.0


def test_real_word_correction_respects_the_threshold(brown_model):
    """An unreachable threshold must silence every real-word suggestion."""
    corrector = SpellingCorrector(brown_model, real_word_threshold=1e9)
    fixed, _ = corrector.correct_real_word("sea", prev="to", nxt="the")
    assert fixed == "sea"


def test_short_words_are_never_second_guessed(brown_model):
    corrector = SpellingCorrector(brown_model, min_real_word_length=3)
    for word in ["a", "an", "of", "to"]:
        fixed, _ = corrector.correct_real_word(word, prev="the", nxt="the")
        assert fixed == word


@pytest.mark.parametrize("text,expected", [
    ("This is a test sentnce.", "This is a test sentence."),
    ("I would like to sea the world.", "I would like to see the world."),
])
def test_correct_text_on_the_specification_examples(brown_model, text, expected):
    corrector = SpellingCorrector(brown_model, real_word_threshold=5.0)
    assert corrector.correct_text(text)[0] == expected


def test_correct_text_preserves_punctuation_and_spacing(brown_model):
    corrector = SpellingCorrector(brown_model, real_word_threshold=1e9)
    text = "Hello,   world!  (Really?)"
    assert corrector.correct_text(text)[0] == text


def test_correct_text_preserves_capitalisation(brown_model):
    corrector = SpellingCorrector(brown_model)
    assert corrector.correct_text("Sentnce")[0] == "Sentence"
    assert corrector.correct_text("SENTNCE")[0] == "SENTENCE"


def test_correct_text_reports_what_it_changed(brown_model):
    corrector = SpellingCorrector(brown_model)
    _, changes, _ = corrector.correct_text("This is a test sentnce.")
    assert [(c.original, c.corrected, c.kind) for c in changes] == \
        [("sentnce", "sentence", "non-word")]


def test_real_word_checking_can_be_switched_off(brown_model):
    corrector = SpellingCorrector(brown_model, real_word_threshold=0.0)
    text = "I would like to sea the world."
    assert corrector.correct_text(text, check_real_words=False)[0] == text


# --------------------------------------------------------------------------- #
# Part 4 - test-set construction and the benchmark batch
# --------------------------------------------------------------------------- #

def test_corrupt_produces_the_requested_error_class(brown_model):
    rng = random.Random(4)
    vocab = brown_model.vocabulary
    for word in ["sentence", "station", "student", "feeling"]:
        non_word = corrupt(word, vocab, rng, want_real_word=False)
        real_word = corrupt(word, vocab, rng, want_real_word=True)
        assert non_word not in vocab and within_edit_distance_1(word, non_word)
        assert real_word in vocab and real_word != word
        assert within_edit_distance_1(word, real_word)


def test_build_test_sets_produces_aligned_pairs(brown_model):
    non_word, real_word = build_test_sets(
        TOY_SENTENCES * 20, brown_model.vocabulary, max_cases=10
    )
    assert len(non_word) == len(real_word)
    for nw, rw in zip(non_word, real_word):
        assert nw.index == rw.index and nw.original == rw.original
        assert nw.noisy[nw.index] == nw.corrupted
        assert nw.corrupted not in brown_model.vocabulary
        assert rw.corrupted in brown_model.vocabulary


def test_benchmark_batch_is_exactly_one_thousand(brown_model):
    batch = make_misspelling_batch(TOY_SENTENCES * 400, brown_model.vocabulary, n=1000)
    assert len(batch) == 1000
    assert all(w not in brown_model.vocabulary for w in batch)


def test_alphabet_covers_contractions():
    assert "'" in ALPHABET
    index = SymmetricDeleteIndex({"don't"})
    assert "don't" in index.candidates("dont")

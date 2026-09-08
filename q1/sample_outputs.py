from corpus import load_brown, split_brown, load_spanish
from language_model import TrigramLanguageModel
from segmentation import ViterbiSegmenter
from pos_tagger import TrigramPOSTagger, MorphologyAwarePOSTagger


# -------------------------
# English
# -------------------------

brown = load_brown()
english_train, _ = split_brown(brown)

english_lm = TrigramLanguageModel()

english_words = [
    [word for word, tag in sentence]
    for sentence in english_train
]

english_lm.train(english_words)

english_segmenter = ViterbiSegmenter(english_lm)

english_tagger = TrigramPOSTagger()
english_tagger.train(english_train)


# -------------------------
# Spanish
# -------------------------

spanish_train, _, _ = load_spanish()

spanish_lm = TrigramLanguageModel()

spanish_words = [
    [token["form"] for token in sentence]
    for sentence in spanish_train
]

spanish_lm.train(spanish_words)

spanish_segmenter = ViterbiSegmenter(spanish_lm)

spanish_tagger = TrigramPOSTagger()

spanish_plain_sentences = [
    [
        (token["form"], token["upos"])
        for token in sentence
    ]
    for sentence in spanish_train
]

spanish_tagger.train(spanish_plain_sentences)

spanish_morph_tagger = MorphologyAwarePOSTagger()
spanish_morph_tagger.train_spanish(spanish_train)


# -------------------------
# Sample sentences
# -------------------------

english_sample_1 = "thequickbrownfoxjumpsoverthelazydog"

english_sample_2 = "tobeornottobethatisthequestion"

spanish_sample_1 = "mispadrespuedenviajar"

spanish_sample_2 = "elcielodespejadoesazul"

spanish_sample_3 = "maríaleeunlibro"


# -------------------------
# English output
# -------------------------

print("\n=== ENGLISH ===")

for sample in [
    english_sample_1,
    english_sample_2,
]:

    segmented = english_segmenter.segment(sample)

    print("\nInput:")
    print(sample)

    print("\nSegmentation:")
    print(segmented)

    print("\nPOS:")
    print(english_tagger.tag(segmented))


# -------------------------
# Spanish output
# -------------------------

print("\n=== SPANISH ===")

for sample in [
    spanish_sample_1,
    spanish_sample_2,
    spanish_sample_3
]:

    segmented = spanish_segmenter.segment(sample)

    print("\nInput:")
    print(sample)

    print("\nSegmentation:")
    print(segmented)

    print("\nPOS:")
    print(spanish_tagger.tag(segmented))

    print("\nMorphology-aware POS:")
    print(spanish_morph_tagger.tag(segmented))

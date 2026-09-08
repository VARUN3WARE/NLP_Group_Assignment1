# Question 1: Word Segmentation and POS Tagging

English + Spanish word segmentation and POS tagging with trigram models,
Viterbi decoding, and a **joint beam decoder** for Q4 reuse.

## Layout

```text
q1/
├── corpus.py            Brown + Spanish CoNLL-U loaders
├── language_model.py    trigram word LM (save/load)
├── segmentation.py      Viterbi segmenter
├── pos_tagger.py        trigram HMM POS + morphology-aware Spanish
├── joint_decoder.py     joint beam segmentation + POS (α, β, beam width)
├── pipeline.py          train / save / load English bundle for Q4
├── baselines.py         greedy segment + most-frequent tag
├── evaluation.py        accuracy, confusion matrix, error sources
├── results.py           full experiment runner
├── sample_outputs.py    legacy sample script
└── main.py              CLI
```

## Setup

```bash
pip install nltk
python -c "import nltk; nltk.download('brown')"

# Spanish corpus (optional for English-only / Q4)
mkdir -p data/spanish
git clone https://github.com/UniversalDependencies/UD_Spanish-GSD.git \
  data/spanish/UD_Spanish-GSD
```

## Commands

```bash
cd q1
python main.py train-english   # writes models/english_pipeline.pkl
python main.py sample
python main.py evaluate        # full metrics (slow)
python results.py              # same evaluation entrypoint
```

## Reuse from Question 4

```python
from pipeline import load_english_pipeline

pipe = load_english_pipeline()                 # no retrain
pipe.decode("thequickbrownfox")                # joint beam → [(w, tag), ...]
split, pairs = pipe.should_split("thequick")   # for [SEGMENT-ALERT]
```

Decoder defaults (also stored in the pickle): `max_word_length=20`, `α=1.0`, `β=1.0`, `beam_width=8`.

## Results (summary)

| Task | Model | Baseline | Improvement |
|------|------:|--------:|------------:|
| EN segmentation | 48.85% | 25.50% | +23.35 pp |
| ES segmentation | 14.99% | 6.09% | +8.90 pp |
| EN POS | 92.94% | 87.29% | +5.65 pp |
| ES POS | 92.94% | 88.55% | +4.39 pp |
| ES morph POS | 90.57% | — | −2.38 pp vs plain |

English end-to-end errors: ~66% from segmentation, ~34% genuine tagging.

English tags use the **Brown** tagset (`AT`, `NN`, `JJ`, …), not Universal POS.

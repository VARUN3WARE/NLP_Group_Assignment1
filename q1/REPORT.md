# Question 1 — Comparative Report

Word segmentation (trigram LM + Viterbi) and POS tagging (trigram HMM + Viterbi)
for **English** (Brown) and **Spanish** (UD Spanish-GSD), plus morphology-aware
tags for Spanish, baselines, and error-source analysis.

## Data handling

| Language | Corpus | Train / Dev / Test |
|----------|--------|--------------------|
| English | NLTK Brown tagged sentences | **80/20** sequential split (45,872 / 11,468). No official UD-style splits. |
| Spanish | UD Spanish-GSD CoNLL-U | Official **train / dev / test** (14,186 / 1,400 / 427). Train for training; test for final numbers; dev reserved for future tuning. |

Segmentation LMs are trained on **lowercased** word forms so they match the
no-space evaluation strings. POS taggers keep original surface forms and corpus
tagsets (**Brown** for English, **UPOS** / morph tags for Spanish).

English sample outputs therefore use Brown tags (`AT`, `NN`, `JJ`, …), not the
Penn-style labels (`DT`, `NN`) shown in the assignment PDF illustration.

## How to reproduce

```bash
cd q1
# Spanish data (once):
mkdir -p data/spanish
git clone https://github.com/UniversalDependencies/UD_Spanish-GSD.git \
  data/spanish/UD_Spanish-GSD

../.venv/bin/python main.py train-english
../.venv/bin/python main.py sample
../.venv/bin/python main.py evaluate   # writes results/evaluation.json
```

Numbers below match `results/evaluation.json` from a full evaluate run.

## Results summary

### Segmentation (exact-match sentence accuracy)

| Language | Viterbi (trigram+DP) | Greedy longest-match | Δ |
|----------|---------------------:|---------------------:|--:|
| English | **48.85%** | 25.50% | **+23.35 pp** |
| Spanish | **14.99%** | 6.09% | **+8.90 pp** |

### POS tagging (token accuracy on gold segmentation)

| Language | Model | Baseline (MFT) | Δ |
|----------|------:|---------------:|--:|
| English | **92.94%** | 87.29% | **+5.65 pp** |
| Spanish (plain UPOS) | **92.94%** | 88.55% | **+4.39 pp** |
| Spanish (morph: UPOS-Gender-Number) | **90.57%** | — | **−2.38 pp** vs plain |

### Error-source breakdown (seg → tag pipeline)

| Language | Seg-caused share of errors | Genuine tagging share | Gold words analyzed |
|----------|---------------------------:|----------------------:|--------------------:|
| English | **65.99%** | 34.01% | 181,546 (full test) |
| Spanish | **82.68%** | 17.32% | 12,000 (full test) |

## Answers to the assignment questions

### 1. Where did English and Spanish differ most in accuracy?

**Segmentation**, not POS. English Viterbi sentence exact-match is ~49%; Spanish
is only ~15% — a gap of about **34 percentage points**. Plain POS accuracy is
almost identical (~92.9% for both). Spanish has richer morphology and compounding
patterns, a smaller treebank relative to Brown, and more OOV surface forms after
lowercasing; the trigram LM therefore has a harder time choosing correct word
boundaries. Once words are given (gold segmentation), the Spanish UPOS tagger
matches English quality.

### 2. Did agreement-aware tagging help, or add noise?

**It added noise on this setup.** Morphology-aware accuracy is **90.57%** vs
**92.94%** plain UPOS (−2.38 pp). Extending tags with Gender/Number multiplies the
tagset and sparsifies emission/transition counts, so the trigram HMM underfits
rare morph tags even when gender/number agreement is linguistically real. Plain
UPOS remains the better operating point for raw accuracy; morph tags are still
useful as a diagnostic that agreement features need more data or better
smoothing, not that agreement is unimportant.

### 3. How much tagging error came from segmentation vs genuine tagging?

On the end-to-end pipeline (Viterbi segment, then Viterbi tag):

- **English:** ~**66%** of word-level errors are caused by wrong boundaries;
  ~**34%** are genuine POS mistakes on correctly segmented spans.
- **Spanish:** ~**83%** are segmentation-caused; only ~**17%** are genuine tagging
  errors.

So for Spanish especially, improving the segmenter would dominate any POS
refinement. Confusion matrices (printed by `evaluate`) show the remaining genuine
tag confusions: English `NN`↔`JJ`/`VB`/`NP`, Spanish `NOUN`↔`PROPN`/`ADJ`,
`VERB`↔`AUX`.

### 4. How much better were the models than the simple baselines?

| Task | Improvement over baseline |
|------|--------------------------:|
| English segmentation (vs greedy) | **+23.35 pp** |
| Spanish segmentation (vs greedy) | **+8.90 pp** |
| English POS (vs most-frequent-tag) | **+5.65 pp** |
| Spanish POS (vs most-frequent-tag) | **+4.39 pp** |

Trigram+DP clearly beats greedy longest-match for segmentation. The POS HMM beats
MFT by a smaller but consistent margin because many words are unambiguous and MFT
is already strong.

## Sample outputs (see `python main.py sample`)

**English** (PDF strings; Brown tags via Viterbi segment-then-tag):

- `thequickbrownfox` → `(the, AT), (quick, JJ), (brown, JJ), (fox, NN)`
- `thequickbrownfoxjumpsoverthelazydog` → segment-then-tag + joint beam shown in CLI

**Spanish** (PDF strings):

- `mispadrespuedenviajar` → `mis/DET padres/NOUN pueden/AUX viajar/VERB`
  (morph: `DET-Pl`, `NOUN-Masc-Pl`, …)
- `lacasarojaesgrande` → `la/DET casa/NOUN roja/ADJ es/AUX grande/ADJ`
  (morph shows Fem-Sg agreement on `la`/`casa`/`roja`)
- `elcielodespejadoesazul` → harder for the segmenter (`despejado` often
  over-segmented); illustrates Spanish boundary difficulty from §1.

## Submission checklist

1. Segmentation model (EN + ES) — `segpos/lm/trigram.py`, `segpos/segmentation/viterbi.py`
2. POS + morph — `segpos/tagging/pos_tagger.py`
3. Baselines — `segpos/baselines/simple.py`
4. Evaluation — `segpos/eval/` (accuracy, confusion matrix, error-source)
5. Sample strings — `python main.py sample`
6. This comparative report — `q1/REPORT.md`

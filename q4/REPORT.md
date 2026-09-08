# Question 4 — Report

Integrated background editor: Q1 joint segmentation/POS, Q3 spelling, Q4
smoothed n-grams + PCFG constituency checking.

## Configuration choices

| Parameter | Value | Justification |
|-----------|------:|---------------|
| Merge probability `p` | **0.08** | Spec example; yields enough spacebar-miss merges for segmentation without flooding every token |
| Grammar trigger `N` | **5** | Short enough to catch local anomalies quickly; keeps trigger cost low vs per-token seg/spell |
| Add-k for Q4 LMs | **0.01** | Same operating point as Q3; avoids zero probs without swamping ML counts |
| Spell candidates | **Method B** | Q3 showed B ≡ A on candidates and ~13× faster — required for live UI |
| Real-word margin | **5.0 nats** | Q3 default threshold |
| Grammar PP threshold | **5000** | Literary Gutenberg text sits far above Brown-trained LM; 5000 flags clear outliers without constant noise |
| PCFG sentence cap | **25 tokens** | CKY/Viterbi on long Melville/Chesterton sentences is too slow for interactive use; longer → `unparseable` |

**Reuse boundary:** Q1 English trigram + beam decoder are **loaded only** for segmentation scoring. Q3 vocab/unigram/bigram + Method B are **loaded only** for spelling / real-word. Q4 trains its **own** Brown add-k bigram+trigram for grammar alerts and final scores, and its **own** PTB PCFG.

## Tagset reconciliation

Q1 emits **Brown** tags (`AT`, `BEDZ`, `PPO`, …). The PCFG lexicon uses **Penn Treebank** tags. We map with a fixed lookup in `editor/tagset.py` (e.g. `AT→DT`, `BEZ→VBZ`, `PPSS→PRP`). Unseen Brown tags fall back to `NN` after stripping Brown hyphen suffixes.

**Accuracy cost:** mapping is many-to-one and loses Brown-specific distinctions (e.g. `BE*` forms → coarse PTB verb tags). For parsing we still run the PCFG over **words** (ViterbiParser lexicon); Brown→PTB is applied when packaging tagged pairs and is documented for the assignment’s reconciliation requirement. PTB-OOV surface words are projected to the known noun `something` so CKY can return a structural score (`logP=…*` in the table). Projection biases lexical probability but preserves a usable constituency signal.

## Decision rule (Part 4)

1. If PCFG returns a finite log-probability and it is not a high-z outlier vs other sentences in the passage → choose **pcfg**, verdict **grammatical**.
2. Else if trigram log-prob is finite → choose **trigram**, verdict **plausible** if `tri_lp > -80` else **implausible**.
3. Else → **bigram** with the same threshold rule.

## Sample run A (`--seed 42`, bible-kjv)

- Merges such as `himout`, `couldnot` correctly raised **SEGMENT-ALERT** and split.
- Grammar triggers fired on high-perplexity windows of archaic KJV wording.
- Latency: ~0.07 ms/token seg+spell, ~0.08 ms/trigger grammar — far below typing delay.

## Sample run B (`--seed 7` / `99`, Chesterton / Moby Dick)

- Merges like `wasmaking`, `ofthe`, `wascarried`, `whofirst` split cleanly when both pieces were in the Q3 vocabulary.
- Spelling occasionally rewrote rare literary tokens (e.g. frequency-driven `whale→while`) — interaction effect: a SPELL fix can change n-gram scores and even which Part-4 method wins.
- One shorter projected sentence received **pcfg / grammatical**; long sentences were `unparseable` (length cap or chart failure) and fell back to trigram.

## Speed Demon (1000 words)

| Path | Total | Per word |
|------|------:|--------:|
| Segmentation + spelling | 0.369 s | **0.369 ms** |
| Grammar-trigger only | 0.019 s | **0.019 ms** |
| Overhead (seg+spell − grammar) | — | **0.350 ms** |

**Conclusion:** The segmentation+spelling layer is cheap enough to keep on every token in live mode; throttling it to the grammar interval is unnecessary at ~0.4 ms/word.

## Comparative analysis

1. **Live alerts vs final verdict:** SEGMENT/SPELL alerts are local and usually agree with the corrected sentence used in Part 4. GRAMMAR-ALERT windows often fire on domain shift (Brown LM vs Gutenberg) even when the final trigram verdict is only “implausible,” i.e. live grammar is more aggressive than the end-of-passage rule.

2. **PCFG vs n-grams:** PCFG (when it parses) prefers short, PTB-like structure. N-grams catch local lexical implausibility and domain shift. Long literary sentences almost never get a PCFG verdict under our length cap — n-grams carry those cases.

3. **`p` and `N`:** Higher `p` increases SEGMENT-ALERT rate and gives Q1 real work; higher `N` delays GRAMMAR-ALERT but reduces trigger overhead. `p=0.08`, `N=5` is a practical balance.

4. **Sub-system interactions:** A successful SEGMENT split can turn an OOV merge into in-vocab words and suppress SPELL. Conversely, SPELL rewriting a rare in-context word (whale→while) can flip window perplexity and the Part-4 method. PCFG projection (`*`) means lexical identity was altered for parseability — another interaction to note when comparing methods.

## Streamlit

```bash
cd q4 && ../.venv/bin/streamlit run app.py
```

Modes: **Simulate passage** (auto stream with merges) and **Live typing** (incremental processing). After completion the UI shows the Part-4 table and latencies.

## How to reproduce

```bash
source ../.venv/bin/activate
python main.py train-lms
python main.py analyze --seed 42
python main.py analyze --seed 99
python main.py bench
streamlit run app.py
```

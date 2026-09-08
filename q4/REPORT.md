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
| Grammar z-threshold | **2.5** | Adaptive relative outlier detection (robust z-score via median + MAD); alerts only when a window's perplexity is >2.5×MAD above the passage's own running baseline. Avoids false alarms from Brown-vs-Gutenberg domain shift that a fixed absolute threshold (e.g. 5000) cannot. |
| Grammar warmup | **3 windows** | Minimum windows seen before z-score alerts fire, preventing spurious alerts on the first few windows of a passage |
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
- Grammar triggers fired only on genuinely anomalous windows (e.g. windows containing false splits or merge artifacts) via the adaptive z-score rule — ~4 alerts instead of ~13 with the old fixed threshold.
- Latency: ~0.08 ms/token seg+spell, ~0.06 ms/trigger grammar — far below typing delay.

## Sample run B (`--seed 7` / `99`, Chesterton / Moby Dick)

- Merges like `wasmaking`, `ofthe`, `wascarried`, `whofirst` split cleanly when both pieces were in the Q3 vocabulary.
- Conservative spell filters prevent harmful real-word cascades (`crab→grab`, `snout→shout`, `hatter→matter` all rejected); rare literary OOV words stay uncorrected rather than being wrongly rewritten.
- Binary-split fallback recovers merges that Q1 over-segments (`saidthank→said+thank`, `anychances→any+chances`).
- One shorter projected sentence received **pcfg / grammatical**; long sentences were `unparseable` (length cap or chart failure) and fell back to trigram.

## Speed Demon (1000 words)

| Path | Total | Per word |
|------|------:|--------:|
| Segmentation + spelling | 0.369 s | **0.369 ms** |
| Grammar-trigger only | 0.019 s | **0.019 ms** |
| Overhead (seg+spell − grammar) | — | **0.350 ms** |

**Conclusion:** The segmentation+spelling layer is cheap enough to keep on every token in live mode; throttling it to the grammar interval is unnecessary at ~0.4 ms/word.

## Comparative analysis

1. **Live alerts vs final verdict:** SEGMENT/SPELL alerts are local and usually agree with the corrected sentence used in Part 4. GRAMMAR-ALERT windows fire adaptively: the robust z-score approach tracks the passage's own perplexity baseline and only flags windows that are statistical outliers relative to that baseline, so normal literary text no longer triggers constant alerts.

2. **PCFG vs n-grams:** PCFG (when it parses) prefers short, PTB-like structure. N-grams catch local lexical implausibility and domain shift. Long literary sentences almost never get a PCFG verdict under our length cap — n-grams carry those cases.

3. **`p` and `N`:** Higher `p` increases SEGMENT-ALERT rate and gives Q1 real work; higher `N` delays GRAMMAR-ALERT but reduces trigger overhead. `p=0.08`, `N=5` is a practical balance.

4. **Sub-system interactions:** A successful SEGMENT split can turn an OOV merge into in-vocab words and suppress SPELL. Conversely, SPELL rewriting a rare in-context word (whale→while) can flip window perplexity and the Part-4 method. PCFG projection (`*`) means lexical identity was altered for parseability — another interaction to note when comparing methods.

## Known limitations

1. **Literary / archaic OOV words:** Words absent from the Brown corpus (e.g. `hatter`, `dormouse`, `besought`) are treated as OOV by the segmenter. The editor applies conservative filters — rejecting spell suggestions that change the first letter or lack a shared prefix (so `hatter→matter` is rejected) and rejecting splits into 2-letter junk fragments (so `jackal→jack+al` is rejected). The tradeoff is that some genuine OOV words stay uncorrected, which is safer for a live editor than aggressive auto-rewriting.

2. **Over-segmentation of morphologically complex words:** Words like `besought` (be+sought) or `faithless` (faith+less) that are OOV in Brown but whose substrings are valid words may be incorrectly split. This is inherent to the spacebar-miss detection model — the editor cannot distinguish `besought` (one word) from `himout` (two merged words) without a larger lexicon.

3. **Binary split fallback:** When Q1's joint beam decoder over-segments (e.g. `saidthank→said+than+k`), Q4 falls back to a binary cut search that prefers a single split into two in-vocab words. This recovers `marchhare→march+hare`, `saidthank→said+thank`, `anychances→any+chances`, and `asit's→as+it's`, but cannot help when one piece is genuinely absent from Brown (e.g. `hare` is in Brown, but a hypothetical OOV piece would block the split).

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

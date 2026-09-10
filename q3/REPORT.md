# Question 3 — Report

A spelling corrector built on the NLTK Brown corpus, handling both non-word and
real-word errors within an edit distance of 1, with two candidate-generation
algorithms benchmarked against each other and deployed behind an interactive CLI.

All numbers below were produced by the scripts in this directory; the raw output
is in `results/`.

---

## Part 1 — Corpus and model preparation

Brown is split **90/10 by sentence** (seed 42) before anything is counted, so the
Part 4 accuracy figures are measured on sentences the vocabulary and the language
models have never seen.

| | |
|---|---|
| Brown sentences | 57,340 → 51,606 train / 5,734 held out |
| Training tokens | 901,816 |
| Vocabulary types | 40,542 |
| Distinct bigrams | 385,197 |
| Model file | 7.98 MB pickle, loads in ~270 ms |

**Tokenisation.** Tokens are lower-cased and kept if they match
`[a-z]+('[a-z]+)?`, so contractions and possessives survive as single words.
Anything containing a digit collapses to a `<num>` placeholder — it should occupy
a slot in the bigram model but must never be offered as a correction, so `<num>`
is in the counts but not in the vocabulary. Pure punctuation is dropped.

The edit alphabet is the 26 letters **plus the apostrophe**, which makes
`dont → don't` reachable in one edit. It costs one extra letter's worth of work in
Method A and nothing at all in Method B.

**Unigram model.** `P(w) = C(w)/N`. This is the ranking signal for non-word errors.

**Bigram model.** Add-k smoothed, `k = 0.01`, with `<s>`/`</s>` padding:

```
P(w2 | w1) = (C(w1,w2) + k) / (C(w1) + k·|V|)
```

The model also exposes a **Jelinek–Mercer interpolated** estimate,
`P(w2|w1) = λ·P_ML(w2|w1) + (1−λ)·P(w2)` with `λ = 0.85`. Add-k is the model the
brief asks for and is what is reported; interpolation is what the *decision rule*
for real-word errors actually uses, and §3 gives the measurement that justifies
that choice.

---

## Part 2 — Candidate generation

Both methods answer the same question — *which vocabulary words are within one
edit of this string?* — and both are exposed through the same `candidates(word)`
API so they are interchangeable everywhere.

### Method A — standard edit-distance-1 generation

Build every deletion, transposition, replacement and insertion of the query, then
intersect with the vocabulary. For a word of length *n* over an alphabet *A* that
is `n + (n−1) + n|A| + (n+1)|A|` strings — **≈ 2n|A|**, so the alphabet is a
multiplier on *every single lookup*.

### Method B — symmetric delete (SymSpell)

*Preprocessing:* map every one-character deletion of every vocabulary word, and
each word itself, back to that word. 40,542 words → **313,857 keys / 348,252
postings**, built in ~142 ms.

*Query:* build only the query's own deletions plus the query itself — **n + 1**
strings — and look them up.

One deletion on each side covers all four edit operations:

| edit in the query | why the buckets meet |
|---|---|
| substitution | delete the changed position on both sides → equal strings |
| transposition | delete one of the two swapped characters on each side |
| insertion | the query's deletion *is* the dictionary word |
| deletion | the query *is* one of the dictionary word's deletions |

**Verification matters.** SymSpell buckets collide at edit distance 2 — `abc` and
`axb` both delete to `ab` — so the raw lookup returns a *superset*. A linear
Damerau–Levenshtein ≤ 1 check filters it. With verification the two methods return
**byte-identical sets**: 0 disagreements on 1,000 benchmark words and 0 across the
randomised equivalence tests in `tests/test_spelling.py`. That equivalence is what
makes the Part 4 speed comparison a fair one — the two methods differ only in
runtime, never in output.

---

## Part 3 — Correction logic

### Non-word errors

Token not in the vocabulary → generate candidates with Method A and Method B, take
the union, return the one with the highest unigram probability. If nothing sits
within one edit the word is returned unchanged rather than guessed at, since the
brief scopes this corrector to edit distance 1.

### Real-word errors

The word is legal but wrong in context, so frequency is useless — *sea* is a
perfectly common word. Each candidate is scored on the local phrase against both
neighbours,

```
score(x) = log P(x | prev) + log P(next | x)
```

and a correction is proposed only when it beats the typed word by a margin:

```
score(best) − score(typed)  ≥  threshold      (default 5.0 nats)
```

That threshold is the "significantly higher probability" the brief asks for, and
it has a reading as a noisy-channel prior: it is `log((1−ε)·N/ε)` for a typing
error rate ε over roughly N edit-1 neighbours. Words shorter than 3 characters are
never second-guessed — *a*, *of*, *to* have dozens of neighbours and almost no
signal.

### Why interpolated smoothing for the decision, add-k for the model

Add-k is a fine probability model but a poor *decision* rule here. With |V| ≈ 40k
the `k·|V|` mass dominates the denominator, so the gap between "seen once" and
"never seen" becomes an arbitrary constant that swamps the real evidence. Backing
off to the unigram degrades gracefully instead. Measured on the real-word test set
at a matched threshold of 5.0:

| scoring | β | real-word accuracy | false alarms /word |
|---|---|---|---|
| add-k | 0.0 | 58.01 % | 2.41 % |
| **interpolated** | **0.0** | **61.37 %** | **1.96 %** |
| interpolated | 0.5 | 53.88 % | 1.48 % |
| interpolated | 1.0 | 36.46 % | 1.49 % |

Interpolation is better on *both* axes, so it is the default. (β is the
`unigram_discount` knob — see §5.)

---

## Part 4 — Evaluation

### Test-set construction

For each of the 5,734 held-out sentences one word (length ≥ 3, in vocabulary) is
picked at random and corrupted **twice** with a single edit — once into a string
outside the vocabulary, once into a *different* string inside it. Sentences where
both corruptions exist for the same word are kept, giving **5,620 aligned pairs**:
the two test sets share the same sentences and the same target positions, so the
two accuracy numbers are directly comparable.

```
gold 'the'    -> non-word 'thle'   | real-word 'thet'
gold 'never'  -> non-word 'neger'  | real-word 'neveh'
```

### Accuracy

*target acc* = gold word restored at the corrupted position; *sent acc* = whole
sentence restored exactly; *collateral* = sentences where a word other than the
corrupted one was changed.

| test set | n | target acc | sent acc | no change | wrong fix | collateral |
|---|---|---|---|---|---|---|
| non-word only | 5,620 | **91.96 %** | 80.64 % | 0 | 452 | 680 |
| non-word + real-word check | 5,620 | 91.96 % | 61.99 % | 0 | 452 | 1,810 |
| real-word | 5,620 | **61.37 %** | 42.21 % | 1,694 | 477 | 1,821 |
| real-word, context off | 5,620 | 0.00 % | 0.00 % | 5,620 | 0 | 680 |

Reading these:

- **Non-word 91.96 %.** The corrector always finds *something* — "no change" is 0
  — so every one of the 452 errors is a wrong choice among real candidates, not a
  failure to generate. The residual errors are almost entirely short function
  words where several high-frequency candidates are one edit apart: `eis → is`
  (gold *his*), `hed → he` (gold *had*), `thez → the` (gold *they*). Unigram
  frequency simply cannot separate these; the context that would is exactly what
  the non-word path is specified not to use.
- **Real-word 61.37 %**, against a floor of **0 %**. This is the number that shows
  the bigram model earning its place: without context checking not a single
  real-word error is caught, because every corrupted token is a legal word.
  Of the 39 % missed, 1,694 (30 %) are cases where nothing beat the threshold and
  477 (8 %) are cases where the wrong candidate won.
- **Turning real-word checking on does not cost non-word accuracy** (91.96 % in
  both rows) but it drops *sentence* accuracy from 80.6 % to 62.0 %, because
  collateral rewrites elsewhere in the sentence rise from 680 to 1,810. That is
  the real price of real-word correction, and it is invisible if you only look at
  the target-position number.

### The false-alarm cost, and the threshold trade-off

Run over 1,000 **uncorrupted** sentences (17,686 words), the corrector at
threshold 5.0 rewrites 346 in-vocabulary words — **1.96 % of words, touching
27.5 % of sentences**. (A further 156 rewrites are of out-of-vocabulary words the
training split never saw; those are the corrector doing its job, not false alarms,
and are counted separately.)

There is no free lunch here, only an operating point:

| threshold | real-word accuracy | false alarms /word | sentences damaged |
|---|---|---|---|
| 1 | 74.29 % | 8.74 % | 70.0 % |
| 2 | 71.35 % | 5.54 % | 55.2 % |
| 3 | 68.67 % | 3.95 % | 45.5 % |
| 4 | 65.53 % | 2.82 % | 36.6 % |
| **5** | **61.37 %** | **1.96 %** | **27.5 %** |
| 6 | 56.90 % | 1.32 % | 20.0 % |
| 8 | 46.85 % | 0.61 % | 10.0 % |
| 10 | 35.98 % | 0.36 % | 6.2 % |

5.0 is the default: it is roughly the knee, keeping most of the achievable recall
while holding damage to clean text under 2 % of words. A product would sit higher
— or, better, surface real-word findings as *suggestions* rather than applying
them, which is why the CLI colours them differently and prints the margin.

Two other levers were measured and rejected. **Protecting the N most frequent
words** from real-word correction (N = 100/300/1000) is dominated at every matched
false-alarm rate — e.g. at ≈2.0 %/word, no protection gives 61.7 % where N = 1000
gives ≈54 %. Raising **minimum word length** from 3 to 4 costs 13 accuracy points
for a smaller false-alarm saving than simply raising the threshold. Both remain
available as constructor arguments.

### The residual errors are systematic, not random

Top residual confusions from `results/results.json` → `evaluation.residual_errors`
(real-word path):

| gold | typed | predicted | count |
|---|---|---|---|
| the | he | he | 31 |
| the | they | they | 10 |
| the | them | them | 9 |
| and | an | an | 9 |
| this | his | his | 8 |
| problem | problems | problems | 5 |
| program | programs | programs | 5 |

Two clusters. The first is closed-class function words, where the corrupted form
is itself extremely frequent and fits almost any context — a bigram window is too
narrow to tell *the* from *they*. The second is **singular/plural inflection**,
where both forms are grammatical and only agreement with a verb several tokens
away would settle it. Neither is reachable with a bigram model; both would need
either a trigram/neural LM or explicit morphological features.

### "Speed Demon" benchmark

One identical batch of **exactly 1,000** non-word misspellings (mean length 5.83),
best of 3 runs, non-word path only. Numbers from `results/results.json` on this
machine (exact × varies slightly by hardware; the structural conclusion does not):

| method | total | per word | words/sec | candidates |
|---|---|---|---|---|
| Method A — edit-distance-1 | 0.0618 s | 0.0618 ms | 16,174 | 3,166 |
| **Method B — SymSpell + verify** | **0.0067 s** | **0.0067 ms** | **148,603** | 3,166 |
| Method B — SymSpell, raw buckets | 0.0025 s | 0.0025 ms | 394,496 | 6,461 |

Preprocessing: Method A ~0 ms, Method B ~248 ms.
Output equivalence: **0 disagreements on 1,000 words.**

#### Conclusion — why Method B achieves its runtime

Method B is **~9.2× faster per word**, and the reason is structural rather than a
constant-factor implementation win.

Method A's cost is tied to the **size of the alphabet**. Every lookup materialises
≈ 2n|A| strings — with |A| = 27 and mean length 5.83 that is **352,312 strings for
this batch**. Each one is allocated, hashed and probed against the vocabulary set,
and that allocate-and-hash loop is where essentially all the time goes.

Method B moves that work off the query path and into a one-off preprocessing step.
Because one deletion on each side covers all four edit operations, the index only
needs each word plus its *n* deletions. At query time **the alphabet has vanished
from the complexity**: only the n+1 probe strings are built — **6,827 for the whole
batch, 51.6× fewer** — so a lookup is O(n) hash probes instead of O(n|A|). That is
the entire trick: the alphabet factor is paid once at build time instead of on
every lookup. The measured ~9× is smaller than the 51.6× reduction in strings
built, because Method B's remaining time is dominated by the verification pass and
the set operations rather than by string construction.

Verification is worth its cost. Raw bucket lookup is ~24× faster than Method A;
adding the exact edit-distance check brings that to ~9× — and buys correctness,
since the raw buckets return 6,461 candidates against the true 3,166, a **2.04×
over-generation** that would otherwise feed junk to the unigram ranker.

The trade is memory and start-up time for query latency. The ~248 ms index build
pays for itself after ≈ **4,500 lookups**. For a one-shot correction Method A is fine; for anything interactive or
batched, Method B is the only sensible choice.

---

## Part 5 — Live interactive application

`python main.py app` is a continuous REPL: prompt, correct, highlight, report latency,
`exit` to stop. Non-word fixes are green and real-word fixes yellow (or
`**asterisks**` with `--plain`), each listed underneath with its kind and, for
real-word fixes, the log-probability margin that justified it. `:realword`,
`:method` and `:threshold` change behaviour without restarting, which is how most
of the tuning above was explored.

Measured latency is **~0.4–1.5 ms per sentence** — three orders of magnitude below the
threshold of noticeability, which is a direct consequence of the Method B index.

Non-interactive captures of the brief’s four sentences and additional demos are
saved in `results/demo_outputs.json` (`python main.py demo`).

```
you > This is a test sentnce.
fix > This is a test sentence.
      non-word   sentnce -> sentence
      latency 0.32 ms

you > I would like to sea the world.
fix > I would like to see the world.
      real-word  sea -> see (+7.25 nats)
      latency 0.59 ms
```

### The brief's four test sentences

Archived in `results/demo_outputs.json` → `brief_sentences`:

| input | output | verdict |
|---|---|---|
| I **hav** a good feeling about this. | I **had** a good feeling about this. | plausible; see below |
| This is a test **sentnce**. | This is a test **sentence**. | correct |
| I would like to **sea** the world. | I would like to **see** the world. | correct |
| Please **meat** me at the station. | Please **beat** me at the station. | wrong |

And ten of our own (`python main.py demo` → `own_sentences`):

```
The govenment anounced a new policey yesterday.
  -> The government announced a new policy yesterday.       3/3 non-word

Thier answer was completly diffrent from ours.
  -> Their answer was completely different from ours.       3/3 non-word

I can not beleive how quikly the yeer went.
  -> I can not believe how quickly the year went.           3/3 non-word

The whether was terrible during the hole trip.
  -> The whether was terrible during the whole trip.        hole->whole caught;
                                                            whether->weather is
                                                            edit distance 2

He wants to by a peace of cake.
  -> He wants to by a peace of case.                        false alarm on cake
```

### Where it fails, and why

- **`hav → had`, not `have`.** Both are one edit away; Brown has *had* 4,602 times
  against *have* 3,519. Ranking non-word candidates by unigram frequency is exactly
  what the brief specifies, and this is that rule working as designed. Adding the
  same bigram context used for real-word errors would fix it — an easy extension,
  deliberately not taken, so the two error paths stay cleanly separated.

- **`meat → beat`, not `meet`.** Neither `please meet` nor `please beat` occurs in
  Brown, so the whole decision rests on the right-hand bigram — and `C(beat, me) =
  2` against `C(meet, me) = 1`. A single count decides it. This is data sparsity in
  a 900k-token corpus, not a flaw in the scoring rule; a larger corpus or a
  trigram window would settle it.

- **`fox → for`, `cake → case`.** When neither bigram is attested, the interpolated
  score collapses to a comparison of unigram frequencies, and the more common word
  wins — *brown fox* is absent from Brown, so *for* (8,616) beats *fox* (9).
  Setting `unigram_discount = 1.0` turns the score into pure PMI, which makes an
  unseen bigram score identically for every candidate (exactly `log(1−λ)`) and
  removes this failure mode entirely. It is **not** the default because it scores
  much worse on the benchmark (36.5 % vs 61.4 %) — and that gap is itself a finding
  about the *test set*, not about the model: corrupting a randomly chosen token
  almost always makes it **rarer**, since tokens are frequency-weighted by usage
  but their edit-neighbours are not. A frequency prior therefore gets rewarded by
  the synthetic benchmark in a way it would not be by real typing errors. Both
  settings ship; β is a constructor argument.

- **`whether → weather` missed — and correctly so.** This one is not a scoring
  failure: *whether* → *weather* takes **two** substitutions, so it is outside the
  edit-distance-1 scope the brief defines and no candidate generator ever proposes
  it. The context evidence is there and would have been decisive
  (`C(the, weather) = 12` and `C(weather, was) = 3` against 0 for both *whether*
  bigrams) — the candidate simply never reaches the scorer. This is the sharpest
  illustration of the edit-distance-1 ceiling: roughly a quarter of real
  misspellings need distance 2, and none of them are reachable here at any
  threshold.

---

## Summary

| | |
|---|---|
| Non-word accuracy | **91.96 %** (0 % of cases left uncorrected) |
| Real-word accuracy | **61.37 %**, against a 0 % floor without context |
| Real-word false-alarm rate | 1.96 % of words at the default threshold |
| Method B speed-up | **~9.2×** per word (this machine), identical output, ~248 ms one-off index |
| Interactive latency | ~0.4–1.5 ms per sentence |
| Tests | 49 passing, including randomised A ≡ B equivalence |

The two headline results are that **SymSpell buys roughly an order of magnitude of latency
for free** — the candidate sets are provably identical, so the only cost is ~250 ms
and some memory at start-up — and that **real-word correction is a genuine
trade-off rather than a strict improvement**: it lifts real-word accuracy from 0 %
to 61 % but rewrites ~2 % of already-correct words, and that cost is only visible
if you measure sentence-level accuracy and false alarms alongside the headline
number.

## Submission checklist (PDF marking scheme)

| Part | Marks | Evidence |
|------|------:|----------|
| 1 Vocabulary + unigram | 3 | `spelling/language_model.py`, `python main.py train` |
| 1 Bigram LM | 3 | add-k bigram in same module; trained pickle `artifacts/brown_lm.pkl` |
| 2 Method A | 5 | `EditDistance1Generator` in `spelling/candidates.py` |
| 2 Method B (+ preprocess) | 5 | `SymmetricDeleteIndex` in `spelling/candidates.py` |
| 3 Non-word correction | 4 | `SpellingCorrector.correct_non_word` (unigram rank) |
| 3 Real-word correction | 4 | bigram context + threshold 5.0 in `corrector.py` |
| 4 Test sets + accuracy | 4 | `python main.py eval --full` → `results/results.json` |
| 4 Speed Demon + conclusion | 4 | `python main.py bench` + §4 conclusion above |
| 5 Interactive Terminal CLI | 8 | `python main.py app` (highlight, latency, `exit`); samples in `results/demo_outputs.json` |

Reproduce:

```bash
cd q3
python main.py eval --full
python main.py bench
python main.py demo
python main.py app
python -m pytest -q
```

# Phase 4 Task 4 — semantic-cache threshold validation (preregistration)

This file, `phase4_handbuilt_pairs.json` and `phase4_threshold_validation.py`
are committed **before any similarity has been computed on any of these
pairs**. Nothing below may be changed after scoring begins; a deviation must be
recorded as a deviation, not edited in.

## Question

Is there a cosine-similarity threshold at which the semantic tier serves
reusable answers for true paraphrases while almost never serving a wrong answer
for prompts that are not equivalent?

## Embedder

The production class `SentenceTransformerEmbedder`
(`sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU, normalized, 384 dimensions).
Similarity is the dot product of the normalized vectors, which equals cosine
similarity. No LLM or provider calls are made.

## Data

| Source | Positives (label 1) | Negatives (label 0) | Role |
|---|---|---|---|
| GLUE QQP, `validation` split | 600 duplicate pairs | 600 non-duplicate pairs | General paraphrase / non-paraphrase |
| PAWS-Wiki `labeled_final`, `validation` split | 400 paraphrase pairs | 400 non-paraphrase pairs | Adversarial: high word overlap, different meaning |
| Hand-built LLM-style prompts (`phase4_handbuilt_pairs.json`) | 40 | 60 | Adversarial: negation, numbers, direction, entity, language/library, near-miss |

- Sampling: `random.Random(20261019)`, within each (source, label) stratum,
  after sorting the source rows by their dataset order. Counts as above.
- Dev/test split: within each (source, label) stratum, the sampled rows are
  shuffled with the same seeded generator and the first half is **dev**, the
  second half **test**. The hand-built set uses the split stored in its file
  (alternating within each label/category group).
- The **adversarial** stratum is PAWS plus hand-built. The expected test
  negative counts are about 300 QQP and about 230 adversarial (200 PAWS + 30
  hand-built), i.e. at least 500 negatives of which at least 200 adversarial.
- Test pairs are never used to choose the threshold.

### Labeling guidance (hand-built set)

A pair is **positive** only if a single stored answer would be fully acceptable
for both prompts. It is **negative** if reusing the stored answer for the second
prompt would be wrong, misleading, or answer a different question (changed
negation, number, unit, direction, entity, language, library, task, or scope).

## Metrics

For a threshold t, a pair is a **hit** if similarity >= t.

- **False-positive rate (FPR)** = fraction of negative pairs that are hits. A
  false positive is a semantic hit between prompts that are not equivalent for
  the purpose of reusing a response.
- **Paraphrase hit rate** = fraction of positive pairs that are hits.
- All proportions are reported with Wilson 95% intervals.

## Decision rule (plain arm — the headline)

1. Grid: t in {0.50, 0.51, ..., 0.99}.
2. **Chosen threshold t\*** = the smallest t in the grid whose **dev adversarial
   FPR** is <= 1%. If none exists, there is no threshold and the semantic tier
   ships disabled.
3. Report, on **test** at t\*: adversarial FPR, QQP-negative FPR, pooled FPR,
   and the paraphrase hit rate for QQP, PAWS, hand-built and pooled.
4. The semantic tier may be enabled **by default** only if both hold on test:
   (a) the Wilson 95% **upper** bound of the adversarial FPR is <= 2%
   (with about 230 adversarial test negatives this requires zero false
   positives), and (b) the pooled paraphrase hit rate is >= 20% (a threshold
   that is safe only because it almost never hits is not useful).
5. Otherwise the tier ships disabled by default, and the documentation states
   the measured FPR and hit rate for anyone who enables it explicitly.

## Guard arm (only meaningful if the plain arm fails the rule)

A pair also needs to pass a surface guard to count as a hit: the lists of
numbers in the two prompts are identical, and the number of negation words
(`not`, `no`, `never`, `cannot`, `without`, `none`, `n't`) has the same
parity. The guard arm is scored with the same grid and the same rule, is
reported **separately**, and never replaces the plain arm in the headline.
Its result is only a pointer to a possible follow-up, not a shipped feature.

## Known limits (stated in advance)

- QQP and PAWS are questions and encyclopedic sentences, not LLM prompts; the
  hand-built set is small (100 pairs), so it cannot support a tight bound by
  itself.
- This is a pairwise evaluation. The deployed cache compares a query against
  every stored entry, so more entries mean more chances for a false near
  neighbour. Pairwise FPR therefore understates deployed risk.
- Single-turn pairs only; the deployed tier also requires identical context and
  parameters through TAG filters, which this experiment does not exercise.
- One embedding model, one run, one seed.
- Hand-built negatives share some base prompts with positives; dev/test
  leakage through shared wording is possible and is a stated limitation.

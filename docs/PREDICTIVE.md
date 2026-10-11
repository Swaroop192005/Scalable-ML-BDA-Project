# Does any of this predict anything?

Mining rules proves a pattern *existed*. It does not prove the pattern
*predicts*. This page covers the parts of the project that test that claim,
and the parts that make the statistics hold up under the volume of
hypotheses rule mining generates.

---

## Held-out evaluation

### The protocol

**Temporal split, not random.** Transactions are split by date: everything
before the cut trains, everything after tests. A random split would let a
model learn from December while being scored on June — leakage, because in
production you only ever have the past. It is also the harsher test, since
tastes and stock drift across the gap.

**Rules are mined on the training split only.** `AssociationRecommender`
re-mines inside `fit()`. Handing it a rules table mined over the whole
dataset would quietly leak test-period co-occurrences into the model and
inflate every number below.

**Leave-one-out within each test basket.** For a test basket of *n* items,
one is hidden and the other *n−1* shown. A hit is the hidden item appearing
in the top-*k*. Baskets with fewer than two items are skipped, and held-out
items never seen in training are excluded — no model can recover those, so
counting them would just subtract a constant from every score.

### Results

Trained on 13,527 baskets (2010-12-01 → 2011-10-11), scored on 4,471
leave-one-out cases from the following two months.

| model | hit@1 | hit@5 | hit@10 | MAP@5 | coverage@5 | novelty@5 | vs popularity @5 |
|---|---|---|---|---|---|---|---|
| **association** | **6.13%** | **13.40%** | **17.69%** | **0.086** | 13.3% | 4.54 | **9.1×** |
| cooccurrence | 2.82% | 8.52% | 12.66% | 0.049 | 54.2% | 9.65 | 5.8× |
| popularity | 0.40% | 1.48% | 2.80% | 0.008 | 0.4% | 3.24 | 1.0× |
| random | 0.02% | 0.13% | 0.38% | 0.001 | 99.7% | 8.87 | 0.09× |

### Reading them honestly

**The association model wins, and the margin is real.** 9.1× the popularity
baseline at k=5, and 1.6× raw co-occurrence. That second comparison is the
one that justifies the mining pipeline: if support thresholds, confidence
and lift could not beat simply counting pairs, they would be cost without
benefit. They beat it.

**But it is the narrowest model in the table.** Coverage of 13.3% against
co-occurrence's 54.2%, and novelty 4.54 against 9.65. The association model
is accurate partly by concentrating on a well-trodden slice of the
catalogue. A recommender that is right more often while surfacing less of
the range is a real trade-off, not an unqualified win, and anyone deploying
this should decide which they want.

**Popularity is the baseline that matters.** 1.48% at k=5 sounds dismissible
until you notice it needs no model at all. Any recommender that cannot
clearly beat a bestseller list is not earning its complexity. Reporting an
absolute hit-rate without this column would be close to meaningless.

**Why coverage and novelty are first-class metrics here.** A model can top
the accuracy column while only ever recommending what the customer would
have bought anyway. Popularity's 0.4% coverage is exactly that failure mode
made visible: accurate-ish, and commercially inert.

---

## Multiple-testing correction

A chi-squared independence test is run per rule. With several hundred rules
tested at p < 0.05, roughly 5% clear the bar by chance — so an uncorrected
"significant" flag does not mean what it appears to.

Bonferroni would fix it and is far too conservative here: the tests are
positively dependent (rules share items) and it would discard most genuine
findings. **Benjamini-Hochberg** instead controls the *false discovery
rate* — the expected share of false positives among the rules declared
significant. For p-values sorted ascending with ranks i = 1..m:

```
q(i) = min over k >= i of  ( p(k) * m / k )
```

The running minimum from the largest rank downward enforces monotonicity, so
no rule is reported as more significant than a worse-ranked one. The
`not_significant` flag now reads the adjusted q-value, not the raw p-value.

Verified in `tests/test_rules.py` against Benjamini & Hochberg's own 1995
worked example, plus order-invariance, monotonicity and the `q >= p` bound.

---

## Compact representations

Two lossless summaries of the frequent itemset lattice:

- **Closed** — no proper superset has the same support. Lossless *for
  support*: every frequent itemset's exact count is recoverable as the
  maximum support among its closed supersets.
- **Maximal** — no proper superset is frequent at all. Much smaller, but
  preserves only the frequent/infrequent boundary.

`maximal ⊆ closed ⊆ frequent`, always.

### The finding: closure does nothing on retail data

| dataset | frequent | closed | maximal |
|---|---|---|---|
| Online Retail @ 1.5% support | 628 | **628** (1.0×) | 491 (1.3×) |
| dense synthetic | 38 | **4** (9.5×) | 3 (12.7×) |

Closure gives **zero compression** on the real data. That is not a bug — the
implementation compresses 9.5× on dense input, and `mba compact --verify`
confirms all 628 supports are recoverable from the closed set. It is a
property of sparse baskets: closure only collapses an itemset into a
superset when the two have *identical* support, which requires every basket
containing the subset to also contain the superset. In retail, with
thousands of products and an average basket of 24, that essentially never
happens.

The lesson is that compact representations are a dense-data technique.
Quoting a compression ratio without saying which regime it came from would
be misleading.

---

## Streaming

Every other miner here assumes the data is finite and re-readable. A live
checkout feed is neither, and exact counting needs one counter per distinct
item — unbounded memory.

**Lossy Counting** (Manku & Motwani, 2002) trades a *bounded, known* amount
of accuracy for bounded memory in one pass. Measured on the real data at
support 2%, ε = 0.002:

| guarantee | expected | observed |
|---|---|---|
| No false negatives | 0 | **0** |
| Counts never overestimate | False | **False** |
| Undercount ≤ ε·N | ≤ 36 | **18** |
| Table within space bound | ≤ 42,882 | **2,676** |

### One correction worth recording

The classical space bound, `(1/ε)·log(ε·N)`, **assumes one item per
observation.** The itemset variant buckets on *transactions*, and a basket
inserts ~24 items between prunes, so the table legitimately exceeds that
figure — measured 2,656 against a classical bound of 1,792. Reporting the
unscaled bound as a guarantee would have been a false claim. `StreamingStats`
therefore exposes `classical_space_bound` and the scaled `space_bound`
separately, and the tests assert the classical one only for a genuine
single-item stream.

A related bug, caught in testing and worth noting because it is subtle:
bucketing on item *occurrences* rather than transactions puts the support
threshold and error bound on a scale ~24× larger than the counts they are
compared against, and **nothing is ever reported.** `tests/test_streaming.py`
has a regression test for it.

---

## Customer segmentation

RFM (recency, frequency, monetary) plus k-means with k-means++ seeding,
implemented from scratch — adding scikit-learn for one k-means would be out
of proportion to the rest of the dependency list.

Features are **log-transformed then z-scored**. Monetary and frequency span
four orders of magnitude, and on raw values Euclidean distance is decided
entirely by a handful of wholesale accounts, which returns one cluster of
three customers and one of everybody else.

| segment | customers | share | median recency | median frequency | median spend | revenue share |
|---|---|---|---|---|---|---|
| champions | 684 | 16% | 8 days | 9 orders | £3,652 | **64%** |
| at risk | 1,148 | 27% | 51 days | 4 orders | £1,332 | 24% |
| loyal | 822 | 19% | 19 days | 2 orders | £423 | 5% |
| one-time buyers | 1,577 | 37% | 178 days | 1 order | £302 | 7% |

**16% of customers produce 64% of revenue.** And the segments shop
differently: the *loyal* segment's strongest rules are Christmas
decorations, while champions and one-time buyers are dominated by the
Regency teacup range — a difference a single global rules table averages
away entirely.

### An honest note on k

Silhouette is highest at k=2 (0.43) and declines as k rises (0.34 at k=4).
RFM is continuous rather than naturally clustered, so there is no sharp
elbow. k=4 is chosen for business readability, not because the data demands
it — `mba segment --choose-k` prints the sweep so the choice is visible
rather than assumed.

---

## Temporal drift

Each period is mined independently at the same relative support, so a rule's
lift in March comes only from March's baskets. Each rule is then classified:

| class | meaning | count (monthly, 3% support) |
|---|---|---|
| stable | holds throughout with little variation | 11 |
| emerging | lift trending clearly upward | 13 |
| decaying | lift trending clearly downward | 20 |
| seasonal | high lift variance across periods | **0** |
| intermittent | present part of the span, steady when present | 120 |
| sporadic | too few periods to call | 680 |

### The finding: affinities are not seasonal, volume is

**No rule is statistically seasonal.** The highest observed lift coefficient
of variation is 0.32, against a 0.60 threshold. Meanwhile basket volume
swings hard — 968 baskets in January against 2,524 in November.

So in this dataset, Christmas changes *how much* people buy, not *what goes
with what*. The threshold was left at 0.60 rather than lowered to manufacture
seasonal labels; the config carries a note saying so.

### Two classifier corrections

**Trend needs R², not just slope.** An alternating series like
`[1, 12, 1, 12, 1, 12]` has a large OLS slope purely because it ends high,
and was being reported as "emerging" — the opposite of the truth. A trend
label now also requires the linear fit to explain the variation (R² ≥ 0.5).
This moved 4 rules out of "emerging" on the real data.

**Low presence is not seasonality.** A rule appearing in 3 of 13 months at
*steady* lift is sitting near the support threshold, not coming and going
with the season. Those are now `intermittent`; calling them seasonal would
invite a campaign built on nothing.

**Partial periods are flagged.** The data ends 9 December, so December 2011
covers 9 days. Unflagged, a short final period reads as collapsing demand
and drags on every trend fit.

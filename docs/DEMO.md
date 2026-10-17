# Presentation demo runbook

A 12-minute live demo, ordered so each step sets up the next. Every command
is copy-pasteable and timed, and each one has a line on *what to say* —
because the commands are not the point, the findings are.

---

## Before you present — do this, it matters

Two things in this project are slow on a cold start and will stall a live
demo:

```bash
# 1. Warm the dashboard cache (first page load mines from scratch, ~30s)
mba dashboard
#    open localhost:8501, click through every tab once, leave it running

# 2. Pre-compute the slow analyses so the CSVs already exist
mba run --benchmark --scaling
mba recommend
mba segment -k 4
mba temporal -f M
```

Budget ~8 minutes for all of that. Do it before the room fills.

Then check you are ready:

```bash
pytest                                  # expect 269 passed
ls reports/ | wc -l                     # expect 20+ files
```

Two windows open: a **terminal**, and a **browser on localhost:8501**.

> If the network is unreliable in the room, everything below also works on
> the committed sample with `-i data/sample/sample_retail.csv`. The numbers
> change; nothing breaks.

---

## The demo

### 0 · Frame the problem (30 seconds, no commands)

> "Market basket analysis finds which products sell together. The naive
> version — count pairs, sort by how often they co-occur — produces a table
> that looks impressive and is almost entirely useless. This project is
> about the gap between those two things."

Keep this short. The demo makes the argument.

---

### 1 · The data is dirty in ways that corrupt the answer (1 min)

```bash
mba eda -q | head -30
```

**Say:** 541,909 raw rows → 431,541 clean, 79.6% retained. Then point at the
audit table and pick the one line that matters:

> "3,088 rows are postage, bank charges and manual adjustments. Those appear
> in a large share of invoices, so they co-occur with *everything*. Leave
> them in and they dominate your top rules with nothing you can act on. The
> cleaning isn't hygiene — it decides whether the output means anything."

Also worth a sentence: cancellations (9,288 rows, `InvoiceNo` starting `C`)
carry negative quantities and would create phantom baskets.

---

### 2 · Sparse encoding (45 seconds)

The encode figures are an INFO log, so drop the `-q` and grep for them:

```bash
mba eda 2>&1 | grep encoded
```

```
encoded 18,022 baskets x 3,355 items | density 0.7117% | 2.2 MB sparse vs 60.5 MB dense (27x)
```

Or, nicer on a projector: the **Overview** tab's "Sparsity" row shows the
same four numbers as large stat tiles.

> "18,022 × 3,355 is 60 million cells, 99.3% of them zero. A `pivot_table`
> would build that densely. We go straight to a sparse CSR matrix by integer
> factorisation — 2.2 MB, 27× smaller, and no dense version exists at any
> point. Memory scales with basket *lines*, not baskets × products."

---

### 3 · Mine the rules — and show the trap (2 min) ⭐

```bash
mba mine -s 0.02 --min-confidence 0.3 --min-lift 1.2 --top 6 -q
```

Two tables print. **This is the most important moment in the demo.**

First, point at the flagged table:

> "These are the *highest-lift rules in the entire dataset*. Pink Regency
> teacup implies green Regency teacup, 19.7× lift, 71% confidence, 461
> baskets. Statistically impeccable. And completely worthless — it says
> customers who buy the pink teacup also buy the green one, which describes
> the product range, not customer behaviour. Any pipeline that ranks on lift
> puts this at the top."

Then the high-value table:

> "These survived seven quality checks. 45 of 130 rules were flagged — every
> one of them a colour variant."

If you have one extra sentence, name the checks that matter: rare pairs,
near-universal consequents, redundancy, and a chi-squared independence test
**corrected for multiple testing** — because testing hundreds of rules at
p < 0.05 means ~5% pass by chance.

---

### 4 · Apriori vs FP-Growth: the headline finding (2 min) ⭐

Open `reports/figures/09_benchmark_apriori_vs_fpgrowth.png`.

```bash
open reports/figures/09_benchmark_apriori_vs_fpgrowth.png     # macOS
```

> "Both algorithms return **identical itemsets** at every threshold — we
> assert that in the test suite. Only cost differs, and the textbook answer
> 'FP-Growth is faster' turns out to be wrong at the top of this chart.
> At 5% support Apriori wins by 14×. The crossover is at 0.012."

Then the lower panel, which is the actual explanation:

> "As support falls from 5% to 0.8%, Apriori's candidate count grows 114× —
> 3,761 to 430,492. FP-Growth's grows 1.15×. Apriori pays for candidate
> generation; FP-Growth generates no candidates at all. So the honest claim
> isn't that FP-Growth is faster — it's that **its cost is flat in the thing
> that explodes**, which is why it's the right default as catalogues grow."

If you have time, run it live at two thresholds to show it reproducing:

```bash
mba bench -g 0.05 0.02 0.01 -q
```

---

### 5 · Beyond RAM (1 min)

```bash
mba ooc --synthetic 1000000 --chunk-size 100000 --memory-budget 800 -s 0.01 -q
```

Takes ~4 minutes, so **either run it beforehand and show the output**, or
run the fast version live:

```bash
mba stream --verify -q | tail -12
```

> "SON mines a million transactions in 206 MB against an enforced 800 MB
> budget — two streaming passes, and the result is bit-identical to mining
> it all at once, which we test. And for a genuinely unbounded stream, Lossy
> Counting gives bounded memory in one pass with three guarantees we check
> against an exact count: no false negatives, no overestimates, and error
> within ε·N."

---

### 6 · Does any of it predict? (2 min) ⭐

Browser → **Recommender** tab (already warm).

> "Everything so far proves a pattern *existed*. None of it proves the rules
> *predict*. So: train on baskets up to October, test on the two months
> after. A temporal split, because in production you only ever have the
> past. Rules are mined on the training split only — no leakage."

| model | hit@5 | vs popularity |
|---|---|---|
| association | **13.4%** | **9.1×** |
| cooccurrence | 8.5% | 5.8× |
| popularity | 1.5% | 1.0× |

> "9× the popularity baseline. But the number I'd actually defend is the
> 1.6× over raw co-occurrence — *that's* what justifies the whole mining
> pipeline over just counting pairs."

Then say the uncomfortable half — it is the most impressive thing you can do
in a viva:

> "It's also the narrowest model in the table: 13% catalogue coverage
> against co-occurrence's 54%. It's accurate partly by sticking to
> well-trodden products. That's a trade-off, not a clean win."

---

### 7 · Customers don't shop alike (1 min)

Browser → **Segments** tab.

> "RFM plus k-means, implemented from scratch. 16% of customers produce 64%
> of revenue — that's the gap between the two bars. And they shop
> differently: the *loyal* segment's strongest rules are Christmas
> decorations, where champions are Regency teacups. A single global rules
> table averages that away entirely."

---

### 8 · A negative result worth presenting (1 min) ⭐

Browser → **Drift** tab.

> "We mined each month independently and classified every rule. **Zero are
> statistically seasonal.** Peak lift variation is 0.32 against a 0.60
> threshold — while volume swings from 968 baskets in January to 2,524 in
> November."

> "So in this dataset Christmas changes *how much* people buy, not *what
> goes with what*. We left the threshold where it was rather than lowering
> it to manufacture seasonal labels."

Examiners notice when you report a negative result instead of burying it.

---

### 9 · Close on engineering (45 seconds)

```bash
pytest -q
```

> "269 tests, 30 seconds. Both miners are checked against an exhaustive
> brute-force reference, against each other over 50 randomised datasets,
> against the downward-closure invariant, and against **mlxtend** — an
> independent library — for both itemsets and every rule metric. That last
> one rules out the possibility that we misread the papers in the same way
> twice."

End there. It is the strongest note available.

---

## Timing

| § | Topic | Time |
|---|---|---|
| 0 | Framing | 0:30 |
| 1 | Cleaning | 1:00 |
| 2 | Sparse encoding | 0:45 |
| 3 | **Rules + the teacup trap** | 2:00 |
| 4 | **Apriori vs FP-Growth** | 2:00 |
| 5 | Out-of-core / streaming | 1:00 |
| 6 | **Held-out evaluation** | 2:00 |
| 7 | Segmentation | 1:00 |
| 8 | **No seasonality** | 1:00 |
| 9 | Tests | 0:45 |
| | **Total** | **~12 min** |

Starred sections are the ones to protect if you are cut short. If you have
only 5 minutes: §3, §4, §6.

---

## Likely questions, and honest answers

**"Why implement Apriori and FP-Growth yourself instead of using mlxtend?"**
> Because the project is about understanding their cost structure, which you
> cannot see from the outside. We do use mlxtend — as a test oracle, to prove
> our implementations are correct.

**"Your Apriori uses tid-set intersection, isn't that really Eclat?"**
> The *structure* is Apriori: level-wise generation, the F(k−1)×F(k−1) join,
> downward-closure pruning. Only the counting substrate is vertical, which is
> mathematically identical to re-scanning and makes pure-Python tractable. It
> also makes the benchmark fair — we're comparing algorithms, not a tuned
> implementation against a naive one.

**"13% hit rate seems low."**
> Against a 3,355-product catalogue with one held-out item, random is 0.13%
> and popularity is 1.5%. The absolute number is meaningless without the
> baselines; the ratio is 9×.

**"Why is closure giving no compression?"**
> Because the data is sparse. Closure only collapses an itemset into a
> superset of *identical* support, which with 24-item baskets over thousands
> of products essentially never happens. The same code compresses 9.5× on
> dense input — we test both.

**"Is the sample data enough, or do I need the full dataset?"**
> Everything runs on the committed sample. The full dataset changes the
> numbers, not the conclusions — the FP-Growth crossover reproduces on both,
> just at a different threshold.

**"What would you do next?"**
> Sequence-aware rules (what people buy *next*, not just together), and
> serving the recommender behind an API with online A/B measurement instead
> of offline metrics.

---

## If something breaks mid-demo

| symptom | do this |
|---|---|
| dashboard slow / spinning | you skipped the warm-up; talk over it, the cache fills once |
| `mba: command not found` | `source .venv/bin/activate` |
| no dataset | add `-i data/sample/sample_retail.csv` to any command |
| download failed | `bash scripts/check_download.sh` — names the broken layer |
| a port is taken | `mba dashboard --port 8502` |

Have `reports/figures/` open in a file browser as a fallback. Every figure
is a pre-rendered PNG, so you can present the entire analysis with no live
execution at all.

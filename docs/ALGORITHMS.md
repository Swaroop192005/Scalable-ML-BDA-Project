# Algorithms

How each miner works, why it is implemented the way it is, and what the
benchmark actually measures.

---

## Apriori

Agrawal & Srikant, VLDB 1994. Level-wise, breadth-first:

1. Count 1-itemsets; keep those at or above `min_support` → *L₁*.
2. **Join.** Generate candidate *k*-itemsets from *L(k−1)* by merging pairs
   that agree on their first *k−2* items (the F(k−1) × F(k−1) join). Keeping
   itemsets as sorted integer tuples makes those pairs adjacent, so the join
   is blocked rather than an O(n²) sweep.
3. **Prune.** Discard any candidate with an infrequent (k−1)-subset. This is
   the *downward closure* property — if `{A,B}` is infrequent then `{A,B,C}`
   cannot be frequent — and it is the whole reason Apriori beats brute force.
4. **Count** the survivors; repeat until nothing survives or `max_len` is hit.

### The counting substrate

Steps 1–3 are textbook. Step 4 uses **vertical tid-sets**: each itemset carries
the sorted array of transaction ids containing it, and a candidate's count is
the size of the intersection of its two parents' tid-sets.

```python
tids = np.intersect1d(tidsets[a], tidsets[b], assume_unique=True)
if tids.size >= threshold:
    ...
```

This is mathematically identical to re-scanning the database at each level,
but it replaces a Python-level pass over every basket with one vectorised
numpy call. Without it, a pure-Python Apriori is not tractable on 430k basket
lines.

It matters for the benchmark's fairness: the comparison is between the two
algorithms' *structures*, not between a tuned implementation and a naive one.
And it does not rescue Apriori, because no counting trick removes the
combinatorial explosion of candidate *generation* — which is the cost
FP-Growth avoids entirely.

---

## FP-Growth

Han, Pei & Yin, SIGMOD 2000. Two database passes, then all the work happens
inside a prefix tree.

**Pass 1** counts single items and discards the infrequent ones.

**Pass 2** inserts each transaction into an **FP-tree**, items ordered by
descending global frequency so transactions sharing a prefix share a path. A
**header table** chains every occurrence of each item across the tree.

On this dataset the tree holds 127,312 nodes for 430,343 basket lines — a 3.4×
structural compression, achieved purely by prefix sharing.

**Mining** walks the header table from the *least* frequent item upwards. For
item `i`:

1. Collect the **conditional pattern base** — every prefix path ending at `i`,
   weighted by that path's count. The header chain makes this an
   O(occurrences) walk rather than a tree traversal.
2. Re-count items within that base and drop the newly-infrequent ones.
3. Build a **conditional FP-tree** from the surviving prefixes and recurse.

Patterns emerge from the recursion already frequent, so no candidate is ever
generated and then discarded.

### The single-path shortcut

When a conditional tree is one linear path, every subset of that path is
frequent, so the implementation enumerates them combinatorially instead of
recursing. On this dataset it fires 26 times in a run that builds 39
conditional trees — the largest constant-factor win in the algorithm.

### Implementation notes

- `FPNode` uses `__slots__`. At 127k nodes, that cuts per-node overhead by
  roughly 60% versus a `__dict__`-backed object.
- The recursion limit is raised for the duration of a run and restored
  afterwards; deep conditional trees on dense data exceed Python's default.

---

## Why the crossover sits where it does

| min_support | Apriori candidates | FP-Growth candidates | Apriori | FP-Growth |
|---|---|---|---|---|
| 0.050 | 3,761 | 3,355 | **0.04 s** | 0.52 s |
| 0.020 | 41,391 | 3,394 | **1.07 s** | 2.67 s |
| 0.012 | 171,137 | 3,539 | 5.17 s | **4.92 s** |
| 0.008 | 430,492 | 3,846 | 12.87 s | **7.83 s** |

As support falls from 5% to 0.8%, Apriori's candidate count grows **114×**
while FP-Growth's grows **1.15×**.

The two cost curves have different shapes:

- **Apriori** pays for candidate generation. The number of candidate
  2-itemsets is quadratic in the frequent-item count, and that count rises
  sharply as the threshold falls. Its curve bends upward steeply.
- **FP-Growth** pays a fixed price to build the tree — a real cost at high
  support, where there is barely anything to mine — then grows roughly with
  the number of patterns that actually exist.

So the honest conclusion is not "FP-Growth is faster". It is that **FP-Growth's
cost is flat in the quantity that explodes.** At a permissive threshold, with a
few hundred frequent items and cheap candidate sets, Apriori wins outright —
FP-Growth is still paying for its tree. The regime where FP-Growth is the right
default is the one with many products, complex baskets and a low threshold,
which is exactly the regime that matters at scale.

---

## SON (out-of-core)

Savasere, Omiecinski & Navathe, VLDB 1995. Two streaming passes:

**Pass 1 — local mining.** Read the data in chunks that fit in memory. Mine
each chunk independently at the same relative support. Union the per-chunk
frequent itemsets into a global candidate set.

**Pass 2 — global counting.** Stream the data again, counting only those
candidates. Keep the ones that clear global support.

### The correctness guarantee

If an itemset has global support *s*, it must reach local support *s* in **at
least one** chunk — if it were below *s* in every chunk it could not reach *s*
overall. So pass 1 has **no false negatives**, and pass 2 removes the false
positives by exact counting. The result is identical to mining the whole
dataset at once.

`tests/test_scaling.py` asserts exactly that, at four different chunk sizes
and across twelve random datasets. A guarantee that is not tested is a
comment.

### What it costs

On the real data at 2% support with 2,500-transaction chunks: 147,387
candidates generated in pass 1, of which 1,366 are false positives removed in
pass 2. Small chunks mean weaker local pruning and therefore more candidates —
the central tuning trade-off, exposed as `scaling.out_of_core.chunk_size`.

### Counting optimisations

Pass 2 performs tens of millions of membership tests, so:

- Items are interned to small integers and candidates to sorted **int tuples**.
  Tuples hash several times faster than frozensets. This alone cut a run from
  61 s to 22 s.
- Per basket and per candidate size, the loop picks the cheaper of enumerating
  the basket's subsets (`C(n,k)` operations) versus testing each candidate for
  containment (`len(pool)` operations). A wide candidate table never forces
  subset enumeration over a 90-item wholesale basket; a narrow one never
  forces a scan of the whole table.

### Memory

The budget is measured, not assumed. `peak_rss_mb()` reads real resident set
size (normalising the macOS/Linux `ru_maxrss` unit difference), and the miner
raises `MemoryBudgetExceeded` when it is crossed.

Measured: **1,000,000 transactions, 10 chunks, 206 MB peak RSS** against an
800 MB budget.

For this to be genuine, the transaction source must be *re-iterable* rather
than materialised — SON reads the data twice, and a generator can only be read
once. `SyntheticTransactions` therefore implements `__len__` and `__iter__`,
replaying a deterministic stream from its seed, so a million baskets never
exist in memory at once.

---

## PFP (distributed)

Li et al., RecSys 2008 — the algorithm behind `pyspark.ml.fpm.FPGrowth`. It
splits the problem by *item* rather than by row:

1. **Parallel counting** — distributed `reduceByKey` → the global F-list.
2. **Grouping** — frequent items divided into *G* groups, each owning a
   disjoint slice of the pattern space.
3. **Parallel projection** — each transaction mapped to one conditional
   transaction per group it touches, then shuffled so all the data a group
   needs lands on one executor. *This is the only global synchronisation
   point.*
4. **Local FP-Growth** — each executor mines its shard with the sequential
   algorithm.
5. **Aggregation** — union; the groups partition the pattern space by
   construction, so no cross-partition merge is needed.

Step 4 never needs the whole dataset on one machine, which is the property
that makes it scale.

See `src/mba/scaling/spark_fpgrowth.py` for the implementation and sizing
guidance.

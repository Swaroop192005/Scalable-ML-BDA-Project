# Rule metrics and quality triage

## The six metrics

For a rule `A ⇒ C` over *N* transactions, with `s(X)` the support of itemset
*X*:

| Metric | Definition | Range | Reads as |
|---|---|---|---|
| **support** | `s(A ∪ C)` | [0, 1] | how often the pair actually happens |
| **confidence** | `s(A ∪ C) / s(A)` | [0, 1] | P(C \| A) |
| **lift** | `confidence / s(C)` | [0, ∞) | ratio over base rate; **1 = independent** |
| **leverage** | `s(A ∪ C) − s(A)·s(C)` | [−0.25, 0.25] | *absolute* excess co-occurrence |
| **conviction** | `(1 − s(C)) / (1 − confidence)` | [0, ∞) | how often the rule would be wrong by chance |
| **Zhang's metric** | `(conf − s(C)) / max(conf·(1−s(C)), s(C)·(1−conf))` | [−1, 1] | bounded association, robust to skew |

### Why not just lift

Lift is unbounded and unstable when the consequent is rare — a pair seen four
times can top the chart, and one wholesale order can create it. The others
cover its blind spots:

- **Leverage** keeps the *size* of the effect in view. A rule with 20× lift and
  0.0001 leverage moves no revenue.
- **Conviction** is asymmetric, so it distinguishes `A ⇒ C` from `C ⇒ A` where
  lift (symmetric) cannot.
- **Zhang's metric** is bounded and stays interpretable when the marginals are
  lopsided, which they always are in a long-tailed retail catalogue.

### Edge cases

Division-by-zero is resolved to the mathematically meaningful limit where one
exists, rather than left as `inf`/`nan`:

- `confidence = 0` when the antecedent never occurs.
- `conviction = ∞` when confidence is 1 — correct: the rule has no
  counter-examples.
- `conviction = 1` when `s(C) = 1` — a universal consequent is vacuous, not
  infinitely convincing.
- `zhangs_metric = 0` when both denominator terms vanish.

Each case is asserted in `tests/test_rules.py`.

---

## Quality triage

A rule can clear every threshold and still be worthless.
`mba.mining.spurious` applies seven checks and assigns a `quality_tier` of
`high_value` (0 flags), `review` (1), or `spurious` (2+).

| Flag | Fires when | Why it matters |
|---|---|---|
| `rare_pair` | fewer than `min_rule_count` baskets, or antecedent support below floor | lift is a ratio of small numbers and swings wildly |
| `ubiquitous_consequent` | `s(C)` above `max_consequent_support` | predicting a near-universal item is nearly free; confidence looks great and adds nothing |
| `negligible_leverage` | \|leverage\| below floor | high lift, no absolute effect |
| `weak_association` | Zhang's metric below floor | near-independent once skew is accounted for |
| `not_significant` | χ² independence test p > 0.05 | compatible with chance |
| `redundant` | a subset antecedent achieves equal or better confidence | the extra conditions narrow the audience for no gain |
| `self_referential` | antecedent and consequent are product variants | describes the catalogue, not behaviour |

### The χ² test

Each rule's 2×2 contingency table counts baskets by presence/absence of the
antecedent and consequent. With one degree of freedom the survival function
has a closed form:

```
p = erfc(sqrt(χ² / 2))
```

so the whole column is computed at once with no per-row scipy call.

### Why `self_referential` matters here

On the real dataset, **all 45 flagged rules** are this case. The highest-lift
rule in the entire run is:

> `PINK REGENCY TEACUP AND SAUCER` ⇒ `GREEN REGENCY TEACUP AND SAUCER` + `ROSES REGENCY TEACUP AND SAUCER` — 19.7× lift, 71% confidence, 461 baskets

Statistically impeccable. Commercially empty: "customers who buy the pink
teacup also buy the green one" is a fact about how the product range is sold,
not a cross-selling opportunity. Any pipeline that ranks on lift puts this at
the top.

Detection strips colour, size and variant tokens (`RED`, `JUMBO`, `SET`, `OF`,
`3`…) from both sides and measures Jaccard overlap of the remaining content
words. Above 0.6, the rule is flagged.

### Tuning

All thresholds live under `spurious:` in `config/config.yaml`, and the triage
is advisory — flagged rules are written to
`reports/rules_flagged_spurious.csv` rather than discarded, so the judgement
is reviewable and reversible.

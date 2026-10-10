"""Transaction cleaning: cancellations, text hygiene and non-product lines.

Raw retail exports are messy in ways that quietly corrupt association rules:

* **Cancellations** (``InvoiceNo`` prefixed with ``C``) carry negative
  quantities. Left in, they create phantom baskets and double-count returns.
* **Non-product lines** — postage, bank charges, manual adjustments, samples —
  appear in a large share of invoices, so they co-occur with *everything* and
  dominate the top rules with nothing actionable behind them.
* **Description drift** — the same product appears as ``"WHITE  HANGING
  HEART"``, ``"white hanging heart "`` and ``"WHITE HANGING HEART."``. Without
  normalisation each spelling is mined as a separate item and support is
  fragmented.

:func:`clean_transactions` applies the full sequence and returns a
:class:`CleaningReport` so the losses are auditable rather than silent.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from mba.config import Config

log = logging.getLogger(__name__)

#: Collapses runs of whitespace introduced by the source system.
_WS_RE = re.compile(r"\s+")
#: Strips trailing punctuation that is purely cosmetic ("RED HEART." -> "RED HEART").
_TRAILING_PUNCT_RE = re.compile(r"[\s.,;:\-_/\\]+$")


@dataclass
class CleaningReport:
    """Row-level audit trail for the cleaning pipeline."""

    rows_in: int = 0
    rows_out: int = 0
    steps: list[dict[str, Any]] = field(default_factory=list)

    def record(self, step: str, before: int, after: int, note: str = "") -> None:
        removed = before - after
        self.steps.append(
            {
                "step": step,
                "rows_before": before,
                "rows_after": after,
                "rows_removed": removed,
                "pct_removed": round(100.0 * removed / before, 3) if before else 0.0,
                "note": note,
            }
        )
        if removed:
            log.info("%-28s removed %7s rows (%.2f%%)", step, f"{removed:,}",
                     100.0 * removed / before if before else 0.0)

    @property
    def total_removed(self) -> int:
        return self.rows_in - self.rows_out

    @property
    def retention_rate(self) -> float:
        return self.rows_out / self.rows_in if self.rows_in else 0.0

    def to_frame(self) -> pd.DataFrame:
        """Tabular form, ready to write to ``reports/cleaning_report.csv``."""
        return pd.DataFrame(self.steps)

    def summary(self) -> str:
        return (
            f"{self.rows_in:,} raw rows -> {self.rows_out:,} clean rows "
            f"({self.retention_rate:.1%} retained, {self.total_removed:,} dropped "
            f"across {len(self.steps)} steps)"
        )


def normalise_text(series: pd.Series) -> pd.Series:
    """Canonicalise free-text product descriptions.

    Upper-cases, collapses internal whitespace and trims trailing punctuation
    so that cosmetic variants of the same product collapse onto one item key.
    """
    out = series.astype("string").str.strip().str.upper()
    out = out.map(lambda s: _WS_RE.sub(" ", s) if isinstance(s, str) else s)
    out = out.map(lambda s: _TRAILING_PUNCT_RE.sub("", s) if isinstance(s, str) else s)
    return out.replace({"": pd.NA})


def flag_cancellations(invoices: pd.Series, prefix: str = "C") -> pd.Series:
    """Boolean mask of cancelled invoices (``InvoiceNo`` starting with *prefix*)."""
    return invoices.astype("string").str.upper().str.startswith(prefix.upper()).fillna(False)


def _non_product_mask(
    frame: pd.DataFrame,
    code_col: str,
    desc_col: str,
    codes: list[str],
    patterns: list[str],
) -> pd.Series:
    """Mask of service/admin lines that are not sellable products."""
    mask = pd.Series(False, index=frame.index)

    if code_col in frame.columns and codes:
        upper_codes = {c.strip().upper() for c in codes}
        mask |= frame[code_col].astype("string").str.strip().str.upper().isin(upper_codes).fillna(False)

    if desc_col in frame.columns and patterns:
        combined = "|".join(f"(?:{p})" for p in patterns)
        desc = frame[desc_col].astype("string").fillna("")
        mask |= desc.str.contains(combined, case=False, regex=True, na=False)

    return mask


def clean_transactions(
    frame: pd.DataFrame,
    cfg: Config,
    report: CleaningReport | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full cleaning sequence.

    Returns the cleaned frame plus a :class:`CleaningReport`. The input frame
    is never mutated.
    """
    schema = cfg["schema"]
    pre = cfg["preprocess"]

    invoice_col = schema["invoice_col"]
    item_col = schema["item_col"]
    code_col = schema["stock_code_col"]
    qty_col = schema["quantity_col"]
    price_col = schema["price_col"]
    customer_col = schema["customer_col"]

    missing = [c for c in (invoice_col, item_col, qty_col) if c not in frame.columns]
    if missing:
        raise KeyError(f"input frame is missing required columns: {missing}")

    df = frame.copy()
    rep = report or CleaningReport()
    rep.rows_in = len(df)

    # 1. Cancelled invoices -------------------------------------------------
    if pre.get("drop_cancellations", True):
        before = len(df)
        cancelled = flag_cancellations(df[invoice_col], pre.get("cancellation_prefix", "C"))
        df = df.loc[~cancelled]
        rep.record("drop_cancellations", before, len(df), "InvoiceNo starting with 'C'")

    # 2. Text hygiene on the item key --------------------------------------
    before = len(df)
    df[item_col] = normalise_text(df[item_col])
    rep.record("normalise_descriptions", before, len(df), "upper/trim/collapse whitespace")

    if pre.get("drop_missing_description", True):
        before = len(df)
        df = df.loc[df[item_col].notna()]
        rep.record("drop_missing_description", before, len(df))

    # 3. Non-product service lines -----------------------------------------
    before = len(df)
    service = _non_product_mask(
        df,
        code_col,
        item_col,
        list(pre.get("non_product_codes", []) or []),
        list(pre.get("non_product_description_patterns", []) or []),
    )
    df = df.loc[~service]
    rep.record("drop_non_product_lines", before, len(df), "postage, fees, adjustments, damages")

    # 4. Quantity and price sanity -----------------------------------------
    if qty_col in df.columns:
        before = len(df)
        df = df.loc[pd.to_numeric(df[qty_col], errors="coerce") >= pre.get("min_quantity", 1)]
        rep.record("drop_nonpositive_quantity", before, len(df))

    if price_col in df.columns:
        before = len(df)
        df = df.loc[pd.to_numeric(df[price_col], errors="coerce") >= pre.get("min_unit_price", 0.0)]
        rep.record("drop_nonpositive_price", before, len(df), "removes £0.00 adjustment lines")

    if pre.get("drop_missing_customer", False) and customer_col in df.columns:
        before = len(df)
        df = df.loc[df[customer_col].notna()]
        rep.record("drop_missing_customer", before, len(df))

    # 5. Duplicate basket lines --------------------------------------------
    if pre.get("drop_duplicate_lines", True):
        before = len(df)
        df = df.drop_duplicates(subset=[invoice_col, item_col], keep="first")
        rep.record("drop_duplicate_basket_lines", before, len(df), "same item twice in one invoice")

    # 6. Basket-size bounds -------------------------------------------------
    min_size = int(pre.get("min_basket_size", 2))
    max_size = int(pre.get("max_basket_size", 10**9))
    if min_size > 1 or max_size < 10**9:
        before = len(df)
        sizes = df.groupby(invoice_col, observed=True)[item_col].transform("size")
        df = df.loc[(sizes >= min_size) & (sizes <= max_size)]
        rep.record(
            "filter_basket_size",
            before,
            len(df),
            f"keep invoices with {min_size}-{max_size} distinct items",
        )

    df = df.reset_index(drop=True)
    rep.rows_out = len(df)
    log.info("cleaning complete: %s", rep.summary())
    return df, rep

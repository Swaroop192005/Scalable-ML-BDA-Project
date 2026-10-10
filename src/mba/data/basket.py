"""Memory-efficient basket encoding.

A dense one-hot basket matrix for the Online Retail dataset would be roughly
22,000 invoices x 4,000 products = 88M cells. As ``bool`` that is 88 MB; as
the ``float64`` that a naive ``pivot_table`` produces, 700 MB — for a matrix
that is 99.8% zeros.

:func:`encode_baskets` builds a ``scipy.sparse`` CSR matrix directly from the
transaction frame via integer factorisation, never materialising the dense
form. Memory then scales with the number of *basket lines* (~500k) rather
than invoices x products.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from mba.config import Config

log = logging.getLogger(__name__)


@dataclass
class BasketMatrix:
    """A sparse transaction x item incidence matrix plus its labels.

    Attributes
    ----------
    matrix:
        CSR boolean matrix, shape ``(n_transactions, n_items)``.
    items:
        Item label for each column, in column order.
    transaction_ids:
        Invoice id for each row, in row order.
    """

    matrix: sparse.csr_matrix
    items: np.ndarray
    transaction_ids: np.ndarray

    # -- shape helpers ---------------------------------------------------
    @property
    def n_transactions(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_items(self) -> int:
        return self.matrix.shape[1]

    @property
    def density(self) -> float:
        cells = self.n_transactions * self.n_items
        return self.matrix.nnz / cells if cells else 0.0

    @property
    def nbytes(self) -> int:
        m = self.matrix
        return int(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes)

    def dense_nbytes(self, itemsize: int = 1) -> int:
        """Bytes the equivalent dense matrix would occupy."""
        return self.n_transactions * self.n_items * itemsize

    # -- lookups ---------------------------------------------------------
    def item_index(self) -> dict[str, int]:
        return {item: i for i, item in enumerate(self.items)}

    def support_counts(self) -> np.ndarray:
        """Per-item basket counts (column sums)."""
        return np.asarray(self.matrix.sum(axis=0)).ravel().astype(np.int64)

    def item_supports(self) -> pd.Series:
        """Per-item support as a fraction of transactions, descending."""
        counts = self.support_counts()
        s = pd.Series(counts / self.n_transactions, index=self.items, name="support")
        return s.sort_values(ascending=False)

    def basket_sizes(self) -> np.ndarray:
        """Number of distinct items in each transaction."""
        return np.diff(self.matrix.indptr)

    def iter_transactions(self) -> Iterator[list[int]]:
        """Yield each basket as a sorted list of integer item ids."""
        indptr, indices = self.matrix.indptr, self.matrix.indices
        for row in range(self.n_transactions):
            yield indices[indptr[row] : indptr[row + 1]].tolist()

    def as_item_lists(self) -> list[list[str]]:
        """Yield each basket as a list of item *labels* (costly; for display)."""
        labels = self.items
        return [[labels[i] for i in basket] for basket in self.iter_transactions()]

    def filter_items(self, min_count: int) -> "BasketMatrix":
        """Drop columns whose basket count is below *min_count*."""
        keep = self.support_counts() >= min_count
        if keep.all():
            return self
        sub = self.matrix[:, keep].tocsr()
        log.info("pruned %d/%d items below %d baskets", (~keep).sum(), len(keep), min_count)
        return BasketMatrix(sub, self.items[keep], self.transaction_ids)

    def drop_empty_transactions(self) -> "BasketMatrix":
        """Drop rows left with fewer than two items (no association possible)."""
        keep = self.basket_sizes() >= 2
        if keep.all():
            return self
        return BasketMatrix(self.matrix[keep].tocsr(), self.items, self.transaction_ids[keep])

    def to_dense_frame(self, max_cells: int = 20_000_000) -> pd.DataFrame:
        """Materialise a dense boolean DataFrame — guarded against blow-ups.

        Only use this for small slices (tests, the dashboard's preview table).
        """
        cells = self.n_transactions * self.n_items
        if cells > max_cells:
            raise MemoryError(
                f"refusing to densify {self.n_transactions:,} x {self.n_items:,} "
                f"= {cells:,} cells (limit {max_cells:,}). Mine on the sparse "
                "matrix instead."
            )
        return pd.DataFrame(
            self.matrix.toarray(), index=self.transaction_ids, columns=self.items
        )

    def describe(self) -> dict[str, float | int]:
        """Summary stats used by the EDA report and the dashboard header."""
        sizes = self.basket_sizes()
        return {
            "n_transactions": self.n_transactions,
            "n_items": self.n_items,
            "nnz": int(self.matrix.nnz),
            "density": self.density,
            "sparse_mb": self.nbytes / 1e6,
            "dense_equivalent_mb": self.dense_nbytes() / 1e6,
            "compression_ratio": (self.dense_nbytes() / self.nbytes) if self.nbytes else 0.0,
            "mean_basket_size": float(sizes.mean()) if sizes.size else 0.0,
            "median_basket_size": float(np.median(sizes)) if sizes.size else 0.0,
            "max_basket_size": int(sizes.max()) if sizes.size else 0,
        }


def encode_baskets(
    frame: pd.DataFrame,
    cfg: Config,
    invoice_col: str | None = None,
    item_col: str | None = None,
    min_item_count: int | None = None,
) -> BasketMatrix:
    """Build a sparse :class:`BasketMatrix` from a cleaned transaction frame.

    Uses ``pandas.factorize`` to map invoices and items to contiguous integer
    ids, then hands those straight to ``scipy.sparse.coo_matrix``. Peak memory
    is a few arrays of length ``len(frame)`` — no dense intermediate exists at
    any point.
    """
    invoice_col = invoice_col or cfg["schema"]["invoice_col"]
    item_col = item_col or cfg["schema"]["item_col"]
    if min_item_count is None:
        min_item_count = int(cfg.get("basket.min_item_support_count", 1) or 1)

    if frame.empty:
        return BasketMatrix(
            sparse.csr_matrix((0, 0), dtype=bool),
            np.array([], dtype=object),
            np.array([], dtype=object),
        )

    rows, tx_labels = pd.factorize(frame[invoice_col], sort=True)
    cols, item_labels = pd.factorize(frame[item_col], sort=True)

    data = np.ones(len(frame), dtype=bool)
    coo = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(len(tx_labels), len(item_labels)),
        dtype=bool,
    )
    # ``sum_duplicates`` on a bool matrix keeps True for repeated (row, col)
    # pairs, which is exactly the set semantics a basket needs.
    csr = coo.tocsr()
    csr.sum_duplicates()
    csr.data = np.ones_like(csr.data, dtype=bool)

    basket = BasketMatrix(csr, np.asarray(item_labels, dtype=object), np.asarray(tx_labels, dtype=object))

    if min_item_count > 1:
        basket = basket.filter_items(min_item_count)
    basket = basket.drop_empty_transactions()

    stats = basket.describe()
    log.info(
        "encoded %s baskets x %s items | density %.4f%% | %.1f MB sparse vs %.1f MB dense (%.0fx)",
        f"{stats['n_transactions']:,}",
        f"{stats['n_items']:,}",
        stats["density"] * 100,
        stats["sparse_mb"],
        stats["dense_equivalent_mb"],
        stats["compression_ratio"],
    )
    return basket


def transactions_from_frame(
    frame: pd.DataFrame,
    invoice_col: str,
    item_col: str,
) -> list[list[str]]:
    """Group a transaction frame into a list-of-lists of item labels.

    This is the representation FP-Growth and the out-of-core miner consume
    when they are not handed a :class:`BasketMatrix`.
    """
    grouped = frame.groupby(invoice_col, observed=True, sort=False)[item_col]
    return [sorted(set(map(str, vals))) for vals in grouped.agg(list)]


def baskets_from_transactions(
    transactions: Iterable[Sequence[str]],
    min_item_count: int = 1,
) -> BasketMatrix:
    """Build a :class:`BasketMatrix` from raw list-of-lists input.

    Mirrors :func:`encode_baskets` for callers (tests, the Spark bridge) that
    already hold transactions rather than a line-level frame.
    """
    tx_list = [sorted(set(t)) for t in transactions]
    vocab: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    for r, basket in enumerate(tx_list):
        for item in basket:
            cols.append(vocab.setdefault(item, len(vocab)))
            rows.append(r)

    items = np.array(sorted(vocab, key=vocab.get), dtype=object)
    csr = sparse.coo_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)),
        shape=(len(tx_list), len(vocab)),
        dtype=bool,
    ).tocsr()

    bm = BasketMatrix(csr, items, np.arange(len(tx_list), dtype=object))
    return bm.filter_items(min_item_count) if min_item_count > 1 else bm

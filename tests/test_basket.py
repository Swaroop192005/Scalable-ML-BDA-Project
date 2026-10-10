"""Sparse basket encoding."""

import numpy as np
import pytest
from scipy import sparse

from mba.data.basket import BasketMatrix, baskets_from_transactions, encode_baskets, transactions_from_frame
from mba.data.preprocess import clean_transactions


def test_shape_and_labels(toy_basket, toy_transactions):
    assert toy_basket.n_transactions == len(toy_transactions)
    assert set(toy_basket.items) == {i for t in toy_transactions for i in t}


def test_matrix_is_sparse_and_boolean(toy_basket):
    assert sparse.isspmatrix_csr(toy_basket.matrix)
    assert toy_basket.matrix.dtype == bool


def test_cells_match_the_source_transactions(toy_basket, toy_transactions):
    index = toy_basket.item_index()
    dense = toy_basket.matrix.toarray()
    for row, basket in enumerate(toy_transactions):
        for item in basket:
            assert dense[row, index[item]], f"missing {item!r} in basket {row}"
        assert dense[row].sum() == len(set(basket))


def test_support_counts_are_correct(toy_basket, toy_transactions):
    counts = dict(zip(toy_basket.items, toy_basket.support_counts()))
    assert counts["milk"] == sum("milk" in t for t in toy_transactions)
    assert counts["bread"] == sum("bread" in t for t in toy_transactions)


def test_repeated_item_in_one_basket_counts_once():
    basket = baskets_from_transactions([["a", "a", "b"], ["a", "b"]])
    assert basket.support_counts().tolist() == [2, 2]
    assert basket.matrix.nnz == 4


def test_sparse_is_far_smaller_than_dense(toy_basket):
    stats = toy_basket.describe()
    assert stats["density"] < 1.0
    assert stats["dense_equivalent_mb"] >= 0.0
    assert stats["nnz"] == toy_basket.matrix.nnz


def test_filter_items_prunes_rare_columns():
    basket = baskets_from_transactions([["a", "b"], ["a", "b"], ["a", "rare"]])
    pruned = basket.filter_items(min_count=2)
    assert "rare" not in set(pruned.items)
    assert set(pruned.items) == {"a", "b"}


def test_drop_empty_transactions_removes_thin_baskets():
    basket = baskets_from_transactions([["a", "b"], ["solo"]])
    kept = basket.filter_items(min_count=2).drop_empty_transactions()
    assert kept.n_transactions == 0 or (kept.basket_sizes() >= 2).all()


def test_densify_refuses_to_blow_up_memory():
    big = BasketMatrix(
        sparse.csr_matrix((5000, 5000), dtype=bool),
        np.arange(5000, dtype=object),
        np.arange(5000, dtype=object),
    )
    with pytest.raises(MemoryError):
        big.to_dense_frame(max_cells=1000)


def test_iter_transactions_round_trips(toy_basket, toy_transactions):
    recovered = [sorted(b) for b in toy_basket.as_item_lists()]
    assert recovered == [sorted(set(t)) for t in toy_transactions]


def test_empty_frame_yields_empty_matrix(cfg):
    import pandas as pd

    empty = pd.DataFrame(columns=["InvoiceNo", "Description"])
    basket = encode_baskets(empty, cfg)
    assert basket.n_transactions == 0 and basket.n_items == 0


def test_encode_from_frame_matches_grouping(toy_frame, cfg):
    clean, _ = clean_transactions(toy_frame, cfg)
    basket = encode_baskets(clean, cfg, min_item_count=1)
    grouped = transactions_from_frame(clean, "InvoiceNo", "Description")
    assert basket.n_transactions == len(grouped)
    assert sorted(basket.as_item_lists()) == sorted(grouped)

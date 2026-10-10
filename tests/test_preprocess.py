"""Cleaning: cancellations, text hygiene, service lines and the audit trail."""

import pandas as pd
import pytest

from mba.data.preprocess import (
    CleaningReport,
    clean_transactions,
    flag_cancellations,
    normalise_text,
)


def test_normalise_text_collapses_variants():
    series = pd.Series(["  white  mug ", "WHITE MUG.", "White   Mug", None, ""])
    out = normalise_text(series)
    assert out.iloc[0] == out.iloc[1] == out.iloc[2] == "WHITE MUG"
    assert pd.isna(out.iloc[3]) and pd.isna(out.iloc[4])


def test_flag_cancellations_matches_c_prefix_only():
    series = pd.Series(["C536379", "536380", "c536381", None])
    flags = flag_cancellations(series)
    assert flags.tolist() == [True, False, True, False]


def test_cleaning_removes_every_planted_defect(messy_frame, cfg):
    clean, report = clean_transactions(messy_frame, cfg)

    assert not clean["InvoiceNo"].str.startswith("C").any(), "cancellation survived"
    assert "POSTAGE" not in set(clean["Description"]), "service line survived"
    assert clean["Description"].notna().all(), "null description survived"
    assert (clean["UnitPrice"] > 0).all(), "zero-price line survived"
    assert (clean["Quantity"] > 0).all(), "non-positive quantity survived"
    assert not clean.duplicated(["InvoiceNo", "Description"]).any(), "duplicate line survived"


def test_text_variants_collapse_to_one_item(messy_frame, cfg):
    clean, _ = clean_transactions(messy_frame, cfg)
    descriptions = set(clean["Description"])
    # "  bread  " and "BREAD" must be the same item, not two.
    assert "BREAD" in descriptions
    assert not any(d != d.strip() or d != d.upper() for d in descriptions)


def test_report_accounts_for_every_dropped_row(messy_frame, cfg):
    clean, report = clean_transactions(messy_frame, cfg)
    assert report.rows_in == len(messy_frame)
    assert report.rows_out == len(clean)
    assert report.total_removed == report.rows_in - report.rows_out
    # Each step's "after" must equal the next step's "before" — no gaps.
    steps = report.steps
    for earlier, later in zip(steps, steps[1:]):
        assert earlier["rows_after"] == later["rows_before"]
    assert 0.0 <= report.retention_rate <= 1.0


def test_min_basket_size_drops_single_item_invoices(cfg):
    # Stock codes must not collide with the configured service codes
    # ("B", "D", "M", "S", "POST", ...), which are dropped as non-products.
    frame = pd.DataFrame(
        [
            {"InvoiceNo": "1", "StockCode": "11111", "Description": "ALPHA MUG", "Quantity": 1,
             "InvoiceDate": pd.Timestamp("2011-01-01"), "UnitPrice": 1.0,
             "CustomerID": 1.0, "Country": "UK"},
            {"InvoiceNo": "2", "StockCode": "11111", "Description": "ALPHA MUG", "Quantity": 1,
             "InvoiceDate": pd.Timestamp("2011-01-01"), "UnitPrice": 1.0,
             "CustomerID": 1.0, "Country": "UK"},
            {"InvoiceNo": "2", "StockCode": "22222", "Description": "BETA BOWL", "Quantity": 1,
             "InvoiceDate": pd.Timestamp("2011-01-01"), "UnitPrice": 1.0,
             "CustomerID": 1.0, "Country": "UK"},
        ]
    )
    clean, _ = clean_transactions(frame, cfg)
    assert set(clean["InvoiceNo"]) == {"2"}, "1-item invoice should be dropped"


def test_single_letter_service_codes_are_dropped(cfg):
    """'B', 'D', 'M' and 'POST' are the real dataset's adjustment codes."""
    frame = pd.DataFrame(
        [
            {"InvoiceNo": "9", "StockCode": code, "Description": f"ITEM {code}",
             "Quantity": 1, "InvoiceDate": pd.Timestamp("2011-01-01"),
             "UnitPrice": 1.0, "CustomerID": 1.0, "Country": "UK"}
            for code in ("B", "D", "M", "POST", "33333", "44444")
        ]
    )
    clean, _ = clean_transactions(frame, cfg)
    assert set(clean["StockCode"]) == {"33333", "44444"}


def test_input_frame_is_not_mutated(messy_frame, cfg):
    before = messy_frame.copy()
    clean_transactions(messy_frame, cfg)
    pd.testing.assert_frame_equal(messy_frame, before)


def test_missing_required_column_raises(cfg):
    with pytest.raises(KeyError):
        clean_transactions(pd.DataFrame({"InvoiceNo": ["1"]}), cfg)


def test_empty_frame_is_handled(cfg):
    empty = pd.DataFrame(
        columns=["InvoiceNo", "StockCode", "Description", "Quantity",
                 "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
    )
    clean, report = clean_transactions(empty, cfg)
    assert clean.empty and report.rows_out == 0

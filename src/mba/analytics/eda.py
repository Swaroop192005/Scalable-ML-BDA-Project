"""Exploratory data analysis over the cleaned transaction table."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from mba.config import Config
from mba.data.basket import BasketMatrix

log = logging.getLogger(__name__)


@dataclass
class EDAReport:
    """Every table the EDA stage produces, keyed by name."""

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    scalars: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> pd.DataFrame:
        return self.tables[key]

    def keys(self):
        return self.tables.keys()

    def summary(self) -> str:
        s = self.scalars
        return (
            f"{s.get('n_transactions', 0):,} transactions | "
            f"{s.get('n_products', 0):,} products | "
            f"{s.get('n_customers', 0):,} customers | "
            f"{s.get('n_countries', 0)} countries | "
            f"revenue £{s.get('total_revenue', 0):,.0f} | "
            f"{s.get('date_min', '?')} to {s.get('date_max', '?')}"
        )


def run_eda(frame: pd.DataFrame, cfg: Config, basket: BasketMatrix | None = None) -> EDAReport:
    """Compute the descriptive tables that drive the report and the dashboard."""
    schema = cfg["schema"]
    invoice, item, qty = schema["invoice_col"], schema["item_col"], schema["quantity_col"]
    price, customer, country = schema["price_col"], schema["customer_col"], schema["country_col"]
    date_col = schema["date_col"]

    top_products = int(cfg.get("analytics.eda.top_n_products", 15))
    top_countries = int(cfg.get("analytics.eda.top_n_countries", 12))

    df = frame.copy()
    if price in df.columns and qty in df.columns:
        df["Revenue"] = df[qty] * df[price]

    report = EDAReport()

    # -- headline scalars ------------------------------------------------
    report.scalars = {
        "n_rows": len(df),
        "n_transactions": int(df[invoice].nunique()),
        "n_products": int(df[item].nunique()),
        "n_customers": int(df[customer].nunique()) if customer in df.columns else 0,
        "n_countries": int(df[country].nunique()) if country in df.columns else 0,
        "total_revenue": float(df["Revenue"].sum()) if "Revenue" in df.columns else 0.0,
        "total_units": int(df[qty].sum()) if qty in df.columns else 0,
    }
    if date_col in df.columns and df[date_col].notna().any():
        report.scalars["date_min"] = str(df[date_col].min().date())
        report.scalars["date_max"] = str(df[date_col].max().date())

    # -- product league tables -------------------------------------------
    by_volume = (
        df.groupby(item, observed=True)
        .agg(
            transactions=(invoice, "nunique"),
            units=(qty, "sum"),
            revenue=("Revenue", "sum") if "Revenue" in df.columns else (qty, "sum"),
        )
        .sort_values("transactions", ascending=False)
    )
    by_volume["support"] = by_volume["transactions"] / report.scalars["n_transactions"]
    report.tables["product_stats"] = by_volume
    report.tables["top_products_volume"] = by_volume.head(top_products).reset_index()
    report.tables["top_products_revenue"] = (
        by_volume.sort_values("revenue", ascending=False).head(top_products).reset_index()
    )

    # -- basket size distribution ----------------------------------------
    basket_sizes = df.groupby(invoice, observed=True)[item].nunique()
    report.tables["basket_sizes"] = basket_sizes.rename("items").reset_index()
    report.scalars.update(
        {
            "mean_basket_size": float(basket_sizes.mean()),
            "median_basket_size": float(basket_sizes.median()),
            "p95_basket_size": float(basket_sizes.quantile(0.95)),
            "max_basket_size": int(basket_sizes.max()),
        }
    )
    hist_counts = basket_sizes.value_counts().sort_index()
    report.tables["basket_size_distribution"] = pd.DataFrame(
        {"basket_size": hist_counts.index.to_numpy(), "n_invoices": hist_counts.to_numpy()}
    )

    # -- basket value ------------------------------------------------------
    if "Revenue" in df.columns:
        basket_value = df.groupby(invoice, observed=True)["Revenue"].sum()
        report.tables["basket_value"] = basket_value.rename("revenue").reset_index()
        report.scalars.update(
            {
                "mean_basket_value": float(basket_value.mean()),
                "median_basket_value": float(basket_value.median()),
            }
        )

    # -- country breakdown --------------------------------------------------
    if country in df.columns:
        by_country = (
            df.groupby(country, observed=True)
            .agg(
                transactions=(invoice, "nunique"),
                products=(item, "nunique"),
                revenue=("Revenue", "sum") if "Revenue" in df.columns else (qty, "sum"),
                customers=(customer, "nunique") if customer in df.columns else (invoice, "nunique"),
            )
            .sort_values("transactions", ascending=False)
        )
        by_country["revenue_share"] = by_country["revenue"] / by_country["revenue"].sum()
        by_country["avg_basket_value"] = by_country["revenue"] / by_country["transactions"]
        report.tables["country_stats"] = by_country
        report.tables["top_countries"] = by_country.head(top_countries).reset_index()

    # -- temporal patterns ---------------------------------------------------
    if date_col in df.columns and df[date_col].notna().any():
        dates = df[date_col]
        monthly = (
            df.assign(month=dates.dt.to_period("M").astype(str))
            .groupby("month", observed=True)
            .agg(
                transactions=(invoice, "nunique"),
                revenue=("Revenue", "sum") if "Revenue" in df.columns else (qty, "sum"),
            )
            .reset_index()
        )
        report.tables["monthly"] = monthly

        report.tables["by_weekday"] = (
            df.assign(weekday=dates.dt.day_name())
            .groupby("weekday", observed=True)
            .agg(transactions=(invoice, "nunique"))
            .reindex(
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            )
            .dropna()
            .reset_index()
        )
        report.tables["by_hour"] = (
            df.assign(hour=dates.dt.hour)
            .groupby("hour", observed=True)
            .agg(transactions=(invoice, "nunique"))
            .reset_index()
        )

    # -- customer concentration (Pareto) -------------------------------------
    if customer in df.columns and "Revenue" in df.columns and df[customer].notna().any():
        per_customer = (
            df.dropna(subset=[customer]).groupby(customer, observed=True)["Revenue"].sum()
        ).sort_values(ascending=False)
        if len(per_customer):
            cumulative = per_customer.cumsum() / per_customer.sum()
            report.tables["customer_pareto"] = pd.DataFrame(
                {
                    "customer_rank_pct": np.arange(1, len(per_customer) + 1) / len(per_customer),
                    "cumulative_revenue_pct": cumulative.to_numpy(),
                }
            )
            top_decile = max(1, len(per_customer) // 10)
            report.scalars["top_10pct_customers_revenue_share"] = float(
                per_customer.head(top_decile).sum() / per_customer.sum()
            )

    # -- sparsity profile from the encoded matrix -----------------------------
    if basket is not None:
        report.scalars.update({f"matrix_{k}": v for k, v in basket.describe().items()})
        supports = basket.item_supports()
        report.tables["item_support_distribution"] = pd.DataFrame(
            {"item": supports.index.to_numpy(), "support": supports.to_numpy()}
        )

    log.info("EDA: %s", report.summary())
    return report

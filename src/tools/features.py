"""
Feature Engineering Tool.
Computes model-ready AML features on demand per customer/account entity.
Features:
- Transaction frequency & volume
- Amount deviation z-score
- Graph degrees (in_degree, out_degree)
- Proportion of round-number amounts
- Proportion in threshold band ($8,500 - $10,000)
"""

from typing import Any
import numpy as np
import polars as pl
from src.tools.graph_analysis import build_transaction_graph, compute_graph_metrics


def compute_aml_features(df: pl.DataFrame, required_features: list[str] | None = None) -> pl.DataFrame:
    """
    Computes tabular feature matrix per account entity.
    Only computes features specified in required_features or default AML set.
    """
    if df is None or len(df) == 0:
        return pl.DataFrame()

    amt_col = "Amount Received"

    # Graph degree metrics
    G = build_transaction_graph(df)
    graph_metrics = compute_graph_metrics(G)

    # Polars aggregations per Account
    feat_df = df.group_by("Account").agg([
        pl.len().alias("tx_count"),
        pl.col(amt_col).sum().alias("total_volume"),
        pl.col(amt_col).mean().alias("mean_amount"),
        pl.col(amt_col).std().fill_null(0.0).alias("std_amount"),
        pl.col(amt_col).max().alias("max_amount"),
        # Round number ratio (ends in .00 or 000)
        (pl.col(amt_col) % 100 == 0).mean().alias("round_number_ratio"),
        # Threshold band ratio ($8,500 - $10,000)
        ((pl.col(amt_col) >= 8500.0) & (pl.col(amt_col) < 10000.0)).mean().alias("threshold_band_ratio")
    ])

    # Compute z-score of max amount relative to customer mean
    feat_df = feat_df.with_columns(
        (
            pl.when(pl.col("std_amount") > 0)
            .then((pl.col("max_amount") - pl.col("mean_amount")) / pl.col("std_amount"))
            .otherwise(0.0)
        ).alias("amount_z_score")
    )

    # Attach graph degrees
    in_degs = []
    out_degs = []
    for acc in feat_df["Account"].to_list():
        m = graph_metrics.get(acc, {})
        in_degs.append(m.get("in_degree", 0.0))
        out_degs.append(m.get("out_degree", 0.0))

    feat_df = feat_df.with_columns([
        pl.Series("in_degree", in_degs),
        pl.Series("out_degree", out_degs)
    ])

    return feat_df

"""
Exploratory Data Analysis (EDA) Tool.
Profiles transaction dataset, computes summary statistics, amount quantiles, velocity distribution, and entity counts.
"""

from typing import Any
import polars as pl
from src.data.preprocess import parse_timestamp_column


def run_eda(df: pl.DataFrame) -> dict[str, Any]:
    """
    Performs EDA on dataset and returns structured summary metrics.
    """
    if df is None or len(df) == 0:
        return {"error": "Empty dataset"}

    df_ts = parse_timestamp_column(df)
    total_tx = len(df_ts)

    # Unique entities
    senders = df_ts["Account"].unique().to_list() if "Account" in df_ts.columns else []
    receivers = df_ts["Account.1"].unique().to_list() if "Account.1" in df_ts.columns else []
    all_accounts = list(set(senders + receivers))

    # Amount statistics
    amt_col = "Amount Received" if "Amount Received" in df_ts.columns else df_ts.columns[0]
    amounts = df_ts[amt_col]

    amt_min = float(amounts.min() or 0.0)
    amt_max = float(amounts.max() or 0.0)
    amt_mean = float(amounts.mean() or 0.0)
    amt_median = float(amounts.median() or 0.0)
    amt_q25 = float(amounts.quantile(0.25) or 0.0)
    amt_q75 = float(amounts.quantile(0.75) or 0.0)

    # Date range
    ts_min = str(df_ts["parsed_timestamp"].min()) if "parsed_timestamp" in df_ts.columns else "N/A"
    ts_max = str(df_ts["parsed_timestamp"].max()) if "parsed_timestamp" in df_ts.columns else "N/A"

    # Payment formats breakdown
    fmt_counts = {}
    if "Payment Format" in df_ts.columns:
        fmt_df = df_ts.group_by("Payment Format").len()
        fmt_counts = dict(zip(fmt_df["Payment Format"].to_list(), fmt_df["len"].to_list()))

    # Laundering labels (if present)
    laundering_stats = {}
    if "Is Laundering" in df_ts.columns:
        launder_df = df_ts.group_by("Is Laundering").len()
        laundering_stats = dict(zip([str(k) for k in launder_df["Is Laundering"].to_list()], launder_df["len"].to_list()))

    # Top high-volume accounts
    top_senders = []
    if "Account" in df_ts.columns:
        top_s_df = df_ts.group_by("Account").agg(
            pl.len().alias("count"),
            pl.col(amt_col).sum().alias("total_vol")
        ).sort("total_vol", descending=True).head(5)
        top_senders = top_s_df.to_dicts()

    return {
        "summary": {
            "total_transactions": total_tx,
            "unique_accounts": len(all_accounts),
            "unique_senders": len(senders),
            "unique_receivers": len(receivers),
            "date_range": f"{ts_min} to {ts_max}"
        },
        "amount_quantiles": {
            "min": round(amt_min, 2),
            "q25": round(amt_q25, 2),
            "median": round(amt_median, 2),
            "mean": round(amt_mean, 2),
            "q75": round(amt_q75, 2),
            "max": round(amt_max, 2)
        },
        "payment_formats": fmt_counts,
        "laundering_ground_truth": laundering_stats,
        "top_senders": top_senders
    }

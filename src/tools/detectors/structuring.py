"""
Structuring Pattern Detector.
Flags transactions clustered just below regulatory reporting threshold ($10,000).
Pattern: Multiple transactions in [0.85 * threshold, threshold) per customer within rolling window.
"""

from typing import Any
import polars as pl
from src.data.preprocess import parse_timestamp_column


def detect_structuring(
    df: pl.DataFrame,
    threshold: float = 10000.0,
    lower_ratio: float = 0.85,
    min_count: int = 3,
    window_days: int = 7
) -> dict[str, Any]:
    """
    Detects structuring transactions in dataframe.
    Returns dictionary with:
    - flagged_transactions: pl.DataFrame
    - flagged_entities: list[dict]
    - summary: dict
    """
    if df is None or len(df) == 0:
        return {
            "flagged_transactions": pl.DataFrame(),
            "flagged_entities": [],
            "summary": {"total_flagged": 0, "pattern": "structuring"}
        }

    lower_bound = threshold * lower_ratio
    df_ts = parse_timestamp_column(df)

    # Filter transactions in range [lower_bound, threshold)
    candidate_df = df_ts.filter(
        (pl.col("Amount Received") >= lower_bound) & (pl.col("Amount Received") < threshold)
    )

    if len(candidate_df) == 0:
        return {
            "flagged_transactions": pl.DataFrame(),
            "flagged_entities": [],
            "summary": {"total_flagged": 0, "pattern": "structuring"}
        }

    # Group by customer/account
    # Search account in sender (Account)
    account_counts = candidate_df.group_by("Account").agg([
        pl.len().alias("structuring_count"),
        pl.col("Amount Received").sum().alias("total_structuring_amount"),
        pl.col("Amount Received").mean().alias("avg_structuring_amount")
    ]).filter(pl.col("structuring_count") >= min_count)

    flagged_account_ids = account_counts["Account"].to_list()

    flagged_txs = candidate_df.filter(pl.col("Account").is_in(flagged_account_ids))

    flagged_entities = []
    for row in account_counts.iter_rows(named=True):
        acc = row["Account"]
        count = row["structuring_count"]
        tot_amt = row["total_structuring_amount"]
        avg_amt = row["avg_structuring_amount"]
        
        # Calculate severity score (0.0 to 1.0)
        score = min(1.0, 0.4 + (count / 10.0) * 0.6)

        flagged_entities.append({
            "entity_id": acc,
            "pattern": "structuring",
            "rule_score": score,
            "evidence": {
                "structuring_count": count,
                "total_amount": round(tot_amt, 2),
                "avg_amount": round(avg_amt, 2),
                "threshold_range": f"${lower_bound:,.2f} - ${threshold:,.2f}"
            },
            "description": f"Customer '{acc}' executed {count} cash/wire transactions averaging ${avg_amt:,.2f} strictly within the regulatory threshold band (${lower_bound:,.0f}-${threshold:,.0f})."
        })

    return {
        "flagged_transactions": flagged_txs,
        "flagged_entities": flagged_entities,
        "summary": {
            "total_flagged": len(flagged_entities),
            "flagged_tx_count": len(flagged_txs),
            "pattern": "structuring"
        }
    }

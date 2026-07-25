"""
Velocity Anomaly Detector.
Computes z-score of customer transaction count and total volume against population or historical baseline.
Catches sudden behavior shifts.
"""

from typing import Any
# pyrefly: ignore [missing-import]
import polars as pl


def detect_velocity_anomaly(
    df: pl.DataFrame,
    z_threshold: float = 3.0
) -> dict[str, Any]:
    """
    Detects customer accounts with anomalous transaction velocity or volume z-score.
    """
    if df is None or len(df) == 0:
        return {
            "flagged_transactions": pl.DataFrame(),
            "flagged_entities": [],
            "summary": {"total_flagged": 0, "pattern": "velocity"}
        }

    # Aggregate by customer
    stats = df.group_by("Account").agg([
        pl.len().alias("tx_count"),
        pl.col("Amount Received").sum().alias("total_volume"),
        pl.col("Amount Received").mean().alias("mean_amount")
    ])

    if len(stats) < 3:
        return {
            "flagged_transactions": pl.DataFrame(),
            "flagged_entities": [],
            "summary": {"total_flagged": 0, "pattern": "velocity"}
        }

    # Compute z-scores for tx_count and total_volume
    count_mean = stats["tx_count"].mean()
    count_std = stats["tx_count"].std() or 1.0

    vol_mean = stats["total_volume"].mean()
    vol_std = stats["total_volume"].std() or 1.0

    stats = stats.with_columns([
        ((pl.col("tx_count") - count_mean) / count_std).alias("z_count"),
        ((pl.col("total_volume") - vol_mean) / vol_std).alias("z_volume")
    ])

    anomalies = stats.filter(
        (pl.col("z_count") > z_threshold) | (pl.col("z_volume") > z_threshold)
    )

    flagged_entities = []
    flagged_acc_ids = []

    for row in anomalies.iter_rows(named=True):
        acc = row["Account"]
        z_c = row["z_count"]
        z_v = row["z_volume"]
        max_z = max(z_c, z_v)
        
        score = min(1.0, 0.4 + (max_z / 10.0) * 0.6)
        flagged_acc_ids.append(acc)

        flagged_entities.append({
            "entity_id": acc,
            "pattern": "velocity",
            "rule_score": score,
            "evidence": {
                "tx_count": row["tx_count"],
                "total_volume": round(row["total_volume"], 2),
                "count_z_score": round(z_c, 2),
                "volume_z_score": round(z_v, 2),
                "baseline_z_threshold": z_threshold
            },
            "description": f"Account '{acc}' exhibited an extreme velocity spike with volume z-score of {z_v:.2f} and transaction count z-score of {z_c:.2f} above baseline."
        })

    flagged_txs = df.filter(pl.col("Account").is_in(flagged_acc_ids))

    return {
        "flagged_transactions": flagged_txs,
        "flagged_entities": flagged_entities,
        "summary": {
            "total_flagged": len(flagged_entities),
            "flagged_tx_count": len(flagged_txs),
            "pattern": "velocity"
        }
    }

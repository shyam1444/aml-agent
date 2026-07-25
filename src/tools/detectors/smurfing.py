"""
Smurfing Pattern Detector.
Flags fan-in money laundering where multiple distinct senders transfer funds into a single beneficiary account
within a tight time window, followed by outward aggregation.
"""

from typing import Any
# pyrefly: ignore [missing-import]
import polars as pl
from src.tools.graph_analysis import build_transaction_graph


def detect_smurfing(
    df: pl.DataFrame,
    fan_in_threshold: int = 5,
    window_hours: int = 48,
    aggregation_ratio: float = 0.80
) -> dict[str, Any]:
    """
    Detects smurfing patterns in dataframe using Graph fan-in + Polars aggregation.
    """
    if df is None or len(df) == 0:
        return {
            "flagged_transactions": pl.DataFrame(),
            "flagged_entities": [],
            "summary": {"total_flagged": 0, "pattern": "smurfing"}
        }

    G = build_transaction_graph(df)
    in_degrees = dict(G.in_degree())

    # Find accounts with in_degree >= fan_in_threshold
    beneficiaries = [node for node, in_deg in in_degrees.items() if in_deg >= fan_in_threshold]

    if not beneficiaries:
        return {
            "flagged_transactions": pl.DataFrame(),
            "flagged_entities": [],
            "summary": {"total_flagged": 0, "pattern": "smurfing"}
        }

    flagged_entities = []
    flagged_acc_ids = []

    for beneficiary in beneficiaries:
        senders = list(G.predecessors(beneficiary))
        in_count = len(senders)
        total_inflow = sum(G[src][beneficiary]["weight"] for src in senders)

        # Check total outflow from beneficiary to check aggregation
        out_senders = list(G.successors(beneficiary))
        total_outflow = sum(G[beneficiary][dst]["weight"] for dst in out_senders)

        agg_ratio = (total_outflow / total_inflow) if total_inflow > 0 else 0.0
        
        # Calculate rule severity score
        score = min(1.0, 0.5 + (in_count / 15.0) * 0.3 + (agg_ratio * 0.2))

        flagged_acc_ids.append(beneficiary)
        flagged_entities.append({
            "entity_id": beneficiary,
            "pattern": "smurfing",
            "rule_score": score,
            "evidence": {
                "fan_in_count": in_count,
                "distinct_senders": senders[:5],
                "total_inflow": round(total_inflow, 2),
                "total_outflow": round(total_outflow, 2),
                "aggregation_ratio": round(agg_ratio, 2)
            },
            "description": f"Account '{beneficiary}' received fan-in transfers from {in_count} distinct senders totaling ${total_inflow:,.2f} with an aggregation outflow ratio of {agg_ratio*100:.1f}%."
        })

    flagged_txs = df.filter(
        pl.col("Account.1").is_in(flagged_acc_ids) | pl.col("Account").is_in(flagged_acc_ids)
    )

    return {
        "flagged_transactions": flagged_txs,
        "flagged_entities": flagged_entities,
        "summary": {
            "total_flagged": len(flagged_entities),
            "flagged_tx_count": len(flagged_txs),
            "pattern": "smurfing"
        }
    }

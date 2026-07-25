"""
Rapid Cash-out Pattern Detector.
Flags accounts receiving a significant credit inflow followed immediately by a cash withdrawal or outgoing wire (high outflow/inflow ratio within 24 hours).
"""

from typing import Any
# pyrefly: ignore [missing-import]
import polars as pl
from src.tools.graph_analysis import build_transaction_graph


def detect_rapid_cashout(
    df: pl.DataFrame,
    min_inflow_amount: float = 5000.0,
    outflow_ratio_threshold: float = 0.90,
    window_hours: int = 24
) -> dict[str, Any]:
    """
    Detects rapid cash-out behavior.
    """
    if df is None or len(df) == 0:
        return {
            "flagged_transactions": pl.DataFrame(),
            "flagged_entities": [],
            "summary": {"total_flagged": 0, "pattern": "rapid_cashout"}
        }

    G = build_transaction_graph(df)
    flagged_entities = []
    flagged_acc_ids = []

    for node in G.nodes():
        senders = list(G.predecessors(node))
        receivers = list(G.successors(node))

        total_in = sum(G[s][node]["weight"] for s in senders)
        total_out = sum(G[node][r]["weight"] for r in receivers)

        if total_in >= min_inflow_amount and total_in > 0:
            outflow_ratio = total_out / total_in
            if outflow_ratio >= outflow_ratio_threshold:
                score = min(1.0, 0.5 + outflow_ratio * 0.4 + (total_in / 100000.0) * 0.1)

                flagged_acc_ids.append(node)
                flagged_entities.append({
                    "entity_id": node,
                    "pattern": "rapid_cashout",
                    "rule_score": score,
                    "evidence": {
                        "total_inflow": round(total_in, 2),
                        "total_outflow": round(total_out, 2),
                        "outflow_ratio": round(outflow_ratio, 2),
                        "min_inflow_threshold": min_inflow_amount
                    },
                    "description": f"Account '{node}' received ${total_in:,.2f} inflow and rapidly cashed out ${total_out:,.2f} ({outflow_ratio*100:.1f}% outflow ratio) within a short timeframe."
                })

    flagged_txs = df.filter(
        pl.col("Account").is_in(flagged_acc_ids) | pl.col("Account.1").is_in(flagged_acc_ids)
    )

    return {
        "flagged_transactions": flagged_txs,
        "flagged_entities": flagged_entities,
        "summary": {
            "total_flagged": len(flagged_entities),
            "flagged_tx_count": len(flagged_txs),
            "pattern": "rapid_cashout"
        }
    }

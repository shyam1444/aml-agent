"""
Round-Tripping Pattern Detector.
Flags circular transaction flows where funds travel through external nodes and return to the origin account.
Uses NetworkX directed cycle detection algorithms.
"""

from typing import Any
import networkx as nx
# pyrefly: ignore [missing-import]
import polars as pl
from src.tools.graph_analysis import build_transaction_graph


def detect_round_tripping(
    df: pl.DataFrame,
    max_cycle_len: int = 5
) -> dict[str, Any]:
    """
    Detects circular transaction cycles in NetworkX graph.
    """
    if df is None or len(df) == 0:
        return {
            "flagged_transactions": pl.DataFrame(),
            "flagged_entities": [],
            "summary": {"total_flagged": 0, "pattern": "round_tripping"}
        }

    G = build_transaction_graph(df)
    if len(G) < 2:
        return {
            "flagged_transactions": pl.DataFrame(),
            "flagged_entities": [],
            "summary": {"total_flagged": 0, "pattern": "round_tripping"}
        }

    try:
        raw_cycles = list(nx.simple_cycles(G))
    except Exception:
        raw_cycles = []

    valid_cycles = [c for c in raw_cycles if 2 <= len(c) <= max_cycle_len]

    flagged_entities_map = {}
    for cycle in valid_cycles:
        cycle_str = " -> ".join(cycle) + " -> " + cycle[0]
        # Calculate volume in cycle
        cycle_vol = 0.0
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            cycle_vol += G[u][v]["weight"]
        avg_vol = cycle_vol / len(cycle)

        score = min(1.0, 0.7 + (len(cycle) / 10.0))

        for node in cycle:
            if node not in flagged_entities_map:
                flagged_entities_map[node] = {
                    "entity_id": node,
                    "pattern": "round_tripping",
                    "rule_score": score,
                    "evidence": {
                        "cycle_path": cycle_str,
                        "cycle_length": len(cycle),
                        "avg_volume": round(avg_vol, 2)
                    },
                    "description": f"Account '{node}' is part of a circular flow cycle ({cycle_str}) where funds return to origin."
                }

    flagged_entities = list(flagged_entities_map.values())
    flagged_acc_ids = list(flagged_entities_map.keys())

    flagged_txs = df.filter(
        pl.col("Account").is_in(flagged_acc_ids) | pl.col("Account.1").is_in(flagged_acc_ids)
    )

    return {
        "flagged_transactions": flagged_txs,
        "flagged_entities": flagged_entities,
        "summary": {
            "total_flagged": len(flagged_entities),
            "cycles_found": len(valid_cycles),
            "pattern": "round_tripping"
        }
    }

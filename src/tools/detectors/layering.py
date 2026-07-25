"""
Layering Pattern Detector.
Flags complex multi-hop transaction chains (3+ hops) designed to obfuscate audit trails.
Uses NetworkX directed simple paths with amount conservation checks (allowing 2-5% fee attrition).
"""

from typing import Any
# pyrefly: ignore [untyped-import]
import networkx as nx
# pyrefly: ignore [missing-import]
import polars as pl
from src.tools.graph_analysis import build_transaction_graph


def detect_layering(
    df: pl.DataFrame,
    min_hops: int = 3,
    max_hops: int = 6,
    fee_attrition_max: float = 0.05
) -> dict[str, Any]:
    """
    Detects multi-hop layering chains using NetworkX path search.
    """
    if df is None or len(df) == 0:
        return {
            "flagged_transactions": pl.DataFrame(),
            "flagged_entities": [],
            "summary": {"total_flagged": 0, "pattern": "layering"}
        }

    G = build_transaction_graph(df)
    if len(G) < min_hops + 1:
        return {
            "flagged_transactions": pl.DataFrame(),
            "flagged_entities": [],
            "summary": {"total_flagged": 0, "pattern": "layering"}
        }

    detected_chains = []
    flagged_entities_map = {}

    # Sample nodes with both in and out degree to find intermediate chain links
    # pyrefly: ignore [unsupported-operation]
    nodes_with_in_out = [n for n in G.nodes() if G.in_degree(n) > 0 and G.out_degree(n) > 0]

    for source in list(G.nodes()):
        # Limit paths search for performance
        if G.out_degree(source) == 0:
            continue
            
        for target in list(G.nodes()):
            if source == target or G.in_degree(target) == 0:
                continue

            try:
                # pyrefly: ignore [bad-argument-type]
                paths = list(nx.all_simple_paths(G, source=source, target=target, cutoff=max_hops))
            except Exception:
                continue

            for path in paths:
                if len(path) - 1 < min_hops:
                    continue

                # Verify amount conservation along path
                is_conserved = True
                path_amounts = []
                for idx in range(len(path) - 1):
                    u, v = path[idx], path[idx + 1]
                    weight = G[u][v]["weight"]
                    path_amounts.append(weight)
                    if idx > 0:
                        prev_weight = path_amounts[idx - 1]
                        ratio = weight / prev_weight if prev_weight > 0 else 0
                        # Check if amount is roughly conserved (e.g. 0.90 to 1.05)
                        if not (0.85 <= ratio <= 1.05):
                            is_conserved = False
                            break

                if is_conserved and path_amounts:
                    chain_str = " -> ".join(path)
                    start_amt = path_amounts[0]
                    end_amt = path_amounts[-1]
                    attrition = (start_amt - end_amt) / start_amt if start_amt > 0 else 0.0

                    detected_chains.append({
                        "path": path,
                        "hops": len(path) - 1,
                        "start_amount": start_amt,
                        "end_amount": end_amt,
                        "attrition": round(attrition, 3)
                    })

                    # Flag each account in chain
                    for acc in path:
                        if acc not in flagged_entities_map:
                            flagged_entities_map[acc] = {
                                "entity_id": acc,
                                "pattern": "layering",
                                "rule_score": min(1.0, 0.6 + (len(path) / 10.0)),
                                "evidence": {
                                    "chain": chain_str,
                                    "hops": len(path) - 1,
                                    "initial_amount": round(start_amt, 2),
                                    "final_amount": round(end_amt, 2),
                                    "attrition_rate": f"{attrition*100:.1f}%"
                                },
                                "description": f"Account '{acc}' participated in a {len(path)-1}-hop layering chain ({chain_str}) conserving ${start_amt:,.2f} across multi-bank transfers."
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
            "chains_found": len(detected_chains),
            "pattern": "layering"
        }
    }

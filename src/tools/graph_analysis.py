"""
Graph Analysis Tool leveraging NetworkX for transaction network modeling.
Constructs directed transaction graphs, extracts subgraphs, detects cycles, and evaluates node centrality.
"""

from typing import Any
import networkx as nx
import polars as pl


def build_transaction_graph(df: pl.DataFrame) -> nx.DiGraph:
    """
    Constructs a NetworkX directed graph from transaction dataframe.
    Edges contain amount, timestamp, payment format, and transaction count.
    """
    G = nx.DiGraph()
    if df is None or len(df) == 0:
        return G

    # Iterate over polars rows
    for row in df.iter_rows(named=True):
        src = str(row["Account"])
        dst = str(row["Account.1"])
        amt = float(row["Amount Received"])
        ts = str(row.get("Timestamp", ""))
        fmt = str(row.get("Payment Format", ""))

        if G.has_edge(src, dst):
            G[src][dst]["weight"] += amt
            G[src][dst]["count"] += 1
            G[src][dst]["transactions"].append({"amount": amt, "timestamp": ts, "format": fmt})
        else:
            G.add_edge(
                src,
                dst,
                weight=amt,
                count=1,
                transactions=[{"amount": amt, "timestamp": ts, "format": fmt}]
            )

    return G


def compute_graph_metrics(G: nx.DiGraph) -> dict[str, dict[str, float]]:
    """
    Computes graph metrics per account entity:
    - in_degree: count of distinct incoming senders
    - out_degree: count of distinct outgoing receivers
    - in_degree_centrality
    - out_degree_centrality
    """
    if len(G) == 0:
        return {}

    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    in_cent = nx.in_degree_centrality(G)
    out_cent = nx.out_degree_centrality(G)

    metrics = {}
    for node in G.nodes():
        metrics[node] = {
            "in_degree": float(in_degrees.get(node, 0)),
            "out_degree": float(out_degrees.get(node, 0)),
            "in_degree_centrality": float(in_cent.get(node, 0.0)),
            "out_degree_centrality": float(out_cent.get(node, 0.0)),
        }
    return metrics


def extract_entity_subgraph_nodes(G: nx.DiGraph, entity_id: str, depth: int = 2) -> list[str]:
    """
    Extracts node list for a k-hop subgraph around a target entity.
    """
    if entity_id not in G:
        # Check if entity exists without prefix
        matches = [n for n in G.nodes() if entity_id in n]
        if not matches:
            return []
        entity_id = matches[0]

    nodes = {entity_id}
    current_frontier = {entity_id}
    for _ in range(depth):
        next_frontier = set()
        for node in current_frontier:
            successors = set(G.successors(node))
            predecessors = set(G.predecessors(node))
            next_frontier.update(successors | predecessors)
        nodes.update(next_frontier)
        current_frontier = next_frontier

    return list(nodes)

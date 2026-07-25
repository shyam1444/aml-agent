"""
Risk Classification and Escalation Recommendation Engine.
Calculates weighted composite risk score:
  composite = 0.40 * rule_score + 0.35 * ml_score + 0.25 * graph_score
Assigns Risk Bands (Low, Medium, High) and Escalation Recommendations (Monitor, Review, Report).
"""

from typing import Any
import polars as pl


def classify_risk(
    entities: list[dict[str, Any]],
    ml_scores_df: pl.DataFrame | None = None,
    rule_weight: float = 0.40,
    ml_weight: float = 0.35,
    graph_weight: float = 0.25,
    low_cutoff: float = 0.35,
    high_cutoff: float = 0.70
) -> list[dict[str, Any]]:
    """
    Computes composite risk score, risk level, and recommendation per flagged entity.
    """
    if not entities:
        return []

    # Map ML scores if available
    ml_map = {}
    if ml_scores_df is not None and len(ml_scores_df) > 0 and "Account" in ml_scores_df.columns:
        for row in ml_scores_df.iter_rows(named=True):
            ml_map[row["Account"]] = float(row.get("ml_score", 0.0))

    classified_entities = []

    for item in entities:
        entity_id = item["entity_id"]
        rule_score = float(item.get("rule_score", 0.5))
        ml_score = ml_map.get(entity_id, 0.0)

        # Graph score derived from pattern type or graph centrality evidence
        pattern = item.get("pattern", "general")
        if pattern in ["smurfing", "layering", "round_tripping"]:
            graph_score = min(1.0, rule_score * 1.1)
        else:
            graph_score = float(item.get("evidence", {}).get("graph_score", 0.2))

        # Composite score
        composite = (rule_weight * rule_score) + (ml_weight * ml_score) + (graph_weight * graph_score)
        composite = min(1.0, max(0.0, composite))

        # Risk level categorization
        if composite >= high_cutoff:
            risk_level = "HIGH"
            recommendation = "REPORT"
            action_desc = "Immediately file a Suspicious Activity Report (SAR) with Financial Intelligence Unit and freeze target account."
        elif composite >= low_cutoff:
            risk_level = "MEDIUM"
            recommendation = "REVIEW"
            action_desc = "Escalate to Senior Compliance Analyst for manual transaction review and enhanced due diligence (EDD)."
        else:
            risk_level = "LOW"
            recommendation = "MONITOR"
            action_desc = "Place account on 30-day automated monitoring watch-list."

        updated_item = dict(item)
        updated_item["rule_score"] = round(rule_score, 3)
        updated_item["ml_score"] = round(ml_score, 3)
        updated_item["graph_score"] = round(graph_score, 3)
        updated_item["composite_score"] = round(composite, 3)
        updated_item["risk_level"] = risk_level
        updated_item["recommendation"] = recommendation
        updated_item["action_description"] = action_desc

        classified_entities.append(updated_item)

    # Sort descending by composite score
    classified_entities.sort(key=lambda x: x["composite_score"], reverse=True)
    return classified_entities

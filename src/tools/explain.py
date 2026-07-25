"""
Two-Stage Explanation Engine.
Stage 1: Deterministic SHAP attribution over IsolationForest + Rule threshold evidence + Graph evidence.
Stage 2: LLM verbalization using Groq Llama 3.3 70B (with offline deterministic fallback) converting exact attributions into concise prose.
"""

import os
from typing import Any
import numpy as np
import shap
from groq import Groq


def compute_shap_attributions(model: Any, X_data: np.ndarray, feature_names: list[str], target_idx: int = 0) -> dict[str, float]:
    """
    Computes SHAP feature importance attributions for a target instance using TreeExplainer.
    """
    if model is None or X_data is None or len(X_data) == 0:
        return {}

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_data[target_idx:target_idx+1])
        
        if isinstance(shap_values, list):
            vals = shap_values[0][0]
        elif len(shap_values.shape) == 2:
            vals = shap_values[0]
        else:
            vals = shap_values

        attributions = {}
        for idx, feat_name in enumerate(feature_names):
            attributions[feat_name] = float(vals[idx])

        return attributions
    except Exception as e:
        # Fallback empty or uniform
        return {feat: 0.0 for feat in feature_names}


def generate_explanation_prose(
    entity_id: str,
    query_intent: str,
    pattern: str,
    composite_score: float,
    risk_level: str,
    evidence: dict[str, Any],
    shap_attributions: dict[str, float] | None = None
) -> str:
    """
    Generates concise natural language reason tied strictly to query intent and stage 1 attributions.
    Uses Groq Llama 3.3 70B if GROQ_API_KEY environment variable is present; otherwise uses deterministic template.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    # Stage 1 evidence summary string
    evidence_items = [f"{k}: {v}" for k, v in evidence.items()]
    evidence_str = ", ".join(evidence_items)

    top_shap_str = ""
    if shap_attributions:
        sorted_shap = sorted(shap_attributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        top_shap_str = ", ".join([f"{k} (SHAP impact {v:+.3f})" for k, v in sorted_shap])

    if api_key:
        try:
            client = Groq(api_key=api_key)
            prompt = f"""You are an AML Compliance AI Explainer.
User Query Intent: {query_intent}
Target Entity: {entity_id}
Detected AML Typology: {pattern}
Composite Risk Score: {composite_score:.3f} ({risk_level})
Stage 1 Ground-Truth Evidence: {evidence_str}
Top Feature Attributions: {top_shap_str}

Write a concise, professional 2-3 sentence AML compliance explanation detailing WHY this entity was flagged as {risk_level} risk.
CRITICAL CONSTRAINT: You MUST use the exact numbers provided in the evidence above. Never invent unprovided facts or figures."""
            
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception:
            pass  # Fallback to deterministic template

    # Stage 2 Fallback Deterministic Prose Generator
    if pattern == "structuring":
        return f"Customer '{entity_id}' executed {evidence.get('structuring_count', 'multiple')} transactions in the threshold range {evidence.get('threshold_range', '$8,500-$10,000')} totaling ${evidence.get('total_amount', 0):,.2f}. This matches structuring behavior to bypass reporting limits."

    elif pattern == "smurfing":
        return f"Account '{entity_id}' received fan-in transfers from {evidence.get('fan_in_count', 0)} distinct senders totaling ${evidence.get('total_inflow', 0):,.2f} with an aggregation outflow ratio of {float(evidence.get('aggregation_ratio', 0))*100:.1f}%, indicating smurfing aggregation."

    elif pattern == "layering":
        return f"Account '{entity_id}' participated in a {evidence.get('hops', 3)}-hop layering chain ({evidence.get('chain', '')}) moving ${evidence.get('initial_amount', 0):,.2f} across accounts with low attrition."

    elif pattern == "rapid_cashout":
        return f"Account '{entity_id}' received inflow of ${evidence.get('total_inflow', 0):,.2f} followed by rapid cash-out of ${evidence.get('total_outflow', 0):,.2f} within 24 hours ({float(evidence.get('outflow_ratio', 0))*100:.1f}% outflow ratio)."

    elif pattern == "round_tripping":
        return f"Account '{entity_id}' engaged in circular transaction flow cycle ({evidence.get('cycle_path', '')}) where funds return to origin."

    elif pattern == "velocity":
        return f"Account '{entity_id}' displayed abnormal velocity spike with volume z-score of {evidence.get('volume_z_score', 0):.2f} and count z-score of {evidence.get('count_z_score', 0):.2f} above baseline."

    return f"Entity '{entity_id}' was assigned a composite risk score of {composite_score:.3f} ({risk_level}) based on observed transaction signals: {evidence_str}."

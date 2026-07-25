"""
Router & Tool Dispatcher Nodes for LangGraph Agent.
Appends detailed TraceEntry records for every step executed.
"""

import time
from typing import Any
import polars as pl

from src.agent.state import AgentState, TraceEntry, PlanStep
from src.data.loader import load_dataset
from src.data.preprocess import apply_time_filter, apply_entity_filter, apply_amount_filter
from src.tools.eda import run_eda
from src.tools.features import compute_aml_features
from src.tools.detectors.structuring import detect_structuring
from src.tools.detectors.smurfing import detect_smurfing
from src.tools.detectors.layering import detect_layering
from src.tools.detectors.rapid_cashout import detect_rapid_cashout
from src.tools.detectors.round_tripping import detect_round_tripping
from src.tools.detectors.velocity import detect_velocity_anomaly
from src.tools.anomaly import detect_anomalies_ml
from src.tools.risk import classify_risk
from src.tools.explain import compute_shap_attributions, generate_explanation_prose


def route_next_step(state: AgentState) -> str:
    """
    Conditional router edge for LangGraph.
    Returns the next node name to execute or 'complete'.
    """
    if state.get("needs_human_input", False):
        return "human_in_the_loop"

    remaining = state.get("remaining_steps", [])
    if not remaining:
        return "complete"

    next_step = remaining[0]
    tool_name = next_step.tool_name

    # Map tool name string to node function key
    if "Time Filter" in tool_name or "Entity Lookup" in tool_name:
        return "preprocess_node"
    elif "EDA" in tool_name:
        return "eda_node"
    elif "Feature Engine" in tool_name:
        return "feature_node"
    elif "Structuring" in tool_name:
        return "structuring_node"
    elif "Smurfing" in tool_name:
        return "smurfing_node"
    elif "Layering" in tool_name:
        return "layering_node"
    elif "Rapid Cashout" in tool_name:
        return "rapid_cashout_node"
    elif "Round Tripping" in tool_name or "Cycle" in tool_name:
        return "round_tripping_node"
    elif "Velocity" in tool_name:
        return "velocity_node"
    elif "Rule Query" in tool_name:
        return "rule_query_node"
    elif "Anomaly Detect" in tool_name:
        return "ml_anomaly_node"
    elif "Risk Classify" in tool_name:
        return "risk_classify_node"
    elif "Explainer" in tool_name:
        return "explainer_node"
    elif "Recommender" in tool_name:
        return "recommender_node"

    return "generic_tool_node"


def preprocess_node(state: AgentState) -> dict[str, Any]:
    """Preprocesses and filters dataframe based on query intent."""
    start_time = time.perf_counter()
    remaining = list(state.get("remaining_steps", []))
    current_step = remaining.pop(0) if remaining else PlanStep(step_index=1, tool_name="Preprocess", purpose="Filter")

    df = state.get("working_df")
    if df is None:
        df = load_dataset()

    in_count = len(df)
    intent = state.get("intent")

    if intent:
        if intent.date_range and intent.date_range.days:
            df = apply_time_filter(df, days=intent.date_range.days)
        if intent.entity_ids:
            df = apply_entity_filter(df, intent.entity_ids)

    out_count = len(df)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    trace_entry = {
        "step_index": current_step.step_index,
        "tool_name": current_step.tool_name,
        "selection_reason": current_step.purpose,
        "input_row_count": in_count,
        "output_row_count": out_count,
        "wall_clock_ms": round(elapsed_ms, 2),
        "status": "COMPLETED"
    }

    completed = list(state.get("completed_steps", [])) + [current_step.tool_name]
    trace = list(state.get("execution_trace", [])) + [trace_entry]

    return {
        "working_df": df,
        "remaining_steps": remaining,
        "completed_steps": completed,
        "execution_trace": trace
    }


def eda_node(state: AgentState) -> dict[str, Any]:
    """Runs EDA profiling tool."""
    start_time = time.perf_counter()
    remaining = list(state.get("remaining_steps", []))
    current_step = remaining.pop(0) if remaining else PlanStep(step_index=1, tool_name="EDA Tool", purpose="Profile")

    df = state.get("working_df")
    if df is None:
        df = load_dataset()

    in_count = len(df)
    eda_summary = run_eda(df)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    trace_entry = {
        "step_index": current_step.step_index,
        "tool_name": current_step.tool_name,
        "selection_reason": current_step.purpose,
        "input_row_count": in_count,
        "output_row_count": in_count,
        "wall_clock_ms": round(elapsed_ms, 2),
        "status": "COMPLETED"
    }

    completed = list(state.get("completed_steps", [])) + [current_step.tool_name]
    trace = list(state.get("execution_trace", [])) + [trace_entry]

    return {
        "remaining_steps": remaining,
        "completed_steps": completed,
        "execution_trace": trace
    }


def feature_node(state: AgentState) -> dict[str, Any]:
    """Runs on-demand feature engineering."""
    start_time = time.perf_counter()
    remaining = list(state.get("remaining_steps", []))
    current_step = remaining.pop(0) if remaining else PlanStep(step_index=1, tool_name="Feature Engine", purpose="Features")

    df = state.get("working_df")
    if df is None:
        df = load_dataset()

    in_count = len(df)
    feat_df = compute_aml_features(df)
    out_count = len(feat_df)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    trace_entry = {
        "step_index": current_step.step_index,
        "tool_name": current_step.tool_name,
        "selection_reason": current_step.purpose,
        "input_row_count": in_count,
        "output_row_count": out_count,
        "wall_clock_ms": round(elapsed_ms, 2),
        "status": "COMPLETED"
    }

    features_dict = state.get("features", {})
    features_dict["features_df"] = feat_df

    completed = list(state.get("completed_steps", [])) + [current_step.tool_name]
    trace = list(state.get("execution_trace", [])) + [trace_entry]

    return {
        "features": features_dict,
        "remaining_steps": remaining,
        "completed_steps": completed,
        "execution_trace": trace
    }


def structuring_node(state: AgentState) -> dict[str, Any]:
    """Runs Structuring Detector."""
    start_time = time.perf_counter()
    remaining = list(state.get("remaining_steps", []))
    current_step = remaining.pop(0) if remaining else PlanStep(step_index=1, tool_name="Structuring Detector", purpose="Detect")

    df = state.get("working_df")
    if df is None:
        df = load_dataset()

    in_count = len(df)
    res = detect_structuring(df)
    flagged_entities = res.get("flagged_entities", [])
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    trace_entry = {
        "step_index": current_step.step_index,
        "tool_name": current_step.tool_name,
        "selection_reason": current_step.purpose,
        "input_row_count": in_count,
        "output_row_count": len(flagged_entities),
        "wall_clock_ms": round(elapsed_ms, 2),
        "status": "COMPLETED"
    }

    existing_flagged = list(state.get("flagged", [])) + flagged_entities
    completed = list(state.get("completed_steps", [])) + [current_step.tool_name]
    trace = list(state.get("execution_trace", [])) + [trace_entry]

    return {
        "flagged": existing_flagged,
        "remaining_steps": remaining,
        "completed_steps": completed,
        "execution_trace": trace
    }


def rule_query_node(state: AgentState) -> dict[str, Any]:
    """Runs threshold aggregation rule query directly (e.g. 10+ tx under $10k)."""
    start_time = time.perf_counter()
    remaining = list(state.get("remaining_steps", []))
    current_step = remaining.pop(0) if remaining else PlanStep(step_index=1, tool_name="Rule Query", purpose="Aggregation")

    df = state.get("working_df")
    if df is None:
        df = load_dataset()

    in_count = len(df)

    # Filter tx under $10,000 and group by customer to find count >= 10
    filtered_df = df.filter(pl.col("Amount Received") < 10000.0)
    agg = filtered_df.group_by("Account").agg([
        pl.len().alias("tx_count"),
        pl.col("Amount Received").sum().alias("total_amt"),
        pl.col("Amount Received").mean().alias("avg_amt")
    ]).filter(pl.col("tx_count") >= 10)

    flagged_entities = []
    for row in agg.iter_rows(named=True):
        acc = row["Account"]
        cnt = row["tx_count"]
        tot = row["total_amt"]
        avg = row["avg_amt"]

        flagged_entities.append({
            "entity_id": acc,
            "pattern": "threshold_rule",
            "rule_score": min(1.0, 0.5 + (cnt / 20.0)),
            "evidence": {
                "tx_under_10k_count": cnt,
                "total_amount": round(tot, 2),
                "avg_amount": round(avg, 2),
                "rule_criterion": "10+ transactions under $10,000"
            },
            "description": f"Customer '{acc}' completed {cnt} transactions under $10,000 (avg ${avg:,.2f}), triggering deterministic threshold rule."
        })

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    trace_entry = {
        "step_index": current_step.step_index,
        "tool_name": current_step.tool_name,
        "selection_reason": current_step.purpose,
        "input_row_count": in_count,
        "output_row_count": len(flagged_entities),
        "wall_clock_ms": round(elapsed_ms, 2),
        "status": "COMPLETED"
    }

    completed = list(state.get("completed_steps", [])) + [current_step.tool_name]
    trace = list(state.get("execution_trace", [])) + [trace_entry]

    return {
        "flagged": flagged_entities,
        "remaining_steps": remaining,
        "completed_steps": completed,
        "execution_trace": trace
    }


def ml_anomaly_node(state: AgentState) -> dict[str, Any]:
    """Runs ML anomaly detection node."""
    start_time = time.perf_counter()
    remaining = list(state.get("remaining_steps", []))
    current_step = remaining.pop(0) if remaining else PlanStep(step_index=1, tool_name="Anomaly Detect (ML)", purpose="ML Anomaly")

    features_dict = state.get("features", {})
    feat_df = features_dict.get("features_df")
    if feat_df is None:
        df = state.get("working_df") or load_dataset()
        feat_df = compute_aml_features(df)

    in_count = len(feat_df)
    ml_res = detect_anomalies_ml(feat_df)
    scores_df = ml_res.get("scores_df")

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    trace_entry = {
        "step_index": current_step.step_index,
        "tool_name": current_step.tool_name,
        "selection_reason": current_step.purpose,
        "input_row_count": in_count,
        "output_row_count": len(scores_df) if scores_df is not None else 0,
        "wall_clock_ms": round(elapsed_ms, 2),
        "status": "COMPLETED"
    }

    features_dict["ml_model"] = ml_res.get("model")
    features_dict["feature_cols"] = ml_res.get("feature_cols")
    features_dict["X_data"] = ml_res.get("X_data")

    completed = list(state.get("completed_steps", [])) + [current_step.tool_name]
    trace = list(state.get("execution_trace", [])) + [trace_entry]

    return {
        "scores": scores_df,
        "features": features_dict,
        "remaining_steps": remaining,
        "completed_steps": completed,
        "execution_trace": trace
    }


def risk_classify_node(state: AgentState) -> dict[str, Any]:
    """Classifies composite risk scores and recommendations."""
    start_time = time.perf_counter()
    remaining = list(state.get("remaining_steps", []))
    current_step = remaining.pop(0) if remaining else PlanStep(step_index=1, tool_name="Risk Classify", purpose="Classify Risk")

    raw_flagged = state.get("flagged", [])
    scores_df = state.get("scores")

    classified = classify_risk(raw_flagged, ml_scores_df=scores_df)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    trace_entry = {
        "step_index": current_step.step_index,
        "tool_name": current_step.tool_name,
        "selection_reason": current_step.purpose,
        "input_row_count": len(raw_flagged),
        "output_row_count": len(classified),
        "wall_clock_ms": round(elapsed_ms, 2),
        "status": "COMPLETED"
    }

    completed = list(state.get("completed_steps", [])) + [current_step.tool_name]
    trace = list(state.get("execution_trace", [])) + [trace_entry]

    return {
        "flagged": classified,
        "remaining_steps": remaining,
        "completed_steps": completed,
        "execution_trace": trace
    }


def explainer_node(state: AgentState) -> dict[str, Any]:
    """Generates 2-stage explanations for flagged entities."""
    start_time = time.perf_counter()
    remaining = list(state.get("remaining_steps", []))
    current_step = remaining.pop(0) if remaining else PlanStep(step_index=1, tool_name="Explainer", purpose="Explain")

    flagged = state.get("flagged", [])
    user_query = state.get("user_query", "")
    features_dict = state.get("features", {})
    ml_model = features_dict.get("ml_model")
    X_data = features_dict.get("X_data")
    feature_cols = features_dict.get("feature_cols", [])

    updated_flagged = []
    explanations_dict = {}

    for idx, entity in enumerate(flagged):
        entity_id = entity["entity_id"]
        pattern = entity.get("pattern", "general")
        composite_score = float(entity.get("composite_score", 0.5))
        risk_level = entity.get("risk_level", "MEDIUM")
        evidence = entity.get("evidence", {})

        shap_attrs = None
        if ml_model is not None and X_data is not None and feature_cols:
            shap_attrs = compute_shap_attributions(ml_model, X_data, feature_cols, target_idx=min(idx, len(X_data)-1))

        prose = generate_explanation_prose(
            entity_id=entity_id,
            query_intent=user_query,
            pattern=pattern,
            composite_score=composite_score,
            risk_level=risk_level,
            evidence=evidence,
            shap_attributions=shap_attrs
        )

        item = dict(entity)
        item["explanation"] = prose
        updated_flagged.append(item)
        explanations_dict[entity_id] = prose

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    trace_entry = {
        "step_index": current_step.step_index,
        "tool_name": current_step.tool_name,
        "selection_reason": current_step.purpose,
        "input_row_count": len(flagged),
        "output_row_count": len(updated_flagged),
        "wall_clock_ms": round(elapsed_ms, 2),
        "status": "COMPLETED"
    }

    completed = list(state.get("completed_steps", [])) + [current_step.tool_name]
    trace = list(state.get("execution_trace", [])) + [trace_entry]

    return {
        "flagged": updated_flagged,
        "explanations": explanations_dict,
        "remaining_steps": remaining,
        "completed_steps": completed,
        "execution_trace": trace
    }


def recommender_node(state: AgentState) -> dict[str, Any]:
    """Final recommender step recording completion trace."""
    start_time = time.perf_counter()
    remaining = list(state.get("remaining_steps", []))
    current_step = remaining.pop(0) if remaining else PlanStep(step_index=1, tool_name="Recommender", purpose="Escalate")

    flagged = state.get("flagged", [])
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    trace_entry = {
        "step_index": current_step.step_index,
        "tool_name": current_step.tool_name,
        "selection_reason": current_step.purpose,
        "input_row_count": len(flagged),
        "output_row_count": len(flagged),
        "wall_clock_ms": round(elapsed_ms, 2),
        "status": "COMPLETED"
    }

    completed = list(state.get("completed_steps", [])) + [current_step.tool_name]
    trace = list(state.get("execution_trace", [])) + [trace_entry]

    return {
        "remaining_steps": remaining,
        "completed_steps": completed,
        "execution_trace": trace
    }

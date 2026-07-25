"""
LangGraph StateGraph Assembly for AML Agent.
Connects QueryParser -> Planner -> Dynamic Router -> Tool Nodes -> Risk Classifier -> Explainer -> Recommender.
"""

from typing import Any
# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.planner import query_parser_node, planner_node
from src.agent.router import (
    route_next_step,
    preprocess_node,
    eda_node,
    feature_node,
    structuring_node,
    rule_query_node,
    ml_anomaly_node,
    risk_classify_node,
    explainer_node,
    recommender_node
)
from src.tools.detectors.smurfing import detect_smurfing
from src.tools.detectors.layering import detect_layering
from src.tools.detectors.rapid_cashout import detect_rapid_cashout
from src.tools.detectors.round_tripping import detect_round_tripping
from src.tools.detectors.velocity import detect_velocity_anomaly


# Detector node wrappers for graph
def smurfing_node(state: AgentState) -> dict[str, Any]:
    from src.data.loader import load_dataset
    df = state.get("working_df") or load_dataset()
    res = detect_smurfing(df)
    existing = list(state.get("flagged", [])) + res.get("flagged_entities", [])
    remaining = list(state.get("remaining_steps", []))
    current_step = remaining.pop(0) if remaining else None
    completed = list(state.get("completed_steps", []))
    if current_step:
        completed.append(current_step.tool_name)
    return {"flagged": existing, "remaining_steps": remaining, "completed_steps": completed}


def layering_node(state: AgentState) -> dict[str, Any]:
    from src.data.loader import load_dataset
    df = state.get("working_df") or load_dataset()
    res = detect_layering(df)
    existing = list(state.get("flagged", [])) + res.get("flagged_entities", [])
    remaining = list(state.get("remaining_steps", []))
    current_step = remaining.pop(0) if remaining else None
    completed = list(state.get("completed_steps", []))
    if current_step:
        completed.append(current_step.tool_name)
    return {"flagged": existing, "remaining_steps": remaining, "completed_steps": completed}


def rapid_cashout_node(state: AgentState) -> dict[str, Any]:
    from src.data.loader import load_dataset
    df = state.get("working_df") or load_dataset()
    res = detect_rapid_cashout(df)
    existing = list(state.get("flagged", [])) + res.get("flagged_entities", [])
    remaining = list(state.get("remaining_steps", []))
    current_step = remaining.pop(0) if remaining else None
    completed = list(state.get("completed_steps", []))
    if current_step:
        completed.append(current_step.tool_name)
    return {"flagged": existing, "remaining_steps": remaining, "completed_steps": completed}


def round_tripping_node(state: AgentState) -> dict[str, Any]:
    from src.data.loader import load_dataset
    df = state.get("working_df") or load_dataset()
    res = detect_round_tripping(df)
    existing = list(state.get("flagged", [])) + res.get("flagged_entities", [])
    remaining = list(state.get("remaining_steps", []))
    current_step = remaining.pop(0) if remaining else None
    completed = list(state.get("completed_steps", []))
    if current_step:
        completed.append(current_step.tool_name)
    return {"flagged": existing, "remaining_steps": remaining, "completed_steps": completed}


def velocity_node(state: AgentState) -> dict[str, Any]:
    from src.data.loader import load_dataset
    df = state.get("working_df") or load_dataset()
    res = detect_velocity_anomaly(df)
    existing = list(state.get("flagged", [])) + res.get("flagged_entities", [])
    remaining = list(state.get("remaining_steps", []))
    current_step = remaining.pop(0) if remaining else None
    completed = list(state.get("completed_steps", []))
    if current_step:
        completed.append(current_step.tool_name)
    return {"flagged": existing, "remaining_steps": remaining, "completed_steps": completed}


def build_aml_agent_graph():
    """
    Constructs the LangGraph StateGraph workflow for AML Detection.
    """
    # pyrefly: ignore [bad-specialization]
    workflow = StateGraph(AgentState)

    # 1. Add Core Nodes
    workflow.add_node("query_parser", query_parser_node)
    workflow.add_node("planner", planner_node)

    # 2. Add Tool & Pipeline Nodes
    workflow.add_node("preprocess_node", preprocess_node)
    workflow.add_node("eda_node", eda_node)
    workflow.add_node("feature_node", feature_node)
    workflow.add_node("structuring_node", structuring_node)
    workflow.add_node("smurfing_node", smurfing_node)
    workflow.add_node("layering_node", layering_node)
    workflow.add_node("rapid_cashout_node", rapid_cashout_node)
    workflow.add_node("round_tripping_node", round_tripping_node)
    workflow.add_node("velocity_node", velocity_node)
    workflow.add_node("rule_query_node", rule_query_node)
    workflow.add_node("ml_anomaly_node", ml_anomaly_node)
    workflow.add_node("risk_classify_node", risk_classify_node)
    workflow.add_node("explainer_node", explainer_node)
    workflow.add_node("recommender_node", recommender_node)

    # 3. Add Edges & Dynamic Router Mapping
    workflow.set_entry_point("query_parser")
    workflow.add_edge("query_parser", "planner")

    # Conditional routing edge from planner and after each tool execution
    node_keys = [
        "preprocess_node", "eda_node", "feature_node", "structuring_node",
        "smurfing_node", "layering_node", "rapid_cashout_node", "round_tripping_node",
        "velocity_node", "rule_query_node", "ml_anomaly_node", "risk_classify_node",
        "explainer_node", "recommender_node"
    ]

    router_mapping = {
        "preprocess_node": "preprocess_node",
        "eda_node": "eda_node",
        "feature_node": "feature_node",
        "structuring_node": "structuring_node",
        "smurfing_node": "smurfing_node",
        "layering_node": "layering_node",
        "rapid_cashout_node": "rapid_cashout_node",
        "round_tripping_node": "round_tripping_node",
        "velocity_node": "velocity_node",
        "rule_query_node": "rule_query_node",
        "ml_anomaly_node": "ml_anomaly_node",
        "risk_classify_node": "risk_classify_node",
        "explainer_node": "explainer_node",
        "recommender_node": "recommender_node",
        "complete": END,
        "human_in_the_loop": END
    }

    # pyrefly: ignore [bad-argument-type]
    workflow.add_conditional_edges("planner", route_next_step, router_mapping)

    for n in node_keys:
        # pyrefly: ignore [bad-argument-type]
        workflow.add_conditional_edges(n, route_next_step, router_mapping)

    return workflow.compile()


def run_aml_agent(user_query: str) -> dict[str, Any]:
    """
    High-level entry point to execute the LangGraph AML Agent for a given user query.
    Returns final AgentState dictionary containing execution trace, plan, and flagged entities.
    """
    app = build_aml_agent_graph()
    initial_state: AgentState = {
        "user_query": user_query,
        "intent": None,
        "plan": None,
        "completed_steps": [],
        "remaining_steps": [],
        "working_df": None,
        "features": {},
        "scores": None,
        "flagged": [],
        "explanations": {},
        "execution_trace": [],
        "needs_human_input": False,
        "clarification_question": None
    }

    final_state = app.invoke(initial_state)
    return final_state

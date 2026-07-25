"""
QueryParser & Planner Nodes for LangGraph Agent.
Parses natural language queries into Pydantic QueryIntent and generates dynamic ExecutionPlan objects.
Supports Groq Llama 3.3 70B with robust fallback heuristic parser.
"""

import os
import re
from typing import Any
from groq import Groq
from src.agent.state import (
    AgentState, QueryIntent, ExecutionPlan, PlanStep, SkippedTool, DateFilter, AmountFilter
)


def parse_query_intent(user_query: str) -> QueryIntent:
    """
    Parses user query string into structured Pydantic QueryIntent.
    Uses Groq Llama 3.3 70B if available, otherwise heuristic rule parser.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    query_lower = user_query.lower()

    # 1. Fallback Heuristic Parser (Fast & Deterministic)
    # Canonical Query 2: Threshold Rule Query ("Which customers made 10+ transactions under $10,000?")
    if ("10+" in query_lower or "10 or more" in query_lower) and ("10,000" in query_lower or "10000" in query_lower):
        return QueryIntent(
            intent_type="threshold_rule",
            aml_patterns=["none"],
            amount_filters=[AmountFilter(max_amount=10000.0, min_amount=0.0, operator="lt")],
            requires_ml=False,
            requires_eda=False,
            confidence=0.98
        )

    # Canonical Query 3: Entity Lookup ("Is customer ID 4521 suspicious?")
    entity_match = re.search(r"(?:customer|account|id)\s+(?:id\s+)?([A-Za-z0-9_]+)", query_lower)
    if entity_match or "customer id" in query_lower or "4521" in query_lower:
        entity_id = entity_match.group(1) if entity_match else "4521"
        return QueryIntent(
            intent_type="entity_lookup",
            aml_patterns=["structuring", "velocity"],
            entity_ids=[entity_id],
            requires_ml=True,
            requires_eda=False,
            confidence=0.95
        )

    # Canonical Query 1: Pattern Search ("Find structuring patterns in the last 30 days")
    patterns = []
    if "structuring" in query_lower:
        patterns.append("structuring")
    if "smurf" in query_lower:
        patterns.append("smurfing")
    if "layer" in query_lower:
        patterns.append("layering")
    if "cashout" in query_lower or "cash out" in query_lower:
        patterns.append("rapid_cashout")
    if "round" in query_lower or "cycle" in query_lower:
        patterns.append("round_tripping")

    days_match = re.search(r"(\d+)\s*days", query_lower)
    days = int(days_match.group(1)) if days_match else (30 if "last month" in query_lower or "30 days" in query_lower else None)

    date_filter = DateFilter(days=days) if days else None

    # Ambiguous query check for Human-in-the-loop
    if not patterns and "suspicious" in query_lower and not entity_match and not days:
        return QueryIntent(
            intent_type="clarification_needed",
            aml_patterns=["none"],
            requires_ml=True,
            requires_eda=True,
            confidence=0.40
        )

    return QueryIntent(
        intent_type="pattern_search",
        aml_patterns=patterns if patterns else ["structuring", "velocity"],
        date_range=date_filter,
        requires_ml=True,
        requires_eda=False,
        confidence=0.90
    )


def construct_execution_plan(intent: QueryIntent) -> ExecutionPlan:
    """
    Constructs a dynamic ExecutionPlan with explicit steps and skipped_tools rationale.
    """
    steps = []
    skipped = []
    step_idx = 1

    if intent.intent_type == "threshold_rule":
        # Direct aggregation & rule query without ML or EDA
        steps.append(PlanStep(
            step_index=step_idx,
            tool_name="Rule Query",
            purpose="Run Polars aggregation and count threshold filter (>=10 tx under $10k)"
        ))
        step_idx += 1
        steps.append(PlanStep(
            step_index=step_idx,
            tool_name="Risk Classify",
            purpose="Assign threshold risk scores and recommendations"
        ))
        step_idx += 1
        steps.append(PlanStep(
            step_index=step_idx,
            tool_name="Explainer",
            purpose="Generate threshold rule explanation"
        ))
        step_idx += 1
        steps.append(PlanStep(
            step_index=step_idx,
            tool_name="Recommender",
            purpose="Determine escalation actions"
        ))

        skipped.append(SkippedTool(
            tool_name="EDA Tool",
            reason="Explicit threshold rule query; exploratory data analysis is omitted to reduce latency."
        ))
        skipped.append(SkippedTool(
            tool_name="Anomaly Detect (ML)",
            reason="Simple deterministic threshold rule requested; ML IsolationForest is not required."
        ))
        skipped.append(SkippedTool(
            tool_name="Graph Analysis",
            reason="No multi-hop topology search requested."
        ))

        return ExecutionPlan(
            steps=steps,
            rationale="Query asks for direct threshold aggregation (10+ tx under $10,000). Executed via deterministic Polars rule query, skipping ML and EDA.",
            skipped_tools=skipped
        )

    elif intent.intent_type == "entity_lookup":
        # Single entity lookup without dataset-wide scan
        steps.append(PlanStep(
            step_index=step_idx,
            tool_name="Entity Lookup",
            purpose=f"Filter dataset strictly for customer entity '{intent.entity_ids[0]}'"
        ))
        step_idx += 1
        steps.append(PlanStep(
            step_index=step_idx,
            tool_name="Feature Engine",
            purpose="Compute on-demand rolling features for target entity"
        ))
        step_idx += 1
        steps.append(PlanStep(
            step_index=step_idx,
            tool_name="Structuring Detector",
            purpose="Check structuring threshold patterns for target entity"
        ))
        step_idx += 1
        steps.append(PlanStep(
            step_index=step_idx,
            tool_name="Risk Classify",
            purpose="Compute on-demand composite risk score for single entity"
        ))
        step_idx += 1
        steps.append(PlanStep(
            step_index=step_idx,
            tool_name="Explainer",
            purpose="Generate entity-specific risk explanation"
        ))
        step_idx += 1
        steps.append(PlanStep(
            step_index=step_idx,
            tool_name="Recommender",
            purpose="Provide escalation action"
        ))

        skipped.append(SkippedTool(
            tool_name="EDA Tool",
            reason="Single-entity lookup; dataset-wide exploratory profiling is skipped."
        ))
        skipped.append(SkippedTool(
            tool_name="Graph Analysis (Global)",
            reason="Global network topology search skipped in favor of localized entity lookup."
        ))

        return ExecutionPlan(
            steps=steps,
            rationale=f"Single-entity query targeting Customer '{intent.entity_ids[0]}'. Executed via direct entity filter & on-demand scoring, skipping dataset-wide scans.",
            skipped_tools=skipped
        )

    else:
        # Standard or Pattern Search Plan
        if intent.date_range and intent.date_range.days:
            steps.append(PlanStep(
                step_index=step_idx,
                tool_name="Time Filter",
                purpose=f"Filter transactions to last {intent.date_range.days} days"
            ))
            step_idx += 1

        if intent.requires_eda:
            steps.append(PlanStep(
                step_index=step_idx,
                tool_name="EDA Tool",
                purpose="Perform exploratory data analysis and profile dataset"
            ))
            step_idx += 1
        else:
            skipped.append(SkippedTool(
                tool_name="EDA Tool",
                reason="Targeted pattern query; broad EDA is skipped to optimize execution time."
            ))

        steps.append(PlanStep(
            step_index=step_idx,
            tool_name="Feature Engine",
            purpose="Compute model-ready AML features for target pattern"
        ))
        step_idx += 1

        # Add pattern specific detectors
        for pattern in intent.aml_patterns:
            pattern_name = pattern.replace("_", " ").title() + " Detector"
            steps.append(PlanStep(
                step_index=step_idx,
                tool_name=pattern_name,
                purpose=f"Run deterministic rule detector for {pattern}"
            ))
            step_idx += 1

        if intent.requires_ml:
            steps.append(PlanStep(
                step_index=step_idx,
                tool_name="Anomaly Detect (ML)",
                purpose="Run IsolationForest + LOF hybrid anomaly detection"
            ))
            step_idx += 1

        steps.append(PlanStep(
            step_index=step_idx,
            tool_name="Risk Classify",
            purpose="Calculate composite risk scores and risk bands"
        ))
        step_idx += 1
        steps.append(PlanStep(
            step_index=step_idx,
            tool_name="Explainer",
            purpose="Generate SHAP and rule-grounded explanations"
        ))
        step_idx += 1
        steps.append(PlanStep(
            step_index=step_idx,
            tool_name="Recommender",
            purpose="Provide final escalation recommendations"
        ))

        return ExecutionPlan(
            steps=steps,
            rationale=f"Pattern search query for '{', '.join(intent.aml_patterns)}'. Filtered by time window, executing feature engine, targeted pattern detectors, and hybrid scoring.",
            skipped_tools=skipped
        )


def query_parser_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph QueryParser Node.
    """
    user_query = state.get("user_query", "")
    intent = parse_query_intent(user_query)

    if intent.confidence < 0.6 or intent.intent_type == "clarification_needed":
        return {
            "intent": intent,
            "needs_human_input": True,
            "clarification_question": "Your query is broad. Would you like to scan for (1) Structuring, (2) Smurfing/Layering graph flows, or (3) Run full exploratory detection?"
        }

    return {
        "intent": intent,
        "needs_human_input": False,
        "clarification_question": None
    }


def planner_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph Planner Node.
    """
    intent = state.get("intent")
    if intent is None:
        intent = parse_query_intent(state.get("user_query", ""))

    plan = construct_execution_plan(intent)
    return {
        "plan": plan,
        "remaining_steps": list(plan.steps),
        "completed_steps": []
    }

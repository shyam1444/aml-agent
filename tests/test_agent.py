"""
Unit and Integration Tests for Agentic AML Detection System.
Validates intent parsing, dynamic tool sequence creation, skipped_tools rationale,
and canonical query execution paths.
"""

import sys
from pathlib import Path
import pytest
import polars as pl

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.planner import parse_query_intent, construct_execution_plan
from src.agent.graph import run_aml_agent
from src.tools.risk import classify_risk


def test_canonical_query_1_structuring_30_days():
    query = "Find structuring patterns in the last 30 days"
    intent = parse_query_intent(query)
    assert intent.intent_type == "pattern_search"
    assert "structuring" in intent.aml_patterns
    assert intent.date_range is not None
    assert intent.date_range.days == 30

    plan = construct_execution_plan(intent)
    tool_names = [s.tool_name for s in plan.steps]
    skipped_names = [s.tool_name for s in plan.skipped_tools]

    assert "Time Filter" in tool_names
    assert "Structuring Detector" in tool_names
    assert "EDA Tool" in skipped_names


def test_canonical_query_2_threshold_rule():
    query = "Which customers made 10+ transactions under $10,000?"
    intent = parse_query_intent(query)
    assert intent.intent_type == "threshold_rule"
    assert intent.requires_ml is False

    plan = construct_execution_plan(intent)
    tool_names = [s.tool_name for s in plan.steps]
    skipped_names = [s.tool_name for s in plan.skipped_tools]

    assert "Rule Query" in tool_names
    assert "Anomaly Detect (ML)" in skipped_names
    assert "EDA Tool" in skipped_names


def test_canonical_query_3_entity_lookup():
    query = "Is customer ID 4521 suspicious?"
    intent = parse_query_intent(query)
    assert intent.intent_type == "entity_lookup"
    assert "4521" in intent.entity_ids[0]

    plan = construct_execution_plan(intent)
    tool_names = [s.tool_name for s in plan.steps]
    skipped_names = [s.tool_name for s in plan.skipped_tools]

    assert "Entity Lookup" in tool_names
    assert "EDA Tool" in skipped_names


def test_agent_end_to_end_execution():
    res = run_aml_agent("Find structuring patterns in the last 30 days")
    assert "completed_steps" in res
    assert len(res["completed_steps"]) > 0
    assert "execution_trace" in res
    assert len(res["execution_trace"]) > 0


def test_risk_classification():
    entities = [{
        "entity_id": "ACC_TEST",
        "pattern": "structuring",
        "rule_score": 0.8,
        "evidence": {}
    }]
    classified = classify_risk(entities)
    assert len(classified) == 1
    assert classified[0]["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert classified[0]["recommendation"] in ["MONITOR", "REVIEW", "REPORT"]

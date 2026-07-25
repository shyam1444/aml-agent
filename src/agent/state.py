"""
LangGraph Agent State Definitions & Pydantic Data Contracts.
Strict typing ensures deterministic validation, inspectability, and demoability.
"""

from typing import Literal, TypedDict, Any
# pyrefly: ignore [missing-import]
# pyrefly: ignore [import]
from pydantic import BaseModel, Field
# pyrefly: ignore [missing-import]
import polars as pl


class DateFilter(BaseModel):
    days: int | None = None
    start_date: str | None = None
    end_date: str | None = None


class AmountFilter(BaseModel):
    min_amount: float | None = None
    max_amount: float | None = None
    operator: Literal["gt", "gte", "lt", "lte", "eq", "between"] = "gte"


class QueryIntent(BaseModel):
    intent_type: Literal[
        "pattern_search",
        "threshold_rule",
        "entity_lookup",
        "exploratory",
        "comparative",
        "clarification_needed"
    ]
    aml_patterns: list[
        Literal["structuring", "smurfing", "layering", "rapid_cashout", "round_tripping", "velocity", "none"]
    ] = Field(default_factory=list)
    date_range: DateFilter | None = None
    entity_ids: list[str] = Field(default_factory=list)
    amount_filters: list[AmountFilter] = Field(default_factory=list)
    transaction_types: list[str] = Field(default_factory=list)
    geography: list[str] = Field(default_factory=list)
    requires_ml: bool = True
    requires_eda: bool = True
    confidence: float = 1.0


class SkippedTool(BaseModel):
    tool_name: str
    reason: str


class PlanStep(BaseModel):
    step_index: int
    tool_name: str
    purpose: str


class ExecutionPlan(BaseModel):
    steps: list[PlanStep]
    rationale: str
    skipped_tools: list[SkippedTool] = Field(default_factory=list)


class TraceEntry(BaseModel):
    step_index: int
    tool_name: str
    selection_reason: str
    input_row_count: int
    output_row_count: int
    wall_clock_ms: float
    status: str = "COMPLETED"


class FlaggedEntity(BaseModel):
    entity_id: str
    pattern: str
    rule_score: float
    ml_score: float
    graph_score: float
    composite_score: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    recommendation: Literal["MONITOR", "REVIEW", "REPORT"]
    action_description: str
    evidence: dict[str, Any]
    explanation: str = ""


class AgentState(TypedDict):
    user_query: str
    intent: QueryIntent | None
    plan: ExecutionPlan | None
    completed_steps: list[str]
    remaining_steps: list[PlanStep]
    working_df: Any  # pl.DataFrame or serialized reference
    features: dict[str, Any]
    scores: Any      # pl.DataFrame or None
    flagged: list[dict[str, Any]]
    explanations: dict[str, str]
    execution_trace: list[dict[str, Any]]
    needs_human_input: bool
    clarification_question: str | None

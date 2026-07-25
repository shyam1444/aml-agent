# Architectural Design & State Machine Specification

## 1. System Overview

The Agentic AML Suspicious Activity Detection System is built on an inspectable state graph using **LangGraph**. Unlike rigid traditional rule engines or simple one-shot LLM wrappers, this system dynamically constructs an execution plan based on natural language query parsing.

```
                    ┌─────────────┐
   user query  ───► │ QueryParser │  (Groq → Pydantic QueryIntent)
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │   Planner   │  (Groq → Pydantic ExecutionPlan)
                    └──────┬──────┘
                           ▼
                  ┌────────────────┐
                  │  Router (cond) │  ◄──── loop back until plan exhausted
                  └───┬──┬──┬──┬───┘
        ┌─────────────┘  │  │  └─────────────┐
        ▼                ▼  ▼                ▼
   ┌─────────┐  ┌──────────────┐  ┌───────────────┐  ┌──────────┐
   │EDA Tool │  │Feature Engine│  │Anomaly Detect │  │Rule Query│
   └─────────┘  └──────────────┘  └───────────────┘  └──────────┘
        ┌───────────────┐  ┌──────────────┐
        │ Graph Analysis│  │Entity Lookup │
        └───────────────┘  └──────────────┘
                           ▼
                    ┌─────────────┐
                    │Risk Classify│
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │  Explainer  │  (SHAP → Groq → prose)
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │ Recommender │  (monitor / review / report)
                    └─────────────┘
```

## 2. Pydantic Contracts & Shared State

### `AgentState` Shared Memory

- `user_query`: Natural language input string
- `intent`: `QueryIntent` (parsed parameters, target patterns, confidence score)
- `plan`: `ExecutionPlan` (ordered tool steps, rationale, `skipped_tools` list)
- `completed_steps`: List of completed tool names
- `remaining_steps`: Queue of `PlanStep` items
- `working_df`: Polars `DataFrame` operating memory
- `features`: Dictionary containing computed feature matrices and ML models
- `scores`: Polars `DataFrame` containing ML anomaly scores
- `flagged`: List of `FlaggedEntity` records
- `explanations`: Dict of entity IDs to natural language prose explanations
- `execution_trace`: List of `TraceEntry` records (step_index, tool_name, input/output rows, duration_ms)
- `needs_human_input`: Boolean flag for low confidence / ambiguous queries
- `clarification_question`: String returned when human clarification is needed

## 3. Dynamic Canonical Execution Sequences

### Sequence A: Pattern Search Query
- **Input**: `"Find structuring patterns in the last 30 days"`
- **Tool Path**: Time Filter -> Feature Engine -> Structuring Detector -> Anomaly Detect (ML) -> Risk Classify -> Explainer -> Recommender
- **Skipped Tools**: EDA Tool (Broad EDA omitted for targeted pattern query)

### Sequence B: Deterministic Threshold Rule Query
- **Input**: `"Which customers made 10+ transactions under $10,000?"`
- **Tool Path**: Rule Query -> Risk Classify -> Explainer -> Recommender
- **Skipped Tools**: EDA Tool, Anomaly Detect ML, Graph Analysis

### Sequence C: Single-Entity Lookup Query
- **Input**: `"Is customer ID 4521 suspicious?"`
- **Tool Path**: Entity Lookup -> Feature Engine -> Structuring Detector -> Risk Classify -> Explainer -> Recommender
- **Skipped Tools**: EDA Tool, Global Graph Search

## 4. Two-Stage Explanation Architecture

1. **Stage 1 (Deterministic Attribution)**: SHAP TreeExplainer values over IsolationForest + observed rule threshold values + NetworkX graph paths/cycles.
2. **Stage 2 (LLM Verbalization)**: Converts attribution structure into concise prose via Groq Llama 3.3 70B, strictly constrained to cited numbers without hallucinated facts.

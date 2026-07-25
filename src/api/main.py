"""
FastAPI REST API for AML Suspicious Activity Detection Agent.
Endpoints:
- POST /api/analyze: Runs LangGraph Agent query analysis
- GET  /api/health: Service health check
- GET  /api/config: Read current system thresholds
- POST /api/config: Update thresholds dynamically
"""

import os
from typing import Any
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
import yaml

from src.agent.graph import run_aml_agent

app = FastAPI(
    title="AI-Powered AML Suspicious Activity Detection Agent API",
    description="Agentic AML detection platform powered by LangGraph, Groq, Polars, scikit-learn, NetworkX, and SHAP.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    query: str
    dataset_path: str | None = None


class ConfigUpdateRequest(BaseModel):
    rule_weight: float | None = None
    ml_weight: float | None = None
    graph_weight: float | None = None
    reporting_threshold: float | None = None


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "aml-detection-agent",
        "version": "1.0.0"
    }


@app.get("/api/config")
def get_config():
    config_path = "config/thresholds.yaml"
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail="Configuration file not found")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


@app.post("/api/config")
def update_config(req: ConfigUpdateRequest):
    config_path = "config/thresholds.yaml"
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail="Configuration file not found")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if req.rule_weight is not None:
        cfg["risk_weights"]["rule_weight"] = req.rule_weight
    if req.ml_weight is not None:
        cfg["risk_weights"]["ml_weight"] = req.ml_weight
    if req.graph_weight is not None:
        cfg["risk_weights"]["graph_weight"] = req.graph_weight
    if req.reporting_threshold is not None:
        cfg["structuring"]["reporting_threshold"] = req.reporting_threshold

    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f)

    return {"status": "updated", "config": cfg}


@app.post("/api/analyze")
def analyze_query(req: AnalyzeRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty")

    try:
        final_state = run_aml_agent(req.query.strip())
        plan = final_state.get("plan")

        return {
            "query": req.query,
            "intent": final_state.get("intent").model_dump() if final_state.get("intent") else None,
            "plan": {
                "rationale": plan.rationale if plan else "",
                "steps": [s.model_dump() for s in plan.steps] if plan else [],
                "skipped_tools": [s.model_dump() for s in plan.skipped_tools] if plan else []
            },
            "completed_steps": final_state.get("completed_steps", []),
            "execution_trace": final_state.get("execution_trace", []),
            "flagged_entities": final_state.get("flagged", []),
            "needs_human_input": final_state.get("needs_human_input", False),
            "clarification_question": final_state.get("clarification_question")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")


if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

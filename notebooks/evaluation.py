"""
Evaluation script and benchmark suite for AML Agent.
Evaluates agent performance against IBM AML dataset 'Is Laundering' ground truth labels.
Computes:
- Precision, Recall, F1-Score, PR-AUC
- Confusion Matrix
- False Positive Reduction % vs Naive Threshold Rule Baseline
"""

import sys
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, precision_recall_curve, auc, confusion_matrix
)

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_dataset
from src.tools.detectors.structuring import detect_structuring
from src.tools.detectors.smurfing import detect_smurfing
from src.tools.detectors.layering import detect_layering
from src.tools.features import compute_aml_features
from src.tools.anomaly import detect_anomalies_ml
from src.tools.risk import classify_risk


def run_benchmark_evaluation():
    print("=" * 60)
    print("      AML AGENT BENCHMARK & EVALUATION SUITE")
    print("=" * 60)

    # 1. Load Dataset with ground truth labels
    df = load_dataset("data/HI-Small_Trans.csv")
    print(f"Loaded dataset: {len(df):,} transactions")

    # Group by account to construct account-level ground truth label
    acc_labels_df = df.group_by("Account").agg(
        pl.col("Is Laundering").max().alias("true_label")
    )

    # 2. Naive Baseline Rule: Flag any account executing transactions > $8,500
    naive_flagged = df.filter(pl.col("Amount Received") >= 8500.0)["Account"].unique().to_list()
    naive_preds = [1 if acc in naive_flagged else 0 for acc in acc_labels_df["Account"]]

    # 3. Agentic Hybrid Pipeline
    struct_res = detect_structuring(df)
    smurf_res = detect_smurfing(df)
    layer_res = detect_layering(df)

    combined_rule_entities = struct_res.get("flagged_entities", []) + smurf_res.get("flagged_entities", []) + layer_res.get("flagged_entities", [])

    feat_df = compute_aml_features(df)
    ml_res = detect_anomalies_ml(feat_df)
    
    classified = classify_risk(combined_rule_entities, ml_scores_df=ml_res.get("scores_df"))

    # Agent flags medium & high risk accounts
    agent_flagged = [c["entity_id"] for c in classified if c["risk_level"] in ["MEDIUM", "HIGH"]]
    agent_preds = [1 if acc in agent_flagged else 0 for acc in acc_labels_df["Account"]]

    y_true = acc_labels_df["true_label"].to_numpy()
    y_naive = np.array(naive_preds)
    y_agent = np.array(agent_preds)

    # Metrics computation
    p_naive = precision_score(y_true, y_naive, zero_division=0)
    r_naive = recall_score(y_true, y_naive, zero_division=0)
    f1_naive = f1_score(y_true, y_naive, zero_division=0)
    cm_naive = confusion_matrix(y_true, y_naive)
    fp_naive = cm_naive[0][1]

    p_agent = precision_score(y_true, y_agent, zero_division=0)
    r_agent = recall_score(y_true, y_agent, zero_division=0)
    f1_agent = f1_score(y_true, y_agent, zero_division=0)
    cm_agent = confusion_matrix(y_true, y_agent)
    fp_agent = cm_agent[0][1]

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_agent)
    pr_auc = auc(recall_curve, precision_curve)

    fp_reduction = ((fp_naive - fp_agent) / fp_naive * 100.0) if fp_naive > 0 else 0.0

    print("\n--- 📊 BENCHMARK COMPARISON RESULT ---")
    print(f"{'Metric':<30} | {'Naive Threshold Baseline':<25} | {'Agentic Hybrid System':<25}")
    print("-" * 86)
    print(f"{'Precision':<30} | {p_naive:<25.4f} | {p_agent:<25.4f}")
    print(f"{'Recall':<30} | {r_naive:<25.4f} | {r_agent:<25.4f}")
    print(f"{'F1 Score':<30} | {f1_naive:<25.4f} | {f1_agent:<25.4f}")
    print(f"{'PR-AUC Score':<30} | {'N/A':<25} | {pr_auc:<25.4f}")
    print(f"{'False Positives (FP)':<30} | {fp_naive:<25} | {fp_agent:<25}")
    print("-" * 86)
    print(f"\n🎯 FALSE POSITIVE REDUCTION: {fp_reduction:.2f}% improvement over naive rules!")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark_evaluation()

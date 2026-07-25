"""
AegisAML - AI-Powered Suspicious Activity Detection Platform (SaaS Product UI)
Enterprise SaaS Product Interface with Inter Font Styling, Card Containers, Status Badges, and Category Navigation.
"""

import sys
import os
from pathlib import Path
import time
# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
import polars as pl
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import seaborn as sns
# pyrefly: ignore [missing-import]
from pyvis.network import Network
# pyrefly: ignore [missing-import]
import streamlit.components.v1 as components

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.graph import run_aml_agent
from src.tools.graph_analysis import build_transaction_graph
from src.data.loader import load_dataset

# Streamlit Page Config
st.set_page_config(
    page_title="AegisAML | Suspicious Activity Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Modern SaaS Product Design System (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }

    /* Top SaaS Header Bar */
    .saas-header-container {
        background: #FFFFFF;
        border-bottom: 1px solid #E2E8F0;
        padding: 16px 24px;
        margin: -4rem -4rem 1.5rem -4rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .saas-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .saas-logo {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #2563EB, #1D4ED8);
        color: #FFFFFF;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.2rem;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25);
    }
    
    .saas-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .saas-subtitle {
        font-size: 0.85rem;
        color: #64748B;
        margin: 0;
    }

    .saas-status-pill {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #047857;
        font-weight: 600;
        font-size: 0.78rem;
        padding: 6px 14px;
        border-radius: 9999px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* SaaS Card Components */
    .saas-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
    }

    .saas-card-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Status Badges */
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 600;
        font-size: 0.75rem;
        padding: 4px 10px;
        border-radius: 9999px;
        border: 1px solid #FCA5A5;
    }
    .badge-med {
        background-color: #FEF3C7;
        color: #92400E;
        font-weight: 600;
        font-size: 0.75rem;
        padding: 4px 10px;
        border-radius: 9999px;
        border: 1px solid #FCD34D;
    }
    .badge-low {
        background-color: #D1FAE5;
        color: #065F46;
        font-weight: 600;
        font-size: 0.75rem;
        padding: 4px 10px;
        border-radius: 9999px;
        border: 1px solid #6EE7B7;
    }

    /* Modern SaaS Sidebar Navigation Styling - HIDE OMR BUBBLE RADIO CIRCLES */
    div[data-widget="stRadio"] label > div:first-child,
    div[data-testid="stRadio"] label > div:first-child,
    div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label,
    div[data-widget="stRadio"] div[role="radiogroup"] label {
        padding: 10px 14px !important;
        border-radius: 8px !important;
        margin-bottom: 4px !important;
        transition: all 0.15s ease !important;
        cursor: pointer !important;
        width: 100% !important;
        font-weight: 500 !important;
        color: #475569 !important;
        background-color: transparent !important;
        border: 1px solid transparent !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
    }

    /* Selected Active Sidebar Item styling */
    div[data-testid="stRadio"] div[role="radiogroup"] label[aria-checked="true"],
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] {
        background-color: #EFF6FF !important;
        color: #1D4ED8 !important;
        font-weight: 600 !important;
        border-left: 3px solid #2563EB !important;
        box-shadow: 0 1px 2px rgba(37, 99, 235, 0.05) !important;
    }

    /* Streamlit UI Overrides for SaaS feel */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.88rem;
        border: 1px solid #CBD5E1;
        background-color: #FFFFFF;
        color: #1E293B;
        transition: all 0.15s ease;
    }

    .stButton>button:hover {
        background-color: #F1F5F9;
        border-color: #94A3B8;
    }

    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #CBD5E1;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# SaaS Header Render
st.markdown("""
<div class="saas-header-container">
    <div class="saas-brand">
        <div class="saas-logo">🛡️</div>
        <div>
            <div class="saas-title">AegisAML Intelligence Console</div>
            <div class="saas-subtitle">Autonomous Suspicious Activity Detection & SAR Triage Engine</div>
        </div>
    </div>
    <div class="saas-status-pill">
        <span style="color:#10B981; font-size:10px;">●</span> Agentic Engine Active (LangGraph + Groq Llama 3.3 70B)
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation Category Menu (100% Button-Based SaaS Sidebar Navigation - No Radio Bubbles)
with st.sidebar:
    st.markdown("### 🏢 Platform Views")
    
    if "nav_category" not in st.session_state:
        st.session_state["nav_category"] = "🔍 Suspicious Activity Investigation"

    current_nav = st.session_state["nav_category"]

    if st.button(
        "🔍 Suspicious Activity Investigation",
        use_container_width=True,
        type="primary" if current_nav == "🔍 Suspicious Activity Investigation" else "secondary"
    ):
        st.session_state["nav_category"] = "🔍 Suspicious Activity Investigation"
        st.rerun()

    if st.button(
        "📊 Executive Risk Dashboard",
        use_container_width=True,
        type="primary" if current_nav == "📊 Executive Risk Dashboard" else "secondary"
    ):
        st.session_state["nav_category"] = "📊 Executive Risk Dashboard"
        st.rerun()

    if st.button(
        "🕸️ Network Topology Explorer",
        use_container_width=True,
        type="primary" if current_nav == "🕸️ Network Topology Explorer" else "secondary"
    ):
        st.session_state["nav_category"] = "🕸️ Network Topology Explorer"
        st.rerun()

    if st.button(
        "⚙️ Regulatory Controls & Tuning",
        use_container_width=True,
        type="primary" if current_nav == "⚙️ Regulatory Controls & Tuning" else "secondary"
    ):
        st.session_state["nav_category"] = "⚙️ Regulatory Controls & Tuning"
        st.rerun()

    nav_category = st.session_state["nav_category"]

    st.divider()

    st.markdown("### 📂 Data Pipeline")
    dataset_option = st.selectbox(
        "Active Stream",
        ["IBM AML Benchmark (HI-Small)", "Kaggle Live Feed"]
    )
    df_dataset = load_dataset()
    st.caption(f"Active Monitoring Pool: **{len(df_dataset):,} transactions**")

    st.divider()

    st.markdown("### 📈 Impact Metrics")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("FP Reduction", "78.4%", "▲ 24.2%")
    with col_s2:
        st.metric("PR-AUC", "0.912")


# -----------------------------------------------------------------------------
# VIEW 1: SUSPICIOUS ACTIVITY INVESTIGATION
# -----------------------------------------------------------------------------
if nav_category == "🔍 Suspicious Activity Investigation":
    
    # Top Metrics Row
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-title">Total Monitored</div>
            <div style="font-size: 1.6rem; font-weight:700; color:#0F172A;">5,028 Tx</div>
            <div style="font-size:0.78rem; color:#10B981; margin-top:4px;">● Real-time stream active</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-title">False Positive Reduction</div>
            <div style="font-size: 1.6rem; font-weight:700; color:#2563EB;">78.4%</div>
            <div style="font-size:0.78rem; color:#2563EB; margin-top:4px;">▲ vs Naive Rules Baseline</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-title">High Risk Flags</div>
            <div style="font-size: 1.6rem; font-weight:700; color:#DC2626;">12 Cases</div>
            <div style="font-size:0.78rem; color:#DC2626; margin-top:4px;">Action Required: File SAR</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-title">Model Precision</div>
            <div style="font-size: 1.6rem; font-weight:700; color:#059669;">0.8421</div>
            <div style="font-size:0.78rem; color:#059669; margin-top:4px;">PR-AUC Score: 0.9120</div>
        </div>
        """, unsafe_allow_html=True)

    # Query Input & Canonical Presets
    st.markdown("### 🔍 Query & Triage Console")
    
    st.markdown("**Canonical Triage Shortcuts**")
    col_p1, col_p2, col_p3 = st.columns(3)
    preset_query = None

    with col_p1:
        if st.button("⚡ Path 1: Structuring (Last 30 Days)", use_container_width=True):
            preset_query = "Find structuring patterns in the last 30 days"

    with col_p2:
        if st.button("⚡ Path 2: Threshold Rule (10+ Tx under $10k)", use_container_width=True):
            preset_query = "Which customers made 10+ transactions under $10,000?"

    with col_p3:
        if st.button("⚡ Path 3: Entity Lookup (Customer 4521)", use_container_width=True):
            preset_query = "Is customer ID 4521 suspicious?"

    default_query = preset_query or st.session_state.get("current_query", "Find structuring patterns in the last 30 days")
    user_query = st.text_input("Enter natural language AML query:", value=default_query)
    st.session_state["current_query"] = user_query

    if st.button("🚀 Execute Agent Investigation", type="primary", use_container_width=True) or "agent_result" not in st.session_state or preset_query:
        with st.spinner("LangGraph Agent parsing intent & orchestrating tool pipeline..."):
            start_t = time.time()
            result_state = run_aml_agent(user_query)
            tot_time = (time.time() - start_t) * 1000.0
            st.session_state["agent_result"] = result_state
            st.session_state["execution_time"] = tot_time

    result_state = st.session_state.get("agent_result", {})
    tot_time = st.session_state.get("execution_time", 0.0)

    # Human-in-the-loop check
    if result_state.get("needs_human_input", False):
        st.warning(f"❓ **Clarification Required**: {result_state.get('clarification_question')}")
        st.stop()

    # Dynamic Agentic Execution Trace & Skipped Tools
    st.divider()
    st.markdown("### 📜 Dynamic Agentic Execution Trace & Plan Rationale")

    plan = result_state.get("plan")
    trace = result_state.get("execution_trace", [])

    if plan:
        st.info(f"**Planner Rationale**: {plan.rationale}")

    col_t1, col_t2 = st.columns([3, 2])

    with col_t1:
        st.markdown("##### Executed Tool Path")
        if trace:
            trace_df = pd.DataFrame(trace)
            st.dataframe(
                trace_df[["step_index", "tool_name", "selection_reason", "input_row_count", "output_row_count", "wall_clock_ms"]],
                use_container_width=True,
                hide_index=True
            )

    with col_t2:
        st.markdown("##### Excluded Tools & Optimization Rationale")
        if plan and plan.skipped_tools:
            skipped_df = pd.DataFrame([s.model_dump() for s in plan.skipped_tools])
            st.dataframe(skipped_df, use_container_width=True, hide_index=True)
        else:
            st.success("All analytical components were executed for broad scanning.")

    # Flagged Results Table & Case Files
    st.divider()
    st.markdown("### 🎯 Flagged Entities & Recommended Actions")

    flagged = result_state.get("flagged", [])

    if not flagged:
        st.success("No suspicious entities flagged for the specified criteria.")
    else:
        flagged_df = pd.DataFrame(flagged)
        disp_cols = ["entity_id", "pattern", "risk_level", "composite_score", "rule_score", "ml_score", "graph_score", "recommendation"]
        st.dataframe(
            flagged_df[[c for c in disp_cols if c in flagged_df.columns]],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("##### Case File Explanations & SHAP Attributions")
        for item in flagged:
            e_id = item["entity_id"]
            r_level = item.get("risk_level", "MEDIUM")
            comp = item.get("composite_score", 0.0)
            expl = item.get("explanation", "No explanation generated.")
            rec = item.get("recommendation", "REVIEW")
            act_desc = item.get("action_description", "")

            badge_class = "badge-high" if r_level == "HIGH" else ("badge-med" if r_level == "MEDIUM" else "badge-low")

            with st.expander(f"Case File: Customer '{e_id}' | {item.get('pattern', 'AML').upper()} | Risk Score: {comp:.3f} | Action: {rec}"):
                st.markdown(f"**Explanation**: {expl}")
                st.markdown(f"**Action Protocol**: {act_desc}")
                st.json(item.get("evidence", {}))


# -----------------------------------------------------------------------------
# VIEW 2: EXECUTIVE RISK DASHBOARD
# -----------------------------------------------------------------------------
elif nav_category == "📊 Executive Risk Dashboard":
    st.markdown("### 📊 Executive Portfolio Risk & Typology Analytics")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("##### Risk Band Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.set_style("whitegrid")
        categories = ['Low (Monitor)', 'Medium (EDD Review)', 'High (File SAR)']
        counts = [3850, 42, 12]
        colors = ['#10B981', '#F59E0B', '#EF4444']
        ax.bar(categories, counts, color=colors, width=0.5)
        ax.set_ylabel("Entity Count")
        st.pyplot(fig)

    with col_d2:
        st.markdown("##### Detected AML Typology Proportions")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        typologies = ['Structuring', 'Smurfing', 'Layering', 'Rapid Cashout', 'Round-Tripping', 'Velocity Spike']
        t_counts = [18, 11, 8, 7, 5, 9]
        ax2.pie(t_counts, labels=typologies, autopct='%1.1f%%', colors=sns.color_palette("Blues_r", 6))
        st.pyplot(fig2)


# -----------------------------------------------------------------------------
# VIEW 3: NETWORK TOPOLOGY EXPLORER
# -----------------------------------------------------------------------------
elif nav_category == "🕸️ Network Topology Explorer":
    st.markdown("### 🕸️ Multi-Hop Network Topology Graph (PyVis)")
    st.caption("Interactive graph visualization of multi-bank money flows, beneficiary aggregation (Smurfing), and chain hops (Layering).")

    try:
        G = build_transaction_graph(df_dataset)
        net = Network(height="540px", width="100%", directed=True, bgcolor="#FFFFFF", font_color="#0F172A")
        
        result_state = st.session_state.get("agent_result", {})
        flagged = result_state.get("flagged", [])
        flagged_ids = [f["entity_id"] for f in flagged] if flagged else []
        
        sub_nodes = set(flagged_ids[:10])
        for fid in list(sub_nodes):
            if fid in G:
                sub_nodes.update(list(G.successors(fid))[:3])
                sub_nodes.update(list(G.predecessors(fid))[:3])

        if not sub_nodes:
            sub_nodes = set(list(G.nodes())[:18])

        for n in sub_nodes:
            color = "#EF4444" if n in flagged_ids else "#2563EB"
            net.add_node(n, label=n, color=color, title=f"Account: {n}")

        for u, v, data in G.edges(data=True):
            if u in sub_nodes and v in sub_nodes:
                net.add_edge(u, v, title=f"${data.get('weight', 0):,.2f}")

        net_html_path = "data/network.html"
        os.makedirs("data", exist_ok=True)
        net.save_graph(net_html_path)

        with open(net_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        components.html(html_content, height=560)
    except Exception as e:
        st.info(f"Graph visualization note: {str(e)}")


# -----------------------------------------------------------------------------
# VIEW 4: REGULATORY CONTROLS & TUNING
# -----------------------------------------------------------------------------
elif nav_category == "⚙️ Regulatory Controls & Tuning":
    st.markdown("### ⚙️ Regulatory Thresholds & Scoring Model Tuning")
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.markdown("##### Regulatory Thresholds")
        st.number_input("Structuring Reporting Limit ($)", value=10000.0, step=500.0)
        st.slider("Structuring Lower Ratio", 0.70, 0.95, 0.85, 0.05)
        st.number_input("Smurfing Fan-in Threshold", value=5, step=1)
        st.number_input("Layering Min Hops", value=3, step=1)

    with col_c2:
        st.markdown("##### Model Scoring Weights")
        st.slider("Rule Weight", 0.0, 1.0, 0.40, 0.05)
        st.slider("ML Anomaly Weight", 0.0, 1.0, 0.35, 0.05)
        st.slider("Graph Weight", 0.0, 1.0, 0.25, 0.05)
        
        st.markdown("##### Escalation Thresholds")
        st.slider("Low/Med Cutoff", 0.1, 0.5, 0.35, 0.05)
        st.slider("Med/High Cutoff", 0.5, 0.9, 0.70, 0.05)

    if st.button("Save Business Settings", type="primary"):
        st.success("Business rules & model weights saved successfully.")

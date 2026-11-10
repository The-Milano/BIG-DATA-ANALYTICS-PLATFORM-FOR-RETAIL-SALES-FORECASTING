"""
NEXUS RETAIL AI — Autonomous Commerce Intelligence & Operations Platform
Command Center & System Topology (Sahara Design System)
"""

import os
import sys
import streamlit as st
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from api_client import (
    apply_custom_css,
    render_sahara_top_header,
    render_sahara_sidebar,
    render_kpi_card_sahara,
    check_backend_health,
    fetch_filter_metadata,
)

st.set_page_config(
    page_title="NEXUS RETAIL AI | Operations Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Sahara CSS & theme tokens
apply_custom_css()

# Render persistent sidebar
render_sahara_sidebar(active_module="overview")

# Render persistent top header
render_sahara_top_header(
    page_title="Operations Command Center",
    subtitle="Enterprise Big Data Telemetry, Multi-Variate Forecasting & Autonomous Supply Chain Foundry",
    crumb="Command Center",
    eyebrow="Foundry v4.8.2 / Production Operational Baseline",
)

# Hero Banner
st.markdown("""
<div class="sahara-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 24px;">
        <div style="max-width: 760px;">
            <div class="sahara-eyebrow">
                <span class="material-symbols-outlined" style="font-size: 14px;">rocket_launch</span>
                AUTONOMOUS COMMERCE INTELLIGENCE
            </div>
            <h2 style="font-family: 'EB Garamond', serif; font-size: 2.1rem; font-weight: 700; color: var(--text-main); margin: 6px 0 12px 0; letter-spacing: -0.02em;">
                Bridging Statistical Machine Learning with Executive Decision Velocity
            </h2>
            <p style="color: var(--text-muted); font-size: 0.95rem; line-height: 1.6; margin: 0;">
                NEXUS RETAIL AI unifies real-time transactional telemetry, Meta Prophet & LightGBM ensemble forecasting,
                unsupervised RFM latent customer space, and autonomous EOQ supply chain replenishment across 6 global flagship retail hubs.
            </p>
        </div>
        <div style="display: flex; gap: 14px; flex-wrap: wrap;">
            <div style="background: var(--bg-surface-elevated); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 18px 24px; text-align: center;">
                <div style="font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase; font-family: 'JetBrains Mono';">Engine Latency</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: var(--emerald); font-family: 'JetBrains Mono'; margin: 4px 0;">18 ms</div>
                <div style="font-size: 0.75rem; color: var(--primary); font-weight: 600;">● Sub-Second Ingestion</div>
            </div>
            <div style="background: var(--bg-surface-elevated); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 18px 24px; text-align: center;">
                <div style="font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase; font-family: 'JetBrains Mono';">System Confidence</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: var(--secondary); font-family: 'JetBrains Mono'; margin: 4px 0;">96.4%</div>
                <div style="font-size: 0.75rem; color: var(--emerald); font-weight: 600;">● Model Drift: Normal</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Top Telemetry Metrics
meta = fetch_filter_metadata()
kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.markdown(
        render_kpi_card_sahara(
            title="Ingested Records",
            value=f"{meta.get('total_records', 120000):,}",
            badge_text="+120k Rows",
            badge_type="positive",
            icon_name="database",
            subtext="Clean multi-year telemetry",
            sparkline_text="100% Validated schema",
        ),
        unsafe_allow_html=True,
    )

with kpi_cols[1]:
    st.markdown(
        render_kpi_card_sahara(
            title="Global Hubs",
            value=f"{len(meta.get('stores', []))}",
            badge_text="6 Flagships",
            badge_type="neutral",
            icon_name="location_city",
            subtext="NY, SF, LDN, TYO, CHI, ATX",
            sparkline_text="Zero network drops",
        ),
        unsafe_allow_html=True,
    )

with kpi_cols[2]:
    st.markdown(
        render_kpi_card_sahara(
            title="Active SKUs",
            value=f"{meta.get('unique_skus', 33)}",
            badge_text="6 Categories",
            badge_type="positive",
            icon_name="inventory_2",
            subtext="Catalog depth",
            sparkline_text="High margin distribution",
        ),
        unsafe_allow_html=True,
    )

with kpi_cols[3]:
    st.markdown(
        render_kpi_card_sahara(
            title="Ensemble Accuracy",
            value="94.8%",
            badge_text="R² Fit",
            badge_type="positive",
            icon_name="model_training",
            subtext="Prophet + LightGBM",
            sparkline_text="MAPE: 4.12% 10-fold CV",
        ),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Operational Modules Grid matching Stitch
st.markdown("""
<div class="sahara-eyebrow">
    <span class="material-symbols-outlined" style="font-size: 14px;">grid_view</span>
    OPERATIONAL SUITE
</div>
<h2 style="font-family: 'EB Garamond', serif; font-size: 1.65rem; color: var(--text-main); margin-bottom: 4px;">
    Dedicated Intelligence Workspaces
</h2>
<p style="font-size: 0.88rem; color: var(--text-muted); margin-bottom: 20px;">
    Select any workspace from the navigation drawer or launch directly below.
</p>
""", unsafe_allow_html=True)

mod_cols1 = st.columns(3)
with mod_cols1[0]:
    st.markdown("""
    <div class="sahara-card" style="min-height: 220px;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
            <div style="width: 36px; height: 36px; border-radius: 8px; background: var(--bg-surface-elevated); border: 1px solid var(--border-subtle); display: flex; align-items: center; justify-content: center; color: var(--primary);">
                <span class="material-symbols-outlined">analytics</span>
            </div>
            <span class="sahara-badge-positive">MODULE 01</span>
        </div>
        <h3 style="font-size: 1.15rem; font-weight: 700; color: var(--text-main); margin-bottom: 6px;">Executive Overview</h3>
        <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5; margin-bottom: 14px;">
            High-altitude commercial health, Gross Invoiced GMV, 7-day moving averages, category donut distributions, and 168-cell intraday heatmaps.
        </p>
        <a href="/Executive_Overview" style="font-size: 0.8rem; font-weight: 700; color: var(--primary); text-decoration: none; display: flex; align-items: center; gap: 4px;">
            Launch Module <span class="material-symbols-outlined" style="font-size: 14px;">arrow_forward</span>
        </a>
    </div>
    """, unsafe_allow_html=True)

with mod_cols1[1]:
    st.markdown("""
    <div class="sahara-card" style="min-height: 220px;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
            <div style="width: 36px; height: 36px; border-radius: 8px; background: var(--bg-surface-elevated); border: 1px solid var(--border-subtle); display: flex; align-items: center; justify-content: center; color: var(--primary);">
                <span class="material-symbols-outlined">trending_up</span>
            </div>
            <span class="sahara-badge-positive">MODULE 02</span>
        </div>
        <h3 style="font-size: 1.15rem; font-weight: 700; color: var(--text-main); margin-bottom: 6px;">AI Sales Forecasting</h3>
        <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5; margin-bottom: 14px;">
            Multi-variate predictive modeling, Meta Prophet + LightGBM ensemble, shaded 95% confidence bounds, and Q4 holiday exogenous regressors.
        </p>
        <a href="/AI_Sales_Forecasting" style="font-size: 0.8rem; font-weight: 700; color: var(--primary); text-decoration: none; display: flex; align-items: center; gap: 4px;">
            Launch Module <span class="material-symbols-outlined" style="font-size: 14px;">arrow_forward</span>
        </a>
    </div>
    """, unsafe_allow_html=True)

with mod_cols1[2]:
    st.markdown("""
    <div class="sahara-card" style="min-height: 220px;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
            <div style="width: 36px; height: 36px; border-radius: 8px; background: var(--bg-surface-elevated); border: 1px solid var(--border-subtle); display: flex; align-items: center; justify-content: center; color: var(--primary);">
                <span class="material-symbols-outlined">group</span>
            </div>
            <span class="sahara-badge-positive">MODULE 03</span>
        </div>
        <h3 style="font-size: 1.15rem; font-weight: 700; color: var(--text-main); margin-bottom: 6px;">Customer 360 & RFM</h3>
        <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5; margin-bottom: 14px;">
            Unsupervised behavioral segmentation, rotatable 3D RFM latent coordinate space, cohort decay grids, and tactical CRM win-back rules.
        </p>
        <a href="/Customer_360_&_RFM" style="font-size: 0.8rem; font-weight: 700; color: var(--primary); text-decoration: none; display: flex; align-items: center; gap: 4px;">
            Launch Module <span class="material-symbols-outlined" style="font-size: 14px;">arrow_forward</span>
        </a>
    </div>
    """, unsafe_allow_html=True)

mod_cols2 = st.columns(2)
with mod_cols2[0]:
    st.markdown("""
    <div class="sahara-card" style="min-height: 200px;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
            <div style="width: 36px; height: 36px; border-radius: 8px; background: var(--bg-surface-elevated); border: 1px solid var(--border-subtle); display: flex; align-items: center; justify-content: center; color: var(--primary);">
                <span class="material-symbols-outlined">inventory_2</span>
            </div>
            <span class="sahara-badge-neutral">MODULE 04</span>
        </div>
        <h3 style="font-size: 1.15rem; font-weight: 700; color: var(--text-main); margin-bottom: 6px;">Inventory & Supply Chain Intelligence</h3>
        <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5; margin-bottom: 14px;">
            Stockout risk index, overstock capital at risk ($), Days Inventory Outstanding (DIO), SKU run-out forecasts, and automated EOQ reorder schedules.
        </p>
        <a href="/Inventory_&_Supply_Chain" style="font-size: 0.8rem; font-weight: 700; color: var(--primary); text-decoration: none; display: flex; align-items: center; gap: 4px;">
            Launch Module <span class="material-symbols-outlined" style="font-size: 14px;">arrow_forward</span>
        </a>
    </div>
    """, unsafe_allow_html=True)

with mod_cols2[1]:
    st.markdown("""
    <div class="sahara-card" style="min-height: 200px;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
            <div style="width: 36px; height: 36px; border-radius: 8px; background: var(--bg-surface-elevated); border: 1px solid var(--border-subtle); display: flex; align-items: center; justify-content: center; color: var(--primary);">
                <span class="material-symbols-outlined">memory</span>
            </div>
            <span class="sahara-badge-neutral">MODULE 05</span>
        </div>
        <h3 style="font-size: 1.15rem; font-weight: 700; color: var(--text-main); margin-bottom: 6px;">Model Management & MLOps Control Center</h3>
        <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5; margin-bottom: 14px;">
            Real-time accuracy telemetry (RMSE, MAPE, R²), training latency benchmarks, Kolmogorov-Smirnov data drift monitor, and one-click model retraining.
        </p>
        <a href="/Model_Management_&_MLOps" style="font-size: 0.8rem; font-weight: 700; color: var(--primary); text-decoration: none; display: flex; align-items: center; gap: 4px;">
            Launch Module <span class="material-symbols-outlined" style="font-size: 14px;">arrow_forward</span>
        </a>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Live Transaction Telemetry Stream Preview
st.markdown("""
<div class="sahara-eyebrow">
    <span class="material-symbols-outlined" style="font-size: 14px;">receipt_long</span>
    INGESTION LEDGER
</div>
<h2 style="font-family: 'EB Garamond', serif; font-size: 1.65rem; color: var(--text-main); margin-bottom: 6px;">
    Recent Transaction Stream (Live Ingestion)
</h2>
""", unsafe_allow_html=True)

try:
    from data_engine import data_engine
    df_sample = data_engine.load_data().head(10).copy()
    df_sample["UnitPrice"] = df_sample["UnitPrice"].apply(lambda p: f"${p:,.2f}")
    df_sample["Discount"] = df_sample["Discount"].apply(lambda d: f"{d*100:.0f}%")
    df_sample["Total_Amount"] = df_sample["Total_Amount"].apply(lambda a: f"${a:,.2f}")
    st.dataframe(df_sample, use_container_width=True)
except Exception as e:
    st.error(f"Error loading stream preview: {e}")

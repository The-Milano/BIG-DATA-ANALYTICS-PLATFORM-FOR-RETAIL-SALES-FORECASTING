"""
NEXUS RETAIL AI — Customer 360 & Behavioral RFM
Unsupervised Customer Segmentation, Latent Vector Mapping & Lifecycle Retention
Design System: Sahara & Sahara Nocturne
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.dirname(CURRENT_DIR)
if FRONTEND_DIR not in sys.path:
    sys.path.insert(0, FRONTEND_DIR)

from api_client import (
    apply_custom_css,
    render_sahara_top_header,
    render_sahara_sidebar,
    render_kpi_card_sahara,
    fetch_customer_segmentation,
    get_sahara_plotly_theme,
)

st.set_page_config(
    page_title="Customer 360 & RFM | NEXUS RETAIL AI",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()
render_sahara_sidebar(active_module="customer")

# Top persistent header matching Stitch
render_sahara_top_header(
    page_title="Customer 360 & Behavioral RFM",
    subtitle="High-dimensional customer segmentation, latent vector mapping, and retention optimization.",
    crumb="Customer 360",
    eyebrow="Lifecycle Intelligence / Unsupervised Behavioral Segmentation",
)

# 1. Sidebar Controls
with st.sidebar:
    st.markdown("### 🎛️ Segmentation Hyperparameters")
    k_clusters = st.slider(
        "Number of AI Clusters (k)",
        min_value=3,
        max_value=6,
        value=4,
        step=1,
        help="Configures K-Means cluster granularity on standardized log-transformed RFM features.",
    )

    st.markdown("#### 📐 Mathematical Framework")
    st.caption("""
    - **Recency (R):** Days since customer's last purchase.
    - **Frequency (F):** Total count of unique checkout invoices.
    - **Monetary (M):** Cumulative lifetime gross spend ($).
    - **Standardization:** Log-transform + Z-score scaling.
    """)

# 2. Fetch Customer Segmentation Data
with st.spinner(f"Fitting K-Means model (k={k_clusters}) on customer behavioral vectors..."):
    seg_data = fetch_customer_segmentation(k=k_clusters)

segments = seg_data.get("segments", [])
scatter_points = seg_data.get("scatter_3d", [])
total_cust = seg_data.get("total_customers", 12500)
total_rev = seg_data.get("total_revenue", 35659491.42)

# 3. Top Row Metrics: Cohort Telemetry Summary
top_segment = segments[0] if segments else {}
at_risk_segment = next((s for s in segments if "At-Risk" in s.get("segment_name", "")), segments[-1] if segments else {})

kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.markdown(
        render_kpi_card_sahara(
            title="Profiles Monitored",
            value=f"{total_cust:,}",
            badge_text="+14.2% YoY",
            badge_type="positive",
            target_text="Goal: 200k Accounts",
            velocity_label="Active Ingestion Window",
            subtext="68.4% active 30D window",
        ),
        unsafe_allow_html=True,
    )

with kpi_cols[1]:
    avg_clv = total_rev / max(total_cust, 1)
    st.markdown(
        render_kpi_card_sahara(
            title="Average Customer LTV",
            value=f"${avg_clv:,.2f}",
            badge_text="+8.6% LTV",
            badge_type="positive",
            target_text="Target: $820.00",
            velocity_label="Cumulative Account Yield",
            subtext="Multi-Item AOV Lift",
        ),
        unsafe_allow_html=True,
    )

with kpi_cols[2]:
    st.markdown(
        render_kpi_card_sahara(
            title="Repeat Order Frequency",
            value="3.42 ord/yr",
            badge_text="48-Day Cycle",
            badge_type="neutral",
            target_text="Benchmark: 3.20/yr",
            velocity_label="Inter-Purchase Velocity",
            subtext="Cohort velocity steady",
        ),
        unsafe_allow_html=True,
    )

with kpi_cols[3]:
    st.markdown(
        render_kpi_card_sahara(
            title="Net Churn Vulnerability",
            value="4.8% Low Risk",
            badge_text="-1.4% Churn",
            badge_type="positive",
            target_text="Ceiling: < 6.0%",
            velocity_label="Retention Triggers Active",
            subtext="1,420 Nudges Automated",
        ),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# 4. Main Visual 1: 3D Behavioral Latent Space & Cluster Inspector
space_col, inspect_col = st.columns([2.1, 1.2])

with space_col:
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
        <span style="font-family: 'EB Garamond', serif; font-size: 1.35rem; font-weight: 700; color: var(--text-main);">
            Behavioral RFM Latent Space (Recency × Frequency × Monetary)
        </span>
        <span style="font-family: 'JetBrains Mono'; font-size: 0.72rem; color: var(--primary);">
            ● Sub-Sampled 2,500 Vectors
        </span>
    </div>
    <div style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 12px;">
        Interactive 3D scatter projection of customer vectors. Dot size maps dynamically to lifetime frequency.
    </div>
    """, unsafe_allow_html=True)

    if scatter_points:
        df_scatter = pd.DataFrame(scatter_points)
        sahara_scatter_palette = ["#FFB955", "#C3841E", "#E07A38", "#8D6E63", "#FF7043", "#4E342E"]

        fig_3d = px.scatter_3d(
            df_scatter,
            x="recency",
            y="frequency",
            z="monetary",
            color="segment_name",
            color_discrete_sequence=sahara_scatter_palette,
            hover_name="customer_id",
            hover_data={
                "recency": ":,d",
                "frequency": ":,d",
                "monetary": ":$,.2f",
                "churn_risk": ":.1f%",
            },
            labels={
                "recency": "Recency (Days Inactive)",
                "frequency": "Frequency (Orders)",
                "monetary": "Monetary ($ Lifetime)",
                "segment_name": "Cluster Archetype",
            },
            opacity=0.85,
        )

        fig_3d.update_traces(marker=dict(size=4.5, line=dict(width=0.5, color="#1D1B19")))

        layout_3d = get_sahara_plotly_theme()
        layout_3d.update({
            "height": 520,
            "scene": {
                "xaxis": {
                    "title": "Recency (Days)",
                    "gridcolor": "rgba(224, 122, 56, 0.15)",
                    "backgroundcolor": "rgba(21, 19, 17, 0.6)",
                    "color": "#A89F91",
                },
                "yaxis": {
                    "title": "Frequency (Orders)",
                    "gridcolor": "rgba(224, 122, 56, 0.15)",
                    "backgroundcolor": "rgba(21, 19, 17, 0.6)",
                    "color": "#A89F91",
                },
                "zaxis": {
                    "title": "Monetary Spend ($)",
                    "gridcolor": "rgba(224, 122, 56, 0.15)",
                    "backgroundcolor": "rgba(21, 19, 17, 0.6)",
                    "color": "#A89F91",
                },
                "camera": {"eye": {"x": 1.4, "y": 1.4, "z": 0.9}},
            },
            "legend": {
                "title": {"text": "AI Archetypes"},
                "orientation": "h",
                "yanchor": "bottom",
                "y": -0.15,
                "xanchor": "center",
                "x": 0.5,
            }
        })
        fig_3d.update_layout(layout_3d)
        st.plotly_chart(fig_3d, use_container_width=True)

with inspect_col:
    st.markdown("""
    <div style="font-family: 'EB Garamond', serif; font-size: 1.35rem; font-weight: 700; color: var(--text-main); margin-bottom: 6px;">
        Cluster Inspector: Champions
    </div>
    """, unsafe_allow_html=True)

    c_name = top_segment.get("segment_name", "Champions")
    c_count = top_segment.get("customer_count", 18420)
    c_share = top_segment.get("revenue_share_pct", 42.8)
    c_aov = top_segment.get("mean_aov", 2840.0)
    c_rec = top_segment.get("mean_recency_days", 4.2)

    st.markdown(f"""
    <div class="sahara-card" style="padding: 18px 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <span style="font-size: 1.1rem; font-weight: 700; color: var(--primary);">{c_name}</span>
            <span class="sahara-badge-positive">{c_share}% GMV SHARE</span>
        </div>
        <p style="font-size: 0.84rem; color: var(--text-muted); line-height: 1.5; margin-bottom: 14px;">
            Elite high-velocity shoppers driving outsized commercial value. Zero coupon dependency and high cross-category attachment.
        </p>
        <div style="display: flex; flex-direction: column; gap: 8px; font-family: 'JetBrains Mono'; font-size: 12px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between;">
                <span style="color: var(--text-dim);">Cohort Population:</span>
                <span style="color: var(--text-main); font-weight: 600;">{c_count:,} accounts</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: var(--text-dim);">Mean Basket (AOV):</span>
                <span style="color: var(--text-main); font-weight: 600;">${c_aov:,.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: var(--text-dim);">Average Recency:</span>
                <span style="color: var(--emerald); font-weight: 600;">{c_rec} days</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: var(--text-dim);">Churn Probability:</span>
                <span style="color: var(--emerald); font-weight: 600;">3.2% (Extremely Low)</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Deploy Targeted Offer to Cohort", use_container_width=True):
        st.toast("VIP Loyalty Code & Early Access Campaign dispatched to Champion Cohort!", icon="🏆")

st.markdown("<br>", unsafe_allow_html=True)

# 5. Main Visual 2: Monthly Acquisition & Retention Heatmap (M+0 through M+7)
st.markdown("""
<div style="font-family: 'EB Garamond', serif; font-size: 1.35rem; font-weight: 700; color: var(--text-main); margin-bottom: 2px;">
    Monthly Acquisition & Retention Heatmap (Eliminating the Month 3 Cliff)
</div>
<div style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 12px;">
    Triangular cohort decay grid tracking percentage of repeat transactions across successive months.
</div>
""", unsafe_allow_html=True)

cohort_months = ["Jan 2024", "Feb 2024", "Mar 2024", "Apr 2024", "May 2024"]
intervals = ["M+0", "M+1", "M+2", "M+3", "M+4", "M+5", "M+6", "M+7"]

# Simulated realistic retention decay
retention_matrix = [
    [100, 48.2, 38.6, 34.2, 31.8, 29.5, 28.1, 27.4],
    [100, 49.5, 39.8, 35.6, 33.1, 31.0, 29.8, None],
    [100, 51.2, 42.1, 38.4, 35.8, 33.2, None, None],
    [100, 53.4, 44.8, 41.2, 38.6, None, None, None],
    [100, 55.8, 47.1, 43.5, None, None, None, None],
]

fig_cohort = go.Figure(data=go.Heatmap(
    z=retention_matrix,
    x=intervals,
    y=cohort_months,
    colorscale=[
        [0.0, "#151311"],
        [0.3, "#25221F"],
        [0.6, "#C2652A"],
        [1.0, "#FFB955"],
    ],
    text=[[f"{v:.1f}%" if v is not None else "" for v in row] for row in retention_matrix],
    texttemplate="%{text}",
    textfont=dict(family="JetBrains Mono", size=11, color="#FFFFFF"),
    hovertemplate="<b>Cohort: %{y}</b><br>Interval: %{x}<br>Retention: %{z:.1f}%<extra></extra>",
    colorbar=dict(title=dict(text="Retention (%)", font=dict(color="#EDE6DD", family="JetBrains Mono"))),
))

layout_cohort = get_sahara_plotly_theme()
layout_cohort.update({
    "height": 280,
    "xaxis": dict(layout_cohort["xaxis"], title="Cohort Lifespan (Interval Months)"),
    "yaxis": dict(layout_cohort["yaxis"], title="", autorange="reversed"),
})
fig_cohort.update_layout(layout_cohort)
st.plotly_chart(fig_cohort, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# 6. Tactical Marketing Rules Playbook Table
st.markdown("""
<div style="font-family: 'EB Garamond', serif; font-size: 1.35rem; font-weight: 700; color: var(--text-main); margin-bottom: 8px;">
    Tactical Marketing Directives & Automated CRM Rules
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sahara-table-wrapper">
    <table class="sahara-table">
        <thead>
            <tr>
                <th>Profile / ID</th>
                <th>Segment Archetype</th>
                <th style="text-align: right;">Total Spend</th>
                <th style="text-align: right;">Orders</th>
                <th style="text-align: right;">Last Active</th>
                <th style="text-align: right;">Action Protocol</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>
                    <div style="font-weight: 600; color: var(--text-main);">Clara Oswald</div>
                    <div style="font-size: 10px; color: var(--text-dim); font-family: 'JetBrains Mono';">CUST-883921 · Chicago</div>
                </td>
                <td><span class="sahara-archetype-tag" style="background: rgba(224, 122, 56, 0.15); color: var(--primary);">Champions</span></td>
                <td style="text-align: right; font-family: 'JetBrains Mono'; color: var(--primary); font-weight: 700;">$4,892.40</td>
                <td style="text-align: right; font-family: 'JetBrains Mono';">18 tx</td>
                <td style="text-align: right; font-family: 'JetBrains Mono';">2 days ago</td>
                <td style="text-align: right;"><button class="sahara-action-pill"><span class="material-symbols-outlined" style="font-size: 13px;">card_giftcard</span> VIP Concierge</button></td>
            </tr>
            <tr>
                <td>
                    <div style="font-weight: 600; color: var(--text-main);">Marcus Sterling</div>
                    <div style="font-size: 10px; color: var(--text-dim); font-family: 'JetBrains Mono';">CUST-441029 · New York</div>
                </td>
                <td><span class="sahara-archetype-tag" style="background: rgba(255, 185, 85, 0.15); color: var(--secondary);">Loyal Patrons</span></td>
                <td style="text-align: right; font-family: 'JetBrains Mono'; color: var(--secondary); font-weight: 700;">$2,410.80</td>
                <td style="text-align: right; font-family: 'JetBrains Mono';">11 tx</td>
                <td style="text-align: right; font-family: 'JetBrains Mono';">8 days ago</td>
                <td style="text-align: right;"><button class="sahara-action-pill"><span class="material-symbols-outlined" style="font-size: 13px;">auto_awesome</span> Early Drop Access</button></td>
            </tr>
            <tr>
                <td>
                    <div style="font-weight: 600; color: var(--text-main);">Elena Rostova</div>
                    <div style="font-size: 10px; color: var(--text-dim); font-family: 'JetBrains Mono';">CUST-310842 · San Francisco</div>
                </td>
                <td><span class="sahara-archetype-tag" style="background: rgba(239, 68, 68, 0.15); color: var(--coral);">At-Risk High LTV</span></td>
                <td style="text-align: right; font-family: 'JetBrains Mono'; color: var(--text-muted); font-weight: 700;">$1,840.00</td>
                <td style="text-align: right; font-family: 'JetBrains Mono';">7 tx</td>
                <td style="text-align: right; font-family: 'JetBrains Mono'; color: var(--coral);">44 days ago</td>
                <td style="text-align: right;"><button class="sahara-action-pill" style="background: #9B4600;"><span class="material-symbols-outlined" style="font-size: 13px;">outgoing_mail</span> Trigger Re-Engage</button></td>
            </tr>
            <tr>
                <td>
                    <div style="font-weight: 600; color: var(--text-main);">Julian Vance</div>
                    <div style="font-size: 10px; color: var(--text-dim); font-family: 'JetBrains Mono';">CUST-901422 · London</div>
                </td>
                <td><span class="sahara-archetype-tag" style="background: rgba(16, 185, 129, 0.15); color: var(--emerald);">New High Potential</span></td>
                <td style="text-align: right; font-family: 'JetBrains Mono'; color: var(--text-muted); font-weight: 700;">$980.50</td>
                <td style="text-align: right; font-family: 'JetBrains Mono';">3 tx</td>
                <td style="text-align: right; font-family: 'JetBrains Mono';">1 day ago</td>
                <td style="text-align: right;"><button class="sahara-action-pill"><span class="material-symbols-outlined" style="font-size: 13px;">recommend</span> Category Expansion</button></td>
            </tr>
        </tbody>
    </table>
</div>
""", unsafe_allow_html=True)

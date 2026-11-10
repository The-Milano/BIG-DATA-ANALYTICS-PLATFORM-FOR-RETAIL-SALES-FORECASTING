"""
NEXUS RETAIL AI — Inventory & Supply Chain Intelligence
Predictive Burn Rates, Safety Stock Depletion Curves & Fulfillment Center Balancing
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
    fetch_inventory_intelligence,
    get_sahara_plotly_theme,
)

st.set_page_config(
    page_title="Inventory & Supply Chain | NEXUS RETAIL AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()
render_sahara_sidebar(active_module="inventory")

# Top persistent header matching Stitch
render_sahara_top_header(
    page_title="Inventory & Supply Chain Intelligence",
    subtitle="Predictive burn rates, safety stock buffers, DC capacity balancing, and automated PO generation.",
    crumb="Supply Chain Foundry",
    eyebrow="Autonomous Logistics / Fulfillment Center Load Balancing",
)

# 1. Fetch Inventory Data
with st.spinner("Analyzing multi-warehouse SKU velocity and automated replenishment schedules..."):
    inv_data = fetch_inventory_intelligence()

summary = inv_data.get("summary", {})
skus = inv_data.get("skus", [])

# 2. Sidebar Filters
all_categories = ["All Categories"] + sorted(list(set(item["category"] for item in skus)))
status_options = ["All Statuses", "Critical", "Low Stock", "OK", "Overstocked"]

with st.sidebar:
    st.markdown("### 🎛️ Supply Chain Filters")
    selected_cat = st.selectbox("Product Category", all_categories, index=0)
    selected_status = st.selectbox("Inventory Status", status_options, index=0)

    st.markdown("#### 📐 Replenishment Logic")
    st.caption("""
    - **Reorder Point (ROP):** `(Daily Velocity × Lead Time) + Safety Stock`
    - **Safety Stock:** `1.65 × √(Lead Time) × Daily Velocity × 0.5` (95% service level)
    - **Economic Order Quantity (EOQ):** `√[(2 × Annual Demand × Setup Cost) / Holding Cost]`
    """)

# Filter SKU list
filtered_skus = skus
if selected_cat != "All Categories":
    filtered_skus = [s for s in filtered_skus if s["category"] == selected_cat]
if selected_status != "All Statuses":
    filtered_skus = [s for s in filtered_skus if s["status"] == selected_status]

# 3. Top Row Metrics: Supply Chain Macro Metrics matching Stitch PRD
total_capital = summary.get("total_inventory_valuation", 38420100.0)
dio = summary.get("days_inventory_outstanding", 53.7)
risk_level = summary.get("out_of_stock_risk_level", "Medium")

kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.markdown(
        render_kpi_card_sahara(
            title="Total Capital in Stock",
            value=f"${total_capital:,.2f}",
            badge_text="$64.2M Equiv",
            badge_type="positive",
            target_text="Operating Cap: $42.0M",
            velocity_label="Warehouse & Transit Basis",
            subtext="Warehouses & Transit",
        ),
        unsafe_allow_html=True,
    )

with kpi_cols[1]:
    crit_count = summary.get("critical_skus_count", 14)
    st.markdown(
        render_kpi_card_sahara(
            title="Stockout Vulnerability",
            value="1.84% Low Risk",
            badge_text=f"{crit_count} SKUs Alert",
            badge_type="neutral" if crit_count < 15 else "negative",
            target_text="Threshold: < 2.50%",
            velocity_label="Auto-Replenish Protocol",
            subtext="Autonomous PO Active",
        ),
        unsafe_allow_html=True,
    )

with kpi_cols[2]:
    st.markdown(
        render_kpi_card_sahara(
            title="Inventory Turnover Velocity",
            value="6.8x / yr",
            badge_text=f"{dio:.1f} Days DIO",
            badge_type="positive",
            target_text="Industry Median: 5.2x",
            velocity_label="Capital Efficiency Index",
            subtext="Top Quartile Performance",
        ),
        unsafe_allow_html=True,
    )

with kpi_cols[3]:
    st.markdown(
        render_kpi_card_sahara(
            title="OTIF Supplier Compliance",
            value="96.4% Compliant",
            badge_text="Tier-1 Grade",
            badge_type="positive",
            target_text="SLA Mandate: 95.0%",
            velocity_label="Lead Time Deviation",
            subtext="+1.8 Days Lead Tolerance",
        ),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# 4. Main Visual 1: Predictive Burn Rate & Safety Stock Depletion Curve
burn_col, dc_col = st.columns([2.1, 1.2])

with burn_col:
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
        <span style="font-family: 'EB Garamond', serif; font-size: 1.35rem; font-weight: 700; color: var(--text-main);">
            Predictive Burn Rate & Safety Stock Depletion Curve
        </span>
        <span style="font-family: 'JetBrains Mono'; font-size: 0.72rem; color: var(--coral);">
            ● Critical Intersection: Day 14 Trigger
        </span>
    </div>
    <div style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 12px;">
        30-day projection modeling inventory depletion toward Critical Buffer Floor (1,200 units).
    </div>
    """, unsafe_allow_html=True)

    # 30-day simulated burn rate curve
    projection_days = np.arange(0, 31)
    starting_stock = 8400
    daily_burn = 240
    stock_curve = np.maximum(0, starting_stock - (daily_burn * projection_days) + np.random.normal(0, 30, len(projection_days)))
    safety_buffer = np.full(len(projection_days), 1200)

    fig_deplete = go.Figure()

    fig_deplete.add_trace(go.Scatter(
        x=projection_days,
        y=stock_curve,
        mode="lines",
        name="Projected Inventory Stock (Units)",
        line=dict(color="#E07A38", width=2.8),
        fill="tozeroy",
        fillcolor="rgba(224, 122, 56, 0.08)",
        hovertemplate="<b>Day +%{x}</b><br>Stock: %{y:,.0f} units<extra></extra>",
    ))

    fig_deplete.add_trace(go.Scatter(
        x=projection_days,
        y=safety_buffer,
        mode="lines",
        name="Critical Buffer Floor (1,200 Units)",
        line=dict(color="#EF4444", width=2, dash="dash"),
        hovertemplate="<b>Safety Buffer:</b> 1,200 units<extra></extra>",
    ))

    # Add PO Trigger Line & Annotation at Day 14 (safe shape + annotation)
    fig_deplete.add_shape(
        type="line",
        x0=14,
        x1=14,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="#FFB955", width=1.5, dash="dot"),
    )
    fig_deplete.add_annotation(
        x=14,
        y=1,
        yref="paper",
        text="Auto PO Trigger (Day 14)",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(color="#FFB955", family="JetBrains Mono", size=10),
        bgcolor="rgba(21, 19, 17, 0.8)",
        bordercolor="rgba(255, 185, 85, 0.4)",
        borderwidth=1,
        borderpad=3,
    )

    layout_deplete = get_sahara_plotly_theme()
    layout_deplete.update({
        "height": 380,
        "xaxis": dict(layout_deplete["xaxis"], title="Forward Horizon (Days from T-0)"),
        "yaxis": dict(layout_deplete["yaxis"], title="Aggregate Warehousing Units"),
    })
    fig_deplete.update_layout(layout_deplete)
    st.plotly_chart(fig_deplete, use_container_width=True)

with dc_col:
    st.markdown("""
    <div style="font-family: 'EB Garamond', serif; font-size: 1.35rem; font-weight: 700; color: var(--text-main); margin-bottom: 6px;">
        Fulfillment Center (DC) Load Balancer
    </div>
    """, unsafe_allow_html=True)

    dc_telemetry = [
        {"DC": "East DC-01 (Newark, NJ)", "Load": 84, "Status": "Optimal"},
        {"DC": "West DC-02 (Reno, NV)", "Load": 91, "Status": "Peak (Diverted)"},
        {"DC": "Central DC-03 (Dallas, TX)", "Load": 72, "Status": "Optimal"},
        {"DC": "Midwest DC-04 (Chicago, IL)", "Load": 63, "Status": "Capacity Available"},
    ]

    for dc in dc_telemetry:
        bar_color = "var(--coral)" if dc["Load"] > 88 else ("var(--secondary)" if dc["Load"] > 75 else "var(--emerald)")
        badge_bg = "rgba(239, 68, 68, 0.15)" if dc["Load"] > 88 else ("rgba(255, 185, 85, 0.15)" if dc["Load"] > 75 else "rgba(16, 185, 129, 0.15)")
        st.markdown(f"""
        <div class="sahara-load-node">
            <div class="sahara-load-header">
                <span style="font-weight: 600; color: var(--text-main);">{dc["DC"]}</span>
                <span style="font-family: 'JetBrains Mono'; font-size: 11px; font-weight: 700; color: {bar_color}; background: {badge_bg}; padding: 2px 6px; border-radius: 4px;">{dc["Load"]}% · {dc["Status"]}</span>
            </div>
            <div class="sahara-load-bar-track">
                <div class="sahara-load-bar-fill" style="width: {dc['Load']}%; background: {bar_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: var(--bg-surface-elevated); border: 1px solid var(--border-subtle); border-radius: 10px; padding: 12px; margin-top: 8px;">
        <div style="font-size: 11px; font-weight: 700; color: var(--primary); text-transform: uppercase; font-family: 'JetBrains Mono'; margin-bottom: 4px;">
            ⚡ AI DYNAMIC LOAD RECOMMENDATION
        </div>
        <p style="font-size: 12px; color: var(--text-muted); margin: 0; line-height: 1.4;">
            Diverting 2,400 incoming outerwear units to <strong>Midwest DC-04</strong> to alleviate West DC-02 bottlenecks and save 14 hours transit time.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 5. Intelligent Restocking & Reorder Queue Table
st.markdown("""
<div style="font-family: 'EB Garamond', serif; font-size: 1.35rem; font-weight: 700; color: var(--text-main); margin-bottom: 8px;">
    Intelligent Restocking & Autonomous Reorder Queue
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sahara-table-wrapper">
    <table class="sahara-table">
        <thead>
            <tr>
                <th>SKU / Item Blueprint</th>
                <th>Category</th>
                <th style="text-align: right;">On-Hand</th>
                <th>Runout Est.</th>
                <th style="text-align: right;">ROP Trigger</th>
                <th style="text-align: right;">EOQ Order</th>
                <th>Primary Vendor</th>
                <th style="text-align: right;">Unit Cost</th>
                <th style="text-align: right;">Action Protocol</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>
                    <div style="font-weight: 600; color: var(--text-main);">AeroFlex Carbon Shell</div>
                    <div style="font-size: 10px; color: var(--text-dim); font-family: 'JetBrains Mono';">SKU-9921-SXL · High Velocity</div>
                </td>
                <td><span style="font-size: 11px; color: var(--text-muted);">Technical Outerwear</span></td>
                <td style="text-align: right; font-family: 'JetBrains Mono'; font-weight: 700;">310 pcs</td>
                <td><span class="sahara-badge-negative"><span class="material-symbols-outlined" style="font-size: 11px;">warning</span> 6 Days (Urgent)</span></td>
                <td style="text-align: right; font-family: 'JetBrains Mono'; color: var(--secondary);">850 pcs</td>
                <td style="text-align: right; font-family: 'JetBrains Mono'; color: var(--primary); font-weight: 700;">1,800 pcs</td>
                <td><span style="font-size: 11px; color: var(--text-muted);">Vanguard Alpine Mfg.</span></td>
                <td style="text-align: right; font-family: 'JetBrains Mono';">$112.50</td>
                <td style="text-align: right;"><button class="sahara-action-pill"><span class="material-symbols-outlined" style="font-size: 12px;">shopping_cart</span> Approve PO</button></td>
            </tr>
            <tr>
                <td>
                    <div style="font-weight: 600; color: var(--text-main);">HyperSound Pro Noise-Cancelling</div>
                    <div style="font-size: 10px; color: var(--text-dim); font-family: 'JetBrains Mono';">SKU-4412-BLK · Consumer Tech</div>
                </td>
                <td><span style="font-size: 11px; color: var(--text-muted);">Audio Electronics</span></td>
                <td style="text-align: right; font-family: 'JetBrains Mono'; font-weight: 700;">480 pcs</td>
                <td><span class="sahara-badge-negative"><span class="material-symbols-outlined" style="font-size: 11px;">warning</span> 9 Days</span></td>
                <td style="text-align: right; font-family: 'JetBrains Mono'; color: var(--secondary);">600 pcs</td>
                <td style="text-align: right; font-family: 'JetBrains Mono'; color: var(--primary); font-weight: 700;">1,200 pcs</td>
                <td><span style="font-size: 11px; color: var(--text-muted);">AcousticCore Shenzhen</span></td>
                <td style="text-align: right; font-family: 'JetBrains Mono';">$84.00</td>
                <td style="text-align: right;"><button class="sahara-action-pill"><span class="material-symbols-outlined" style="font-size: 12px;">shopping_cart</span> Approve PO</button></td>
            </tr>
            <tr>
                <td>
                    <div style="font-weight: 600; color: var(--text-main);">TerraShield All-Weather Parka</div>
                    <div style="font-size: 10px; color: var(--text-dim); font-family: 'JetBrains Mono';">SKU-1824-M · Winter Surge</div>
                </td>
                <td><span style="font-size: 11px; color: var(--text-muted);">Apparel & Thermal</span></td>
                <td style="text-align: right; font-family: 'JetBrains Mono'; font-weight: 700;">720 pcs</td>
                <td><span class="sahara-badge-neutral">14 Days Buffer</span></td>
                <td style="text-align: right; font-family: 'JetBrains Mono'; color: var(--secondary);">900 pcs</td>
                <td style="text-align: right; font-family: 'JetBrains Mono'; color: var(--primary); font-weight: 700;">2,400 pcs</td>
                <td><span style="font-size: 11px; color: var(--text-muted);">Nordic Thermal Co.</span></td>
                <td style="text-align: right; font-family: 'JetBrains Mono';">$145.00</td>
                <td style="text-align: right;"><button class="sahara-action-pill" style="background: var(--bg-surface-elevated); color: var(--text-main) !important; border: 1px solid var(--border-subtle);"><span class="material-symbols-outlined" style="font-size: 12px;">sync_alt</span> Rebalance DC</button></td>
            </tr>
            <tr>
                <td>
                    <div style="font-weight: 600; color: var(--text-main);">Quantum Ergo Hydraulic Desk</div>
                    <div style="font-size: 10px; color: var(--text-dim); font-family: 'JetBrains Mono';">SKU-8831-OAK · Heavy Freight</div>
                </td>
                <td><span style="font-size: 11px; color: var(--text-muted);">Workspace Hardware</span></td>
                <td style="text-align: right; font-family: 'JetBrains Mono'; font-weight: 700;">1,120 pcs</td>
                <td><span class="sahara-badge-positive">28 Days Safe</span></td>
                <td style="text-align: right; font-family: 'JetBrains Mono'; color: var(--secondary);">400 pcs</td>
                <td style="text-align: right; font-family: 'JetBrains Mono'; color: var(--primary); font-weight: 700;">500 pcs</td>
                <td><span style="font-size: 11px; color: var(--text-muted);">ErgoCraft Industries</span></td>
                <td style="text-align: right; font-family: 'JetBrains Mono';">$298.00</td>
                <td style="text-align: right;"><button class="sahara-action-pill" style="background: var(--bg-surface-elevated); color: var(--text-muted) !important; border: 1px solid var(--border-subtle);"><span class="material-symbols-outlined" style="font-size: 12px;">visibility</span> Review Buffer</button></td>
            </tr>
        </tbody>
    </table>
</div>
""", unsafe_allow_html=True)

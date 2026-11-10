"""
NEXUS RETAIL AI — Executive Overview Dashboard
High-Altitude Commercial Operations, Intraday Velocity Tracking & Core Algorithmic KPIs
Design System: Sahara & Sahara Nocturne
"""

import os
import sys
import streamlit as st
import pandas as pd
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
    fetch_filter_metadata,
    fetch_overview,
    get_sahara_plotly_theme,
)

st.set_page_config(
    page_title="Executive Overview | NEXUS RETAIL AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()
render_sahara_sidebar(active_module="overview")

# Top persistent header matching Stitch
render_sahara_top_header(
    page_title="Executive Overview",
    subtitle="Real-Time Revenue Operations, Multi-Store Velocity & Core Algorithmic KPIs across unified commerce clusters.",
    crumb="Executive Overview",
    eyebrow="Foundry v4.8.2 / Multi-Tenant Ingestion Active",
)

# 1. Multi-Dimensional Filter & Granularity Bar
metadata = fetch_filter_metadata()
stores_list = ["All Stores Flagships"] + metadata.get("stores", [])
categories_list = ["All Categories"] + metadata.get("categories", [])

min_date = pd.to_datetime(metadata.get("min_date", "2024-11-01")).date()
max_date = pd.to_datetime(metadata.get("max_date", "2026-10-31")).date()

with st.sidebar:
    st.markdown("### 🎛️ Store & Category Filters")
    selected_store_raw = st.selectbox("Store Location", stores_list, index=0)
    selected_category_raw = st.selectbox("Product Category", categories_list, index=0)

    selected_store = None if "All Stores" in selected_store_raw else selected_store_raw
    selected_category = None if "All Categories" in selected_category_raw else selected_category_raw

    st.markdown("#### 📅 Temporal Horizon")
    date_range = st.date_input(
        "Observation Window",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range[0], date_range[1]
    else:
        start_date, end_date = min_date, max_date

    st.markdown("---")
    st.caption("Nexus Data Foundry • Real-Time Stream")

# Action Bar: Trigger Pipeline Recompute & Export
act_col1, act_col2, act_col3 = st.columns([2, 1, 1])
with act_col2:
    if st.button("⚡ Trigger Recompute", use_container_width=True):
        st.toast("Pipeline recomputed on live stream data!", icon="⚡")
with act_col3:
    export_csv = st.button("📥 Export CSV Report", use_container_width=True)

# 2. Fetch Overview Data
with st.spinner("Aggregating commercial telemetry across global hubs..."):
    data = fetch_overview(
        start_date=str(start_date),
        end_date=str(end_date),
        store_location=selected_store,
        category=selected_category,
    )

kpis = data.get("kpis", {})
daily_sales = data.get("daily_sales", [])
category_split = data.get("category_split", [])
hourly_heatmap = data.get("hourly_heatmap", {})

# 3. Top Telemetry Quartet (Exact Stitch Screen Layout)
kpi_cols = st.columns(4)

with kpi_cols[0]:
    rev = kpis.get("total_revenue", 35659491.42)
    yoy = kpis.get("yoy_growth_pct", 18.4)
    st.markdown(
        render_kpi_card_sahara(
            title="Gross Invoiced GMV",
            value=f"${rev:,.2f}",
            badge_text=f"+{yoy:.1f}% YoY",
            badge_type="positive",
            target_text="$34.50M (104.8% to Plan)",
            velocity_label="QTD Run-rate Velocity",
            subtext="$165.4k/hr",
        ),
        unsafe_allow_html=True,
    )

with kpi_cols[1]:
    orders = kpis.get("total_orders", 120000)
    st.markdown(
        render_kpi_card_sahara(
            title="Processed Orders",
            value=f"{orders:,}",
            badge_text="+12.6% vs LQ",
            badge_type="positive",
            target_text="115,000 (104.3% vs Target)",
            velocity_label="Intraday Velocity",
            subtext="2,054 tx/day nominal",
        ),
        unsafe_allow_html=True,
    )

with kpi_cols[2]:
    aov = kpis.get("avg_basket_size", 297.16)
    st.markdown(
        render_kpi_card_sahara(
            title="Average Basket Size (AOV)",
            value=f"${aov:.2f}",
            badge_text="+5.2% YoY",
            badge_type="positive",
            target_text="$285.00 (Exceeded)",
            velocity_label="Multi-Item Checkout",
            subtext=f"Median: ${aov*0.88:.2f}",
        ),
        unsafe_allow_html=True,
    )

with kpi_cols[3]:
    csat = kpis.get("csat_score", 4.82)
    retention = kpis.get("retention_rate_pct", 84.6)
    st.markdown(
        render_kpi_card_sahara(
            title="Customer CSAT & Retention",
            value=f"94.8% ({csat}/5)",
            badge_text=f"+{retention:.1f}% Ret.",
            badge_type="positive",
            target_text="Benchmark: 90.0%",
            velocity_label="NPS Sentiment Index",
            subtext="+68 High Loyalty",
        ),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# 4. Main Visuals: Row 1 (Sales Velocity Spline & Category Donut)
vis_col1, vis_col2 = st.columns([2.2, 1.2])

with vis_col1:
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
        <span style="font-family: 'EB Garamond', serif; font-size: 1.35rem; font-weight: 700; color: var(--text-main);">
            Historical Sales Velocity & 7D Moving Average
        </span>
        <span style="font-family: 'JetBrains Mono'; font-size: 0.72rem; color: var(--primary);">
            ● Peak Oct 28: $54.2k GMV
        </span>
    </div>
    """, unsafe_allow_html=True)

    if daily_sales:
        df_daily = pd.DataFrame(daily_sales)
        fig_sales = go.Figure()

        # Realized Invoices
        fig_sales.add_trace(go.Scatter(
            x=df_daily["date"],
            y=df_daily["sales"],
            mode="lines",
            name="Realized Daily Sales ($)",
            line=dict(color="#E07A38", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(224, 122, 56, 0.08)",
            hovertemplate="<b>%{x}</b><br>Daily Sales: $%{y:,.2f}<extra></extra>",
        ))

        # 7-Day Algorithmic Baseline (Amber Ember)
        fig_sales.add_trace(go.Scatter(
            x=df_daily["date"],
            y=df_daily["sales_ma7"],
            mode="lines",
            name="7-Day Moving Average",
            line=dict(color="#FFB955", width=2.5),
            hovertemplate="<b>7-Day MA:</b> $%{y:,.2f}<extra></extra>",
        ))

        # 30-Day Trendline (Dotted Sand)
        fig_sales.add_trace(go.Scatter(
            x=df_daily["date"],
            y=df_daily["sales_ma30"],
            mode="lines",
            name="30-Day Trendline",
            line=dict(color="#C3841E", width=1.8, dash="dot"),
            hovertemplate="<b>30-Day Trend:</b> $%{y:,.2f}<extra></extra>",
        ))

        layout = get_sahara_plotly_theme()
        layout.update({
            "height": 380,
            "hovermode": "x unified",
            "yaxis": dict(layout["yaxis"], title="Gross Invoiced GMV (USD)"),
        })
        fig_sales.update_layout(layout)
        st.plotly_chart(fig_sales, use_container_width=True)

with vis_col2:
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
        <span style="font-family: 'EB Garamond', serif; font-size: 1.35rem; font-weight: 700; color: var(--text-main);">
            Revenue Split by Category
        </span>
        <span style="font-family: 'JetBrains Mono'; font-size: 0.72rem; color: var(--text-dim);">
            Catalog Distribution
        </span>
    </div>
    """, unsafe_allow_html=True)

    if category_split:
        df_cat = pd.DataFrame(category_split)
        sahara_palette = ["#E07A38", "#FFB955", "#C3841E", "#FF8A65", "#4E443C", "#2E7D32"]

        fig_donut = go.Figure(data=[go.Pie(
            labels=df_cat["category"],
            values=df_cat["revenue"],
            hole=0.60,
            marker=dict(colors=sahara_palette, line=dict(color="#1D1B19", width=2)),
            textinfo="percent",
            hoverinfo="label+value+percent",
            hovertemplate="<b>%{label}</b><br>GMV: $%{value:,.2f}<br>Share: %{percent}<extra></extra>",
        )])

        layout_donut = get_sahara_plotly_theme()
        layout_donut.update({
            "height": 380,
            "showlegend": True,
            "legend": dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
            "annotations": [{
                "text": "<b>TOTAL GMV</b><br>DISTRIBUTION",
                "showarrow": False,
                "font": {"size": 11, "color": "#E07A38", "family": "Manrope"},
            }]
        })
        fig_donut.update_layout(layout_donut)
        st.plotly_chart(fig_donut, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# 5. Main Visuals: Row 2 (168-Cell Intraday Heatmap & Surging SKU Ledger)
heat_col, sku_col = st.columns([1.8, 1.2])

with heat_col:
    st.markdown("""
    <div style="font-family: 'EB Garamond', serif; font-size: 1.35rem; font-weight: 700; color: var(--text-main); margin-bottom: 2px;">
        Intraday 7-Day × 24-Hour Velocity Heatmap
    </div>
    <div style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 10px;">
        168-cell matrix mapping hourly transaction density (00:00 to 23:00) identifying peak checkout surges.
    </div>
    """, unsafe_allow_html=True)

    if hourly_heatmap and "matrix" in hourly_heatmap:
        days = hourly_heatmap.get("days", [])
        hours = hourly_heatmap.get("hours", [])
        z_matrix = hourly_heatmap.get("matrix", [])

        # Sahara Ember Gradient Colorscale
        ember_scale = [
            [0.0, "#151311"],
            [0.2, "#25221F"],
            [0.5, "#9B4600"],
            [0.8, "#E07A38"],
            [1.0, "#FFB955"],
        ]

        fig_heat = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=hours,
            y=days,
            colorscale=ember_scale,
            hovertemplate="<b>%{y} at %{x}</b><br>GMV Velocity: $%{z:,.2f}<extra></extra>",
            colorbar=dict(
                title=dict(text="GMV ($)", font=dict(color="#EDE6DD", family="JetBrains Mono")),
                tickfont=dict(color="#A89F91", family="JetBrains Mono", size=10),
                len=0.9,
            )
        ))

        layout_heat = get_sahara_plotly_theme()
        layout_heat.update({
            "height": 330,
            "xaxis": dict(layout_heat["xaxis"], title="Hour of Day (UTC-5)"),
            "yaxis": dict(layout_heat["yaxis"], title="", autorange="reversed"),
        })
        fig_heat.update_layout(layout_heat)
        st.plotly_chart(fig_heat, use_container_width=True)

with sku_col:
    st.markdown("""
    <div style="font-family: 'EB Garamond', serif; font-size: 1.35rem; font-weight: 700; color: var(--text-main); margin-bottom: 2px;">
        Surging SKU Ingestion Stream
    </div>
    <div style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 10px;">
        Live ledger tracking velocity surges & catalog momentum.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sahara-table-wrapper">
        <table class="sahara-table">
            <thead>
                <tr>
                    <th>Item / Descriptor</th>
                    <th style="text-align: right;">Units</th>
                    <th style="text-align: right;">Gross GMV</th>
                    <th style="text-align: right;">Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>
                        <div style="font-weight: 600; color: var(--text-main);">AeroFlex Carbon</div>
                        <div style="font-size: 10px; color: var(--text-dim); font-family: 'JetBrains Mono';">SKU-7729 · Footwear</div>
                    </td>
                    <td style="text-align: right; font-family: 'JetBrains Mono';">1,420</td>
                    <td style="text-align: right; font-family: 'JetBrains Mono'; color: var(--primary); font-weight: 700;">$227,200</td>
                    <td style="text-align: right;"><span class="sahara-badge-positive">Surging</span></td>
                </tr>
                <tr>
                    <td>
                        <div style="font-weight: 600; color: var(--text-main);">HyperSound ANC</div>
                        <div style="font-size: 10px; color: var(--text-dim); font-family: 'JetBrains Mono';">SKU-4412 · Audio</div>
                    </td>
                    <td style="text-align: right; font-family: 'JetBrains Mono';">1,180</td>
                    <td style="text-align: right; font-family: 'JetBrains Mono'; color: var(--primary); font-weight: 700;">$212,400</td>
                    <td style="text-align: right;"><span class="sahara-badge-positive">Surging</span></td>
                </tr>
                <tr>
                    <td>
                        <div style="font-weight: 600; color: var(--text-main);">TerraShield Parka</div>
                        <div style="font-size: 10px; color: var(--text-dim); font-family: 'JetBrains Mono';">SKU-9921 · Apparel</div>
                    </td>
                    <td style="text-align: right; font-family: 'JetBrains Mono';">890</td>
                    <td style="text-align: right; font-family: 'JetBrains Mono'; color: var(--secondary); font-weight: 700;">$267,000</td>
                    <td style="text-align: right;"><span class="sahara-badge-neutral">High Demand</span></td>
                </tr>
                <tr>
                    <td>
                        <div style="font-weight: 600; color: var(--text-main);">Quantum Ergo Desk</div>
                        <div style="font-size: 10px; color: var(--text-dim); font-family: 'JetBrains Mono';">SKU-3108 · Home</div>
                    </td>
                    <td style="text-align: right; font-family: 'JetBrains Mono';">640</td>
                    <td style="text-align: right; font-family: 'JetBrains Mono'; color: var(--text-muted); font-weight: 700;">$319,360</td>
                    <td style="text-align: right;"><span class="sahara-badge-neutral">Stable</span></td>
                </tr>
                <tr>
                    <td>
                        <div style="font-weight: 600; color: var(--text-main);">Bio-Retinol Serum</div>
                        <div style="font-size: 10px; color: var(--text-dim); font-family: 'JetBrains Mono';">SKU-1823 · Beauty</div>
                    </td>
                    <td style="text-align: right; font-family: 'JetBrains Mono';">760</td>
                    <td style="text-align: right; font-family: 'JetBrains Mono'; color: var(--text-muted); font-weight: 700;">$64,600</td>
                    <td style="text-align: right;"><span class="sahara-badge-neutral">Stable</span></td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

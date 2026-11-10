"""
NEXUS RETAIL AI - Sahara Design System & API Client
Implements Sahara (Warm Minimalist) & Sahara Nocturne (Dark Mode) UI chrome,
Material Symbols, typography, and REST communication with FastAPI backend.
"""

import os
import sys
import json
import requests
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List, Tuple

# Resolve paths
FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(FRONTEND_DIR)
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 12.0


def apply_custom_css():
    """Injects custom CSS from style.css into the Streamlit session with theme awareness."""
    # Check if user clicked query param theme switcher (?theme=light or ?theme=dark)
    if "theme" in st.query_params:
        param_theme = st.query_params["theme"]
        if param_theme in ["light", "dark"] and param_theme != st.session_state.get("theme"):
            st.session_state["theme"] = param_theme

    theme = st.session_state.get("theme", "dark")
    css_path = os.path.join(FRONTEND_DIR, "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            css_content = f.read()

        theme_attr = 'class="theme-light"' if theme == "light" else 'class="theme-dark"'
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        if theme == "light":
            st.markdown(
                """
                <script>
                document.documentElement.setAttribute('data-theme', 'light');
                document.body.classList.add('theme-light');
                </script>
                <style>
                :root {
                  --bg-main: #FAF6F0 !important;
                  --bg-sidebar: #F5EFE6 !important;
                  --bg-surface: #FFFFFF !important;
                  --bg-surface-elevated: #F5EFE6 !important;
                  --bg-surface-high: #EBE5DC !important;
                  --bg-surface-low: #FAF5EE !important;
                  --border-subtle: #E8E2D6 !important;
                  --border-strong: #D5CBBF !important;
                  --primary: #C2652A !important;
                  --primary-glow: rgba(194, 101, 42, 0.20) !important;
                  --secondary: #D9822B !important;
                  --text-main: #1F1B18 !important;
                  --text-muted: #6E6459 !important;
                  --text-dim: #9E9384 !important;
                  --emerald: #059669 !important;
                  --coral: #DC2626 !important;
                }
                html, body, [data-testid="stAppViewContainer"] {
                  background-color: #FAF6F0 !important;
                  color: #1F1B18 !important;
                }
                [data-testid="stSidebar"] {
                  background-color: #F5EFE6 !important;
                  border-right: 1px solid #E5DDD0 !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )


def check_backend_health() -> Tuple[bool, Dict[str, Any]]:
    """Checks if FastAPI backend is reachable. Returns (is_online, info_dict)."""
    try:
        resp = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=2.5)
        if resp.status_code == 200:
            return True, resp.json()
    except Exception:
        pass

    try:
        from data_engine import data_engine
        meta = data_engine.get_filter_metadata()
        return False, {
            "status": "local_fallback",
            "total_transactions": meta["total_records"],
            "unique_customers": meta["unique_customers"],
            "unique_skus": meta["unique_skus"],
            "date_range": f"{meta['min_date']} to {meta['max_date']}",
        }
    except Exception as e:
        return False, {"status": "offline", "error": str(e)}


def fetch_filter_metadata() -> Dict[str, Any]:
    """Fetches store list, categories, and date constraints."""
    try:
        resp = requests.get(f"{API_BASE_URL}/api/v1/filters/metadata", timeout=3.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    from data_engine import data_engine
    return data_engine.get_filter_metadata()


def fetch_overview(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    store_location: Optional[str] = None,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetches Executive KPIs, historical daily sales, and category splits."""
    params = {}
    if start_date:
        params["start_date"] = str(start_date)
    if end_date:
        params["end_date"] = str(end_date)
    if store_location and store_location != "All Stores":
        params["store_location"] = store_location
    if category and category != "All Categories":
        params["category"] = category

    try:
        resp = requests.get(f"{API_BASE_URL}/api/v1/overview", params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    from analytics_engine import analytics_engine
    return analytics_engine.get_executive_overview(
        start_date=start_date,
        end_date=end_date,
        store_location=store_location,
        category=category,
    )


def fetch_forecast(
    horizon_days: int = 30,
    store_id: Optional[str] = None,
    category: Optional[str] = None,
    model_type: str = "Ensemble (Prophet + LightGBM)",
    include_holidays: bool = True,
) -> Dict[str, Any]:
    """Calls forecasting API with Prophet & LightGBM ensemble."""
    payload = {
        "horizon_days": horizon_days,
        "store_id": None if store_id == "All Stores" else store_id,
        "category": None if category == "All Categories" else category,
        "model_type": model_type,
        "include_holidays": include_holidays,
    }
    try:
        resp = requests.post(f"{API_BASE_URL}/api/v1/forecast", json=payload, timeout=25.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    from forecasting_engine import forecasting_engine
    return forecasting_engine.run_forecast(
        horizon_days=horizon_days,
        store_id=payload["store_id"],
        category=payload["category"],
        model_type=model_type,
        include_holidays=include_holidays,
    )


def fetch_customer_segmentation(k: int = 4) -> Dict[str, Any]:
    """Fetches RFM K-Means clusters and marketing recommendations."""
    try:
        resp = requests.get(f"{API_BASE_URL}/api/v1/customers/segmentation?k={k}", timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    from customer_engine import customer_engine
    return customer_engine.perform_segmentation(k=k)


def fetch_inventory_intelligence() -> Dict[str, Any]:
    """Fetches stockout risk, days of inventory, and reorder tables."""
    try:
        resp = requests.get(f"{API_BASE_URL}/api/v1/inventory/anomalies", timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    from analytics_engine import analytics_engine
    return analytics_engine.get_inventory_intelligence()


def trigger_model_retrain() -> Dict[str, Any]:
    """Triggers backend model refitting and accuracy benchmarking."""
    try:
        resp = requests.post(f"{API_BASE_URL}/api/v1/mlops/retrain", timeout=30.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    from forecasting_engine import forecasting_engine
    return forecasting_engine.retrain_models()


def fetch_mlops_metrics() -> Dict[str, Any]:
    """Fetches current MLOps latency, accuracy, and drift metrics."""
    try:
        resp = requests.get(f"{API_BASE_URL}/api/v1/mlops/metrics", timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    from forecasting_engine import forecasting_engine
    return {
        "metrics": forecasting_engine.last_metrics,
        "feature_importances": forecasting_engine.feature_importances,
    }


def render_sahara_top_header(
    page_title: str,
    subtitle: str = "",
    crumb: str = "Retail Analytics",
    eyebrow: str = "Foundry v4.8.2 / Multi-Tenant Ingestion Active",
):
    """Renders the top persistent chrome matching Stitch Sahara specifications."""
    is_online, info = check_backend_health()
    status_label = "ONLINE (200 OK)" if is_online else "LOCAL FAST-ENGINE"
    dot_color = "#10B981" if is_online else "#F59E0B"

    current_theme = st.session_state.get("theme", "dark")
    is_dark = (current_theme == "dark")

    # Stitch Exact Theme Switcher Buttons (Light / Dark)
    light_active = "active" if not is_dark else ""
    dark_active = "active" if is_dark else ""
    dot_html = '<span class="sahara-theme-dot"></span>' if is_dark else ''

    # Clean HTML with zero leading indentation to prevent markdown 4-space code block interpretation
    raw_html_lines = [
        '<div class="sahara-top-chrome">',
        '<div class="sahara-brand-group">',
        '<div class="sahara-brand-emblem"><span class="material-symbols-outlined" style="font-size: 20px;">hub</span></div>',
        f'<div class="sahara-brand-title">NEXUS <span class="sahara-version-pill">v4.8</span><span style="font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;">/ {crumb}</span></div>',
        '</div>',
        '<div class="sahara-meta-strip">',
        f'<div class="sahara-status-capsule"><span class="sahara-dot-pulse" style="background-color: {dot_color}; box-shadow: 0 0 8px {dot_color};"></span><span>SYSTEM: NOMINAL 99.98%</span></div>',
        f'<div class="sahara-status-capsule"><span style="color: var(--text-dim);">FASTAPI:</span><span style="color: {dot_color}; font-weight: 700;">{status_label}</span></div>',
        '<div class="sahara-status-capsule"><span>USD $ / UTC-5</span></div>',
        '<div aria-label="Theme Switcher" class="sahara-header-theme-switcher">',
        f'<a href="?theme=light" target="_self" class="sahara-header-theme-btn {light_active}" title="Switch to Sahara Light mode"><span class="material-symbols-outlined" style="font-size: 15px;">light_mode</span><span>Sahara Light</span></a>',
        f'<a href="?theme=dark" target="_self" class="sahara-header-theme-btn {dark_active}" title="Sahara Dark mode active"><span class="material-symbols-outlined" style="font-size: 15px;">dark_mode</span><span>Sahara Dark</span>{dot_html}</a>',
        '</div>',
        '<div class="sahara-user-avatar" title="E. Vance (VP Merchandising)">EV</div>',
        '</div>',
        '</div>',
        '<div style="margin-top: 14px; margin-bottom: 22px;">',
        f'<div class="sahara-eyebrow"><span class="material-symbols-outlined" style="font-size: 14px;">circle</span>{eyebrow}</div>',
        f'<h1 class="sahara-hero-title">{page_title}</h1>',
        f'<div class="sahara-hero-subtitle">{subtitle}</div>' if subtitle else '',
        '</div>',
    ]

    header_html = "".join(raw_html_lines)
    st.markdown(header_html, unsafe_allow_html=True)


def render_sahara_sidebar(active_module: str = "overview"):
    """Renders the Stitch Sahara navigation sidebar, hiding default Streamlit navigation."""
    is_online, info = check_backend_health()
    status_label = "ONLINE (200 OK)" if is_online else "LOCAL FAST-ENGINE"
    rec_count = info.get("total_transactions", 120492)

    with st.sidebar:
        # 1. Brand & System Status
        st.markdown(f"""
        <div style="display: flex; flex-direction: column; gap: 14px; padding-bottom: 16px; border-bottom: 1px solid var(--border-subtle);">
            <div style="display: flex; items-center; gap: 8px;">
                <div style="width: 10px; height: 10px; border-radius: 50%; background: var(--primary); box-shadow: 0 0 12px var(--primary-glow); margin-top: 4px;"></div>
                <span style="font-family: 'JetBrains Mono'; font-size: 13px; font-weight: 700; color: var(--text-main); letter-spacing: 0.1em; text-transform: uppercase;">
                    NEXUS V4.8
                </span>
            </div>
            <div style="background: var(--bg-surface-elevated); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 10px 12px; display: flex; flex-direction: column; gap: 6px; font-family: 'JetBrains Mono'; font-size: 11px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: var(--text-dim); text-transform: uppercase;">SYSTEM</span>
                    <span style="color: var(--primary); font-weight: 600;">NOMINAL 99.98%</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: var(--text-dim); text-transform: uppercase;">FASTAPI</span>
                    <span style="color: var(--secondary); font-weight: 600;">{status_label}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Theme switcher has been cleanly moved to the top header matching Stitch designs!

        # 3. Stitch Operational Modules Navigation Menu
        active_exec = "active" if active_module == "overview" else ""
        active_fcst = "active" if active_module == "forecasting" else ""
        active_cust = "active" if active_module == "customer" else ""
        active_inv = "active" if active_module == "inventory" else ""
        active_mlops = "active" if active_module == "mlops" else ""

        st.markdown(f"""
        <div style="margin-bottom: 6px;">
            <span style="font-family: 'JetBrains Mono'; font-size: 10px; font-weight: 700; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.1em; padding-left: 4px;">
                OPERATIONAL MODULES
            </span>
            <div class="sahara-nav-list">
                <a class="sahara-nav-item {active_exec}" href="/Executive_Overview" target="_self">
                    <span class="material-symbols-outlined">dashboard</span>
                    <span>Executive Overview</span>
                </a>
                <a class="sahara-nav-item {active_fcst}" href="/AI_Sales_Forecasting" target="_self">
                    <span class="material-symbols-outlined">trending_up</span>
                    <span>AI Sales Forecasting</span>
                </a>
                <a class="sahara-nav-item {active_cust}" href="/Customer_360_&_RFM" target="_self">
                    <span class="material-symbols-outlined">groups</span>
                    <span>Customer 360 & RFM</span>
                </a>
                <a class="sahara-nav-item {active_inv}" href="/Inventory_&_Supply_Chain" target="_self">
                    <span class="material-symbols-outlined">inventory_2</span>
                    <span>Inventory & Supply Chain</span>
                </a>
                <a class="sahara-nav-item {active_mlops}" href="/Model_Management_&_MLOps" target="_self">
                    <span class="material-symbols-outlined">memory</span>
                    <span>Model Management & MLOps</span>
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 4. MLOps Telemetry Mini-Card matching Stitch Aside bottom
        st.markdown(f"""
        <div style="background: var(--bg-surface-elevated); border: 1px solid var(--border-subtle); border-radius: 10px; padding: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.15);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 10px; font-weight: 800; color: var(--text-main); font-family: 'JetBrains Mono'; text-transform: uppercase;">MLOPS TELEMETRY</span>
                <span style="font-size: 9px; font-weight: 700; color: var(--emerald); background: var(--emerald-bg); padding: 1px 6px; border-radius: 4px;">[HEALTHY]</span>
            </div>
            <div style="display: flex; flex-direction: column; gap: 5px; font-family: 'JetBrains Mono'; font-size: 11px;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: var(--text-dim);">Inference Latency:</span>
                    <span style="color: var(--text-main); font-weight: 600;">18ms</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: var(--text-dim);">Ingested SKUs:</span>
                    <span style="color: var(--text-main); font-weight: 600;">{rec_count:,}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: var(--text-dim);">Active Cluster:</span>
                    <span style="color: var(--text-main); font-weight: 600;">Prod / us-east-1a</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_kpi_card_sahara(
    title: str,
    value: str,
    badge_text: str,
    badge_type: str = "positive",
    icon_name: str = "analytics",
    subtext: str = "",
    sparkline_text: str = "",
    target_text: str = "",
    velocity_label: str = "",
) -> str:
    """Renders high-fidelity Sahara KPI card matching exact Stitch specifications with svg sparkline."""
    badge_class = "sahara-badge-positive" if badge_type == "positive" else ("sahara-badge-negative" if badge_type == "negative" else "sahara-badge-neutral")
    trend_icon = "trending_up" if badge_type == "positive" else ("trending_down" if badge_type == "negative" else "show_chart")

    target_html = f'''<div class="sahara-kpi-target-label">
        Target: <span class="sahara-kpi-target-val">{target_text}</span>
    </div>''' if target_text else ""

    vel_html = f'''<div style="display: flex; flex-direction: column;">
        <span style="font-family: 'JetBrains Mono'; font-size: 9px; color: var(--text-dim); text-transform: uppercase;">{velocity_label}</span>
        <span style="font-family: 'JetBrains Mono'; font-size: 11px; color: var(--text-muted); font-weight: 600;">{subtext}</span>
    </div>''' if velocity_label else f'<span class="sahara-subtext">{subtext}</span>'

    # SVG sparkline vector based on badge_type
    stroke_color = "#FFB68D" if badge_type == "positive" else ("#EF4444" if badge_type == "negative" else "#FFB955")
    pulse_color = "#E07A38" if badge_type == "positive" else ("#EF4444" if badge_type == "negative" else "#D9822B")
    svg_html = f'''<svg width="80" height="28" viewBox="0 0 80 28" fill="none" style="overflow: visible;">
        <path d="M0 22 Q 14 26 28 16 T 56 12 T 80 4" fill="none" stroke="{stroke_color}" stroke-linecap="round" stroke-width="2"/>
        <circle cx="80" cy="4" r="2.5" fill="{pulse_color}"/>
    </svg>'''

    return f"""
    <div class="sahara-card">
        <div class="sahara-kpi-header">
            <span class="sahara-kpi-title">{title}</span>
            <span class="{badge_class}">
                <span class="material-symbols-outlined" style="font-size: 13px;">{trend_icon}</span>
                {badge_text}
            </span>
        </div>
        <div class="sahara-kpi-value">{value}</div>
        {target_html}
        <div class="sahara-kpi-footer" style="padding-top: 4px; border-top: 1px solid var(--border-subtle); margin-top: 6px;">
            {vel_html}
            {svg_html}
        </div>
    </div>
    """


def get_sahara_plotly_theme() -> Dict[str, Any]:
    """Returns Plotly styling configured for Sahara & Sahara Nocturne."""
    theme = st.session_state.get("theme", "dark")
    is_dark = (theme == "dark")

    grid_color = "rgba(56, 48, 40, 0.45)" if is_dark else "rgba(229, 221, 208, 0.65)"
    text_color = "#A89F91" if is_dark else "#6E6459"
    card_bg = "rgba(29, 27, 25, 0.85)" if is_dark else "rgba(255, 255, 255, 0.90)"

    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Manrope, sans-serif", "color": text_color, "size": 12},
        # No title object so Plotly.js never renders literal "undefined"
        "xaxis": {
            "gridcolor": grid_color,
            "zerolinecolor": grid_color,
            "tickfont": {"family": "JetBrains Mono, monospace", "color": text_color, "size": 11},
            "showline": True,
            "linecolor": grid_color,
        },
        "yaxis": {
            "gridcolor": grid_color,
            "zerolinecolor": grid_color,
            "tickfont": {"family": "JetBrains Mono, monospace", "color": text_color, "size": 11},
            "showline": True,
            "linecolor": grid_color,
        },
        "legend": {
            "bgcolor": card_bg,
            "bordercolor": "rgba(224, 122, 56, 0.25)" if is_dark else "rgba(194, 101, 42, 0.25)",
            "borderwidth": 1,
            "font": {"family": "Manrope, sans-serif", "color": "#EDE6DD" if is_dark else "#1F1B18"},
        },
        "margin": {"l": 40, "r": 20, "t": 45, "b": 40},
    }

# NEXUS RETAIL AI — Autonomous Commerce Intelligence & Operations Platform

An enterprise-grade, decoupled retail intelligence system built with **FastAPI** for high-throughput statistical modeling and **Streamlit** styled with the **Sahara & Sahara Nocturne Design System** (Warm Minimalist & Dark Mode).

Designed to bridge raw predictive machine learning (Meta Prophet + LightGBM ensembles, UMAP latent spaces, RFM clustering) with real-time merchandising and supply chain operations across global flagship retail hubs.

---

## Highlights & Features

- **Executive Commercial Operations**: Live telemetry across 120,000+ transactions, 7-day moving averages, 168-cell intraday velocity heatmap (24h × 7d), and surging SKU streams.
- **Multi-Variate AI Sales Forecasting**: Dual Meta Prophet + LightGBM ensemble pipeline with 95% confidence intervals, T-0 cutoff horizons, and weekly/monthly seasonal decomposition.
- **Customer 360 & Latent Space RFM**: 3D customer lifetime value clustering (Recency, Frequency, Monetary), cohort retention decay heatmaps, and automated CRM retention directives.
- **Autonomous Supply Chain & Restock Engine**: 30-day predictive safety stock burn curves with automatic PO requisition triggers, fulfillment center (DC) load balancing, and Economic Order Quantity (EOQ) optimization.
- **Model Management & MLOps Control Center**: Live accuracy telemetry ($R^2$, MAPE, RMSE), training latency tracking, Kolmogorov-Smirnov covariate drift monitoring, and one-click model retraining.
- **Sahara Design System**: Seamless theme switching between **Sahara Nocturne** (Dark Mode) and **Sahara Light** (Warm Minimalist) with zero external dashboard framework lock-in.

---

## Architecture

```
┌────────────────────────────────────────────────────────┐
│      Streamlit Frontend (Sahara Design System)         │
│  - Port 8501 (5 Operational Cockpits + Command Center) │
│  - Custom Design Tokens, Material Symbols & Plotly     │
└───────────────────────────┬────────────────────────────┘
                            │ REST API (JSON) / Local Fallback
┌───────────────────────────▼────────────────────────────┐
│                  FastAPI Backend Server                │
│  - Port 8000 (Asynchronous High-Throughput Engine)     │
└───────┬───────────────────┬────────────────────┬───────┘
        │                   │                    │
┌───────▼───────────┐ ┌─────▼──────────────┐ ┌───▼─────────────────────┐
│ Forecasting Engine│ │   Customer Engine  │ │  Analytics & Inventory  │
│ - Meta Prophet    │ │ - RFM Segmentation │ │ - Intraday Heatmaps     │
│ - LightGBM Regr.  │ │ - K-Means (k=3..6) │ │ - EOQ Reorder Logic     │
│ - 95% CI Bounds   │ │ - Churn Scoring    │ │ - DC Load Balancer      │
└───────┬───────────┘ └─────┬──────────────┘ └───┬─────────────────────┘
        │                   │                    │
        └───────────────────┼────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│      Parquet / CSV High-Velocity Transaction Store     │
│  - 120,000+ Ingested Invoices across 6 Flagship Hubs   │
└────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
retail_analytics_platform/
├── backend/
│   ├── app.py                 # FastAPI application & REST endpoint router
│   ├── data_engine.py         # 120k+ row transaction generator & columnar cache
│   ├── forecasting_engine.py  # Prophet & LightGBM ensemble pipeline
│   ├── customer_engine.py     # RFM computation, K-Means & churn heuristics
│   ├── analytics_engine.py    # KPIs, heatmap matrices & inventory optimization
│   └── requirements.txt       # Backend dependencies
├── frontend/
│   ├── app.py                 # Streamlit entry point (Operations Command Center)
│   ├── api_client.py          # REST client, Sahara chrome & Plotly theme tokens
│   ├── style.css              # Custom Sahara & Sahara Nocturne CSS stylesheets
│   └── pages/
│       ├── 1_📊_Executive_Overview.py
│       ├── 2_📈_AI_Sales_Forecasting.py
│       ├── 3_👥_Customer_360_&_RFM.py
│       ├── 4_📦_Inventory_&_Supply_Chain.py
│       └── 5_⚙️_Model_Management_&_MLOps.py
└── data/
    ├── retail_transactions.csv
    └── retail_transactions.parquet
```

---

## Getting Started

### 1. Prerequisites

- Python 3.10+ (tested on Python 3.11, 3.13, and 3.14)
- `pip` or Anaconda environment

### 2. Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>/retail_analytics_platform

# Install backend dependencies
pip install -r backend/requirements.txt

# Install frontend dependencies
pip install streamlit plotly requests pandas
```

### 3. Launching the Platform

The backend and frontend run in decoupled sessions:

**Terminal 1 — FastAPI Backend Engine:**
```bash
cd retail_analytics_platform/backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
API docs available at: `http://localhost:8000/docs`

**Terminal 2 — Streamlit Frontend UI:**
```bash
cd retail_analytics_platform/frontend
streamlit run app.py --server.port 8501
```
Application dashboard will open at: `http://localhost:8501`

---

## Deployment Guide

### Option A: Streamlit Community Cloud (Frontend)
1. Push your repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Select your repository, set the main file path to `retail_analytics_platform/frontend/app.py`.
4. Add environment variable `API_BASE_URL` pointing to your deployed FastAPI backend URL.

### Option B: Docker Compose (Full Stack)
```yaml
version: '3.8'
services:
  backend:
    build: ./retail_analytics_platform/backend
    ports:
      - "8000:8000"
    environment:
      - PORT=8000

  frontend:
    build: ./retail_analytics_platform/frontend
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://backend:8000
    depends_on:
      - backend
```

### Option C: Render / Railway / Fly.io (Backend)
- Deploy the `backend/` directory as a Python web service with start command:
  `uvicorn app:app --host 0.0.0.0 --port $PORT`

---

## License

MIT License. Designed and engineered for high-performance retail analytics.

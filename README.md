# NEXUS RETAIL AI: Big Data Analytics & Forecasting Platform

An enterprise-grade, multi-page, decoupled retail intelligence and predictive forecasting platform built with **FastAPI** for the backend engine and **Streamlit** styled with a **Modern Dark Mode Glassmorphism Theme** for the executive UI.

---

## 🚀 Quick Run Instructions

### 1. Launch FastAPI Backend Server (Terminal 1)
```bash
cd retail_analytics_platform/backend
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
- API Docs: `http://localhost:8000/docs`

### 2. Launch Streamlit UI (Terminal 2)
```bash
cd retail_analytics_platform/frontend
streamlit run app.py --server.port 8501
```
- Web Application: `http://localhost:8501`

---

## 📁 Project Architecture

```
retail_analytics_platform/
├── backend/
│   ├── app.py                 # FastAPI server main entry point & routing
│   ├── data_engine.py         # 120,000+ row synthetic/real retail data generator & cache
│   ├── forecasting_engine.py  # Meta Prophet & LightGBM time-series ensemble pipelines
│   ├── customer_engine.py     # RFM Analysis, K-Means clustering & Churn prediction
│   ├── analytics_engine.py    # Cross-filtering, drill-downs, and EOQ inventory replenishment
│   └── requirements.txt       # Dependencies
├── frontend/
│   ├── app.py                 # Streamlit entry point (Navigation & Command Center)
│   ├── api_client.py          # REST client with automatic local fast-engine fallback
│   ├── style.css              # Custom CSS: Glassmorphism, glowing KPI cards, neon palette
│   └── pages/
│       ├── 1_📊_Executive_Overview.py          # Top-line KPIs, 7d MA sales, hourly heatmap
│       ├── 2_📈_AI_Sales_Forecasting.py       # Prophet + LightGBM, 95% CI, seasonality
│       ├── 3_👥_Customer_360_&_RFM.py          # 3D RFM scatter, K-Means, marketing rules
│       ├── 4_📦_Inventory_&_Supply_Chain.py   # Run-out forecasts, EOQ reorders, DIO
│       └── 5_⚙️_Model_Management_&_MLOps.py   # RMSE/MAPE/R², KS data drift, live retrain
└── data/
    ├── retail_transactions.csv      # Persistent CSV storage (120,000 rows)
    └── retail_transactions.parquet  # High-speed columnar cache
```

For complete documentation, see [retail_analytics_platform/README.md](retail_analytics_platform/README.md).

# PRODUCT REQUIREMENTS DOCUMENT (PRD) & SYSTEM BRIEF
## NEXUS RETAIL AI — Autonomous Commerce Intelligence & Operations Platform

**Document Version:** 1.0.0  
**Status:** Approved / Production Baseline  
**Product Lead:** Principal Data & Product Engineering  
**Design System:** Sahara (Warm Minimalist) & Sahara Nocturne (Dark Mode)  
**Target Form Factor:** Desktop (1440px+ optimized, 12-column responsive grid) & Mobile Companion

---

## 1. Executive Summary & Vision

### 1.1 Product Mission
**NEXUS RETAIL AI** is an enterprise-grade autonomous commerce intelligence and decision engine. It synthesizes real-time transactional telemetry, multi-variate demand forecasting, high-dimensional customer segmentation, and supply chain logistics into a single, cohesive operational cockpit.

The platform eliminates fragmented legacy dashboards (e.g., standard Streamlit scripts or disconnected BI tools) by bridging the gap between raw statistical machine learning models (LightGBM, Prophet, UMAP) and real-time executive decision-making.

### 1.2 Core Value Proposition
- **Unified Algorithmic Truth:** Single source of truth linking statistical telemetry (MAPE, R², Covariate Drift) directly to financial metrics (GMV, AOV, Gross Margin, Net Churn).
- **Proactive Autonomous Ops:** Replaces passive reporting with active recommendations—autonomous PO reorders, automated DC transit rerouting, and real-time model canary deployments.
- **Dual Aesthetic Craft:** Purpose-engineered in dual high-fidelity themes (**Sahara Light** and **Sahara Nocturne Dark**) delivering sun-baked warmth, high information density, and zero visual fatigue.

---

## 2. Target Personas & Stakeholders

| Persona | Role & Objectives | Primary Modules Used | Key Pain Points Solved |
| :--- | :--- | :--- | :--- |
| **E. Vance** *(VP Merchandising & Retail)* | High-level commercial health, revenue velocity, category performance, and target attainment. | Executive Overview, Customer 360 | Stale daily rollups; lacks intraday transaction density visibility. |
| **Staff Demand Planner** | Inventory burn rates, safety stock buffers, DC capacity, automated purchase requisition. | Inventory & Supply Chain Intelligence | Manual Excel lead-time calculations; frequent stockouts during promo surges. |
| **Lead Data Scientist / MLOps** | Model drift detection, hyperparameter validation, latency SLAs, pipeline lineage, rollbacks. | Model Management & MLOps, AI Sales Forecasting | Inability for non-technical stakeholders to see model health; siloed notebooks. |
| **Head of Growth & Retention** | Cohort decay tracking, RFM cluster migration, churn intervention triggers. | Customer 360 & Behavioral RFM | Static quarterly cohort analysis; slow CDP sync cycles. |

---

## 3. Product Architecture & Page Specifications

The platform is architected around 5 core operational modules sharing persistent navigation, top telemetry headers, and dual-theme switching.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        NEXUS RETAIL AI PLATFORM                        │
├─────────────────┬─────────────────┬──────────────────┬─────────────────┤
│  1. EXECUTIVE   │  2. AI SALES    │  3. CUSTOMER 360 │  4. INVENTORY   │
│    OVERVIEW     │   FORECASTING   │      & RFM       │  & SUPPLY CHAIN │
├─────────────────┴─────────────────┴──────────────────┴─────────────────┤
│                 5. MODEL MANAGEMENT & MLOPS CONTROL CENTER             │
└────────────────────────────────────────────────────────────────────────┘
```

---

### Page 1: Executive Overview (`/executive-overview`)
*High-altitude commercial operations and intraday velocity tracking.*

- **Top KPI Telemetry Quartet:**
  - **Gross Invoiced GMV:** `$14,892,450.00` (+18.4% YoY, 104.8% to plan, $165.4k/hr velocity sparkline).
  - **Processed Orders:** `184,920` (+12.6% vs LQ, 2,054 tx/day, 3.42% nominal conversion).
  - **Average Basket Size (AOV):** `$80.53` (+5.2% YoY, multi-item median $68.20).
  - **Customer CSAT & FastApi Sentiment:** `94.8%` (+2.1% MoM, NPS +68, retention 84.6%).
- **Historical Sales Velocity & 7D Moving Average:**
  - Dual-line spline visualization comparing realized invoices vs. 7-day algorithmic baseline with confidence interval shaded bounds.
  - Interactive peak marker with drill-down tooltips (e.g., Oct 28 Peak: $54.2k GMV).
- **Revenue Split by Category:**
  - Circular distribution donut (Apparel & Footwear 38%, Electronics 29%, Home & Living 18%, Health & Beauty 15%).
- **Intraday 7-Day × 24-Hour Velocity Heatmap:**
  - 168-cell matrix displaying hourly transaction intensity (00:00 to 22:00) highlighting weekend checkout surges.
- **SKU Velocity Ingestion Stream:**
  - Live ledger tracking surging items (`AeroFlex Carbon`, `HyperSound ANC`, `Quantum Ergo Desk`) with direct catalog links.

---

### Page 2: AI Sales Forecasting (`/sales-forecasting`)
*Multi-variate predictive modeling, exogenous factor attribution, and confidence intervals.*

- **Algorithmic Engine Controls:**
  - Multi-engine toggle: *Ensemble (Prophet + LightGBM)*, *Meta Prophet Solo*, *LightGBM Regressor*.
  - Forecast Horizon Selectors: 7D, 14D, **30D (Active)**, 60D, 90D.
  - Exogenous Covariate Checkers: Holiday Regressors (+18.4% Q4 Lift), Promo Calendar, Weather Shock Absorber.
- **Predictive Metric Cards:**
  - 30-Day Predicted GMV (`$16,420,800`, 96.4% confidence score).
  - Ensemble MAPE (`4.12%`, 10-fold cross-validation).
  - Order Velocity (`6,480 units/day`, Peak 9,210 units).
  - Volatility Index (`0.14 Low Beta`, Gaussian dispersion).
- **Temporal Trajectory & 95% Confidence Band:**
  - Forward-projected 30-day spline curve with upper/lower bounds.
  - T-0 Cutoff line separating historical telemetry from projected horizon.
  - Pinned milestone flags (e.g., Day +18 Black Friday: $582,400 GMV).
- **Decomposition Diagnostics:**
  - Day-of-week cyclicity bar charts (+28.5% Saturday lift).
  - Annual trendline velocity spline (+41.0% Q4 ramp).
  - Regressor feature weights (Cyber Week +46.2%, Black Friday +38.8%, Fall Preview +14.1%).
- **Granular Ingestion Ledger:**
  - High-density table with date, predicted units, GMV bounds, dominant factor driver, and export options (CSV/Parquet).

---

### Page 3: Customer 360 & Behavioral RFM (`/customer-360`)
*Unsupervised customer segmentation and lifecycle retention optimization.*

- **Cohort Telemetry Summary:**
  - Profiles Monitored: `284,520` (+14.2% YoY, 68.4% active 30D window).
  - Average Customer Lifetime Value (CLV/LTV): `$842.50` (Peak $1,140.00).
  - Order Frequency: `3.42 ord/yr` (48-day average inter-purchase cycle).
  - Net Churn Vulnerability Index: `4.8%` (Low Risk; 1,420 automated nudges active).
- **2D UMAP Recency-Monetary Latent Space:**
  - Multi-cluster interactive scatter plot projecting customer vectors into 2D space.
  - Distinct clusters: *Champions (Amber)*, *Loyalists (Sand)*, *At-Risk (Bronze)*, *Dormant (Charcoal)*.
  - Dot sizing dynamically mapped to lifetime purchase frequency.
- **Cluster Inspector (Champions & High Velocity):**
  - Cohort size: 18,420 customers (6.5% of total).
  - GMV Contribution: $52.3M (**42.8% of total revenue**).
  - Average Spend: $2,840.00 | Average Recency: 4.2 days.
  - One-click action: *Deploy Targeted Offer to Cohort*.
- **Monthly Acquisition & Retention Heatmap:**
  - Triangular retention decay grid from Jan 2024 to May 2024 across M+0 through M+7.
  - Highlights elimination of the Month 3 cliff (+12.4% retention lift).
- **Recent Transacting Profiles Feed:**
  - Live user stream with segment badge, recency, total order count, monetary spend, and next expected order window.

---

### Page 4: Inventory & Supply Chain Intelligence (`/supply-chain`)
*Autonomous replenishment, depletion curves, and fulfillment center load balancing.*

- **Supply Chain Macro Metrics:**
  - Total Capital in Stock: `$38,420,100` ($64.2M retail equivalent).
  - Stockout Vulnerability: `1.84%` (Threshold < 2.50%, 14 SKUs in danger zone).
  - Inventory Turnover Velocity: `6.8x / yr` (53.7 Days Sales of Inventory).
  - OTIF Supplier Rate: `96.4%` (Tier-1 compliant, +1.8 days lead time deviation).
- **Predictive Burn Rate & Safety Stock Depletion Curve:**
  - 30-day projection modeling inventory depletion toward Critical Buffer Floor (1,200 units).
  - Visual intersection at Day 14 triggering automated Purchase Orders.
  - SKU alert cards (`AeroFlex Carbon`: 6 days left; `TerraShield Parka`: 11 days left).
- **Fulfillment Center (DC) Load Balancer:**
  - Node telemetry: East DC-01 (84%), West DC-02 (91% Peak / Inbound Diverted), Central DC-03 (72%), Midwest DC-04 (63%).
  - AI Dynamic Load Recommendation: Diverting 2,400 incoming outerwear units to Midwest DC to save 14 hours transit time.
- **Intelligent Restocking & Reorder Queue:**
  - SKU blueprint table with Runout Est., Reorder Point (ROP), Economic Order Quantity (EOQ), and Primary Vendor.
  - Quick action buttons: `Approve PO`, `Rebalance DC`, `Review Buffer`.

---

### Page 5: Model Management & MLOps (`/mlops`)
*Production model lineage, drift diagnostics, and continuous training infrastructure.*

- **Autonomous Pipeline Topology Hero Card:**
  - Live execution diagram: Ingestion (33 SKUs / 120k records) ➔ LightGBM (80%) + Prophet (20%) Ensemble ➔ FastAPI Serving (18ms average latency).
- **Production Performance Telemetry:**
  - Model Accuracy (R²): `85.0%` (+3.0% vs target threshold).
  - Prediction Error (MAPE): `13.63%` (RMSE: $7,624.18 | MAE: $4,112.50).
  - Training & Fit Latency: `1,150ms` (Batch inference: 18ms/req, cold-start: 420ms).
  - Covariate Shift Monitor: `NORMAL (STABLE)` (Kolmogorov-Smirnov p-value: 0.6235 > 0.05).
- **LightGBM Feature Importance Rankings:**
  - Ranked bar chart of top 12 predictors (rolling 7-day mean: 12.8%, day-of-week: 11.4%, rolling 28-day: 10.2%, lags, promo elasticity).
- **Residual Error Distribution:**
  - Unbiased estimator curve centered symmetrically at $\mu = \$0.00$.
  - Skewness: `-0.04` | Kurtosis: `2.98` (mesokurtic normal) | Outliers (>3σ): `0.01%`.
- **Active Deployment Ledger & Artifact Registry:**
  - Version matrix: `v2.4.1-ensemble` (Production 100%), `v2.5.0-rc1` (Canary 10%), `v2.3.8-lightgbm` (Standby).
  - One-click operations: `Promote`, `Rollback`, `Telemetry Inspect`.

---

## 4. Design System & Ergonomics

### 4.1 Visual Direction
The interface leverages two distinct sub-themes within a shared design token hierarchy:

1. **Sahara Light (Warm Minimalism):**
   - Backgrounds: `#FAF8F5`, `#F3EFEA`, `#EBE5DC`.
   - Typography: Manrope / Plus Jakarta Sans for UI; Newsreader for editorial luxury accents.
   - Accents: Desert terracotta (`#C2652A`), warm ochre (`#D9822B`), deep charcoal ink (`#1E1A17`).
2. **Sahara Nocturne (Dark Mode):**
   - Backgrounds: Deep espresso slate (`#151311`, `#1D1B19`, `#25221F`).
   - Accents: Glowing terracotta (`#E07A38`), amber ember (`#E6A23C`), soft sand typography (`#EDE6DD`).
   - Micro-borders: Warm bronze/espresso dividers (`#383028`).

### 4.2 Global Shell Architecture
- **Persistent Header:** System status indicator (`NOMINAL 99.98%`), FastAPI health (`200 OK`), Omnichannel search bar (`⌘K`), currency telemetry, **Sahara Light ⇄ Sahara Dark theme toggle pill**, and `Deploy` primary CTA.
- **Left Navigation Drawer:** Brand emblem (`NEXUS v4.8`), operational modules list with active indicator pills, MLOps telemetry mini-card with live ingestion metrics.

---

## 5. Technical Requirements & Integrations

- **Frontend Tech Stack:** Semantic HTML5, Tailwind CSS token architecture, SVG charting engines with vector gradients, zero heavy runtime dependencies for instant rendering.
- **Backend APIs:**
  - `GET /api/v1/telemetry/executive`: Returns intraday GMV, velocity sparklines, category splits.
  - `POST /api/v1/forecast/predict`: Executes 30/60/90-day multi-variate inferences.
  - `GET /api/v1/customers/umap`: Fetches 2D coordinates for 1,000 sub-sampled customer points.
  - `POST /api/v1/inventory/reorder`: Triggers automated purchase orders to ERP.
  - `POST /api/v1/mlops/retrain`: Initiates AWS c6i.2xlarge automated pipeline retraining.

---

## 6. Implementation Roadmap & Milestones

- **Phase 1 (Completed):** Core 5-Page Desktop Suite in Sahara Light.
- **Phase 2 (Completed):** Sahara Nocturne Dark Mode generation across all 5 operational views.
- **Phase 3 (Completed):** Global dual-state Theme Switcher in persistent navigation chrome.
- **Phase 4 (Next):** Mobile responsive companion layouts (~390px portrait) for executive and field operations.
- **Phase 5 (Next):** Production REST API wiring & automated ERP purchase order connectors.

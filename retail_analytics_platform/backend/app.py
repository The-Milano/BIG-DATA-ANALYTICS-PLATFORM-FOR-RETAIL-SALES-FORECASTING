"""
NEXUS RETAIL AI - FastAPI Enterprise Backend Server
High-throughput REST API serving executive analytics, Prophet & LightGBM forecasts,
RFM customer clustering, automated inventory replenishment, and MLOps management.
"""

import os
import sys

# Ensure temp directory for matplotlib cache
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# Ensure current package can be resolved
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data_engine import data_engine
from analytics_engine import analytics_engine
from customer_engine import customer_engine
from forecasting_engine import forecasting_engine

# Initialize FastAPI App
app = FastAPI(
    title="NEXUS RETAIL AI Platform API",
    description="Enterprise Big Data Analytics, Time-Series Forecasting & Supply Chain API",
    version="2.4.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for Streamlit frontend and local clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class ForecastRequest(BaseModel):
    horizon_days: int = Field(default=30, ge=7, le=90, description="Forecast horizon in days (7 to 90)")
    store_id: Optional[str] = Field(default=None, description="Store location filter or None for all stores")
    category: Optional[str] = Field(default=None, description="Product category filter or None for all categories")
    model_type: Optional[str] = Field(
        default="Ensemble (Prophet + LightGBM)",
        description="Model architecture: 'Ensemble (Prophet + LightGBM)', 'Prophet Solo', or 'LightGBM Solo'"
    )
    include_holidays: Optional[bool] = Field(default=True, description="Whether to include retail holiday effects")


@app.on_event("startup")
async def startup_event():
    """Ensure dataset is primed and ready in memory upon server boot."""
    try:
        data_engine.load_data()
    except Exception as e:
        print(f"Startup data loading note: {e}")


@app.get("/", tags=["System"])
async def root():
    """Root entry point with service metadata."""
    return {
        "platform": "NEXUS RETAIL AI",
        "description": "Big Data Analytics & Forecasting Platform",
        "status": "online",
        "version": "2.4.0",
        "docs": "/docs",
    }


@app.get("/api/v1/health", tags=["System"])
async def health_check():
    """System health verification and dataset statistics."""
    try:
        metadata = data_engine.get_filter_metadata()
        return {
            "status": "healthy",
            "database": "active",
            "total_transactions": metadata["total_records"],
            "unique_customers": metadata["unique_customers"],
            "unique_skus": metadata["unique_skus"],
            "date_range": f"{metadata['min_date']} to {metadata['max_date']}",
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check encountered error: {str(e)}",
        )


@app.get("/api/v1/filters/metadata", tags=["Metadata"])
async def get_filter_metadata():
    """Retrieves unique store locations, categories, and date constraints."""
    try:
        return data_engine.get_filter_metadata()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching metadata: {str(e)}",
        )


@app.get("/api/v1/overview", tags=["Executive Analytics"])
async def get_overview(
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    store_location: Optional[str] = Query(None, description="Store location filter"),
    category: Optional[str] = Query(None, description="Product category filter"),
):
    """
    Returns Executive KPI metrics (Total Revenue, YoY Growth, Total Orders,
    Average Basket Size, Retention Rate, CSAT), historical sales with 7d MA,
    category revenue breakdown, and hourly velocity heatmap matrix.
    """
    try:
        return analytics_engine.get_executive_overview(
            start_date=start_date,
            end_date=end_date,
            store_location=store_location,
            category=category,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate executive overview: {str(e)}",
        )


@app.post("/api/v1/forecast", tags=["Forecasting & AI"])
async def generate_forecast(payload: ForecastRequest):
    """
    Executes Meta Prophet & LightGBM ensemble time-series pipeline.
    Returns:
    - Multi-step timeline (historical actuals + forecast with upper/lower bounds)
    - Seasonal decomposition (weekly, monthly trend patterns)
    - Accuracy metrics (RMSE, MAE, MAPE, R2)
    """
    try:
        result = forecasting_engine.run_forecast(
            horizon_days=payload.horizon_days,
            store_id=payload.store_id,
            category=payload.category,
            model_type=payload.model_type or "Ensemble (Prophet + LightGBM)",
            include_holidays=payload.include_holidays if payload.include_holidays is not None else True,
        )
        if "error" in result:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecast execution failed: {str(e)}",
        )


@app.get("/api/v1/customers/segmentation", tags=["Customer Intelligence"])
async def get_customer_segmentation(
    k: int = Query(4, ge=3, le=6, description="Number of K-Means customer clusters (3 to 6)")
):
    """
    Computes RFM scores (Recency, Frequency, Monetary), standardizes features,
    and fits K-Means clustering.
    Returns:
    - 3D coordinate vectors (R, F, M) for scatter visualization
    - Segment profiles (revenue share, AOV, churn risk)
    - Actionable marketing playbook recommendations
    """
    try:
        return customer_engine.perform_segmentation(k=k)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform customer segmentation: {str(e)}",
        )


@app.get("/api/v1/inventory/anomalies", tags=["Supply Chain Intelligence"])
async def get_inventory_anomalies():
    """
    Calculates stockout risk index, overstock capital loss ($),
    Days Inventory Outstanding (DIO), and SKU depletion & EOQ reorder recommendations.
    """
    try:
        return analytics_engine.get_inventory_intelligence()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate inventory intelligence: {str(e)}",
        )


@app.post("/api/v1/mlops/retrain", tags=["Model Management & MLOps"])
async def retrain_models():
    """Triggers backend refitting and updates performance metrics live."""
    try:
        return forecasting_engine.retrain_models()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model retraining failed: {str(e)}",
        )


@app.get("/api/v1/mlops/metrics", tags=["Model Management & MLOps"])
async def get_mlops_metrics():
    """Returns current model evaluation metrics, drift status, and feature importances."""
    try:
        return {
            "metrics": forecasting_engine.last_metrics,
            "feature_importances": forecasting_engine.feature_importances,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch MLOps metrics: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

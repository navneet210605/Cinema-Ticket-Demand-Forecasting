"""
CineForecast REST API Backend.
High-performance ASGI server built with Starlette (FastAPI core) and Uvicorn.
Serves real-time demand predictions using the serialized Joblib Gradient Boosting pipeline.
"""

import os
import json
from typing import Dict, Any
import numpy as np
import pandas as pd
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request

from src.predict import CinemaDemandPredictor

# Initialize predictor
predictor = CinemaDemandPredictor()

# Load evaluation and dataset caches
comp_path = "models/model_comparison.json"
fi_path = "models/feature_importance.json"
data_path = "data/raw/cinema_ticket_sales.csv"


async def health(request: Request) -> JSONResponse:
    """API healthcheck."""
    return JSONResponse({
        "status": "healthy",
        "service": "CineForecast REST API",
        "model_loaded": predictor.model is not None,
        "champion_model": "Gradient Boosting Regressor",
        "version": "2.0.0"
    })


async def root(request: Request) -> JSONResponse:
    """API documentation & endpoints overview."""
    return JSONResponse({
        "service": "CineForecast - Cinema Ticket Demand Forecasting API",
        "endpoints": {
            "health": "/health",
            "predict": "POST /api/predict",
            "metrics": "GET /api/metrics",
            "feature_importance": "GET /api/features",
            "dataset_summary": "GET /api/dataset-summary"
        }
    })


async def predict(request: Request) -> JSONResponse:
    """
    Generate demand prediction from JSON payload using serialized Joblib model.
    Enforces physical invariants: advance_bookings <= predicted <= capacity.
    """
    try:
        payload = await request.json()
        if not payload:
            return JSONResponse({"error": "Empty JSON payload provided"}, status_code=400)
        
        result = predictor.predict_single(payload)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_metrics(request: Request) -> JSONResponse:
    """Return model leaderboard and evaluation metrics on held-out test data."""
    if os.path.exists(comp_path):
        with open(comp_path, "r") as f:
            data = json.load(f)
        return JSONResponse(data)
    elif os.path.exists(os.path.join("..", comp_path)):
        with open(os.path.join("..", comp_path), "r") as f:
            data = json.load(f)
        return JSONResponse(data)
    return JSONResponse({"error": "Model evaluation metrics not found. Run train.py first."}, status_code=404)


async def get_features(request: Request) -> JSONResponse:
    """Return feature importance ranking."""
    if os.path.exists(fi_path):
        with open(fi_path, "r") as f:
            data = json.load(f)
        return JSONResponse(data)
    elif os.path.exists(os.path.join("..", fi_path)):
        with open(os.path.join("..", fi_path), "r") as f:
            data = json.load(f)
        return JSONResponse(data)
    return JSONResponse({"error": "Feature importance data not found."}, status_code=404)


async def get_dataset_summary(request: Request) -> JSONResponse:
    """Return dataset overview, statistics, and preview rows."""
    resolved_data = data_path if os.path.exists(data_path) else os.path.join("..", data_path)
    if os.path.exists(resolved_data):
        df = pd.read_csv(resolved_data)
        summary = {
            "source": "Domain-informed synthetic cinema simulation (Seed: 42)",
            "is_synthetic": True,
            "total_records": len(df),
            "total_features": len(df.columns),
            "target_variable": "tickets_sold",
            "missing_values_count": int(df.isnull().sum().sum()),
            "duplicate_records_count": int(df.duplicated(subset=['show_id']).sum()),
            "columns": list(df.columns),
            "numerical_summary": df.describe().T[["count", "mean", "std", "min", "50%", "max"]].to_dict(),
            "preview": df.head(10).to_dict(orient="records")
        }
        return JSONResponse(summary)
    return JSONResponse({"error": "Dataset file not found."}, status_code=404)


# Routes
routes = [
    Route("/", root, methods=["GET"]),
    Route("/health", health, methods=["GET"]),
    Route("/api/predict", predict, methods=["POST"]),
    Route("/api/metrics", get_metrics, methods=["GET"]),
    Route("/api/features", get_features, methods=["GET"]),
    Route("/api/dataset-summary", get_dataset_summary, methods=["GET"]),
]

# CORS Middleware to allow requests from web/index.html or Netlify
middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )
]

app = Starlette(debug=False, routes=routes, middleware=middleware)

if __name__ == "__main__":
    import uvicorn
    print("Starting CineForecast REST API on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

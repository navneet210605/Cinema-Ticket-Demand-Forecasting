"""
Utility functions for Cinema Ticket Demand Forecasting.
Provides reproducibility helpers, evaluation metrics (MAE, RMSE, WAPE),
and showtime/demand classification logic.
"""

import random
import numpy as np
import pandas as pd
from typing import Dict, Any, Union


def set_seed(seed: int = 42) -> None:
    """Ensure reproducibility across random generators."""
    random.seed(seed)
    np.random.seed(seed)


def calculate_metrics(y_true: Union[np.ndarray, pd.Series], 
                      y_pred: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Calculate regression performance metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - WAPE: Weighted Absolute Percentage Error (sum(|y - y_hat|) / sum(y) * 100)
    - R2: Coefficient of Determination
    - MedAE: Median Absolute Error
    """
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)
    
    # Clip negative predictions to zero if any
    y_p = np.clip(y_p, 0, None)
    
    mae = float(np.mean(np.abs(y_t - y_p)))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    medae = float(np.median(np.abs(y_t - y_p)))
    
    sum_actual = np.sum(y_t)
    if sum_actual > 0:
        wape = float((np.sum(np.abs(y_t - y_p)) / sum_actual) * 100.0)
    else:
        wape = 0.0
        
    # R2 calculation
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    ss_res = np.sum((y_t - y_p) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "WAPE": round(wape, 2),
        "R2": round(r2, 4),
        "MedAE": round(medae, 2)
    }


def categorize_showtime(time_val: str) -> str:
    """
    Categorize a showtime (HH:MM) into Morning, Afternoon, Evening, or Night.
    """
    try:
        if isinstance(time_val, str):
            hour = int(time_val.split(":")[0])
        elif hasattr(time_val, "hour"):
            hour = time_val.hour
        else:
            hour = int(time_val)
    except Exception:
        hour = 12

    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


def categorize_demand(occupancy_pct: float) -> str:
    """
    Categorize occupancy percentage into Low, Medium, or High demand.
    """
    if occupancy_pct >= 75.0:
        return "High Demand"
    elif occupancy_pct >= 40.0:
        return "Medium Demand"
    else:
        return "Low Demand"

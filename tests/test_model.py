"""
Unit tests for model training and metric calculations.
"""

import pytest
import numpy as np
from src.utils import calculate_metrics, categorize_demand


def test_calculate_metrics():
    y_true = np.array([100, 150, 200, 250])
    y_pred = np.array([110, 140, 205, 240])

    metrics = calculate_metrics(y_true, y_pred)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "WAPE" in metrics
    assert "R2" in metrics
    
    assert metrics["MAE"] == 8.75
    assert metrics["WAPE"] > 0
    assert metrics["R2"] > 0.90


def test_categorize_demand():
    assert categorize_demand(85.0) == "High Demand"
    assert categorize_demand(60.0) == "Medium Demand"
    assert categorize_demand(30.0) == "Low Demand"

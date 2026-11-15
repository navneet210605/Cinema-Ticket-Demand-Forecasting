"""
Unit tests for model training, metrics calculation, and evaluation suites.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from src.utils import calculate_metrics, categorize_demand
from src.evaluate import evaluate_cold_start_cohort


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


def test_ml_candidate_outperforms_baseline():
    """Verify ML models strictly outperform Capacity Baseline on held-out test data."""
    comp_path = "models/model_comparison.json"
    if not os.path.exists(comp_path):
        pytest.skip("model_comparison.json not found")
        
    with open(comp_path, "r") as f:
        data = json.load(f)
        
    models = data["models"]
    assert "Gradient Boosting" in models
    assert "Capacity Baseline" in models
    
    gb_wape = models["Gradient Boosting"]["test"]["WAPE"]
    base_wape = models["Capacity Baseline"]["test"]["WAPE"]
    
    # Gradient boosting must show at least a 50% relative error reduction over naive baseline
    assert gb_wape < base_wape
    relative_reduction = (base_wape - gb_wape) / base_wape
    assert relative_reduction > 0.50, f"Expected >50% improvement over baseline, got {relative_reduction:.1%}"


def test_cold_start_cohort_evaluation():
    """Verify quantitative evaluation of cold-start and premiere screenings."""
    cohorts = evaluate_cold_start_cohort()
    if not cohorts:
        pytest.skip("Processed test data or comparison JSON not available")
        
    assert "overall_test" in cohorts
    assert "mature_shows" in cohorts
    assert "cold_start_shows" in cohorts
    assert "day_0_premieres" in cohorts
    
    # Day 0 premieres and cold start shows must have valid bounded WAPE and MAE
    assert cohorts["day_0_premieres"]["metrics"]["WAPE"] < 15.0
    assert cohorts["cold_start_shows"]["metrics"]["MAE"] < 15.0


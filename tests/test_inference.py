"""
Unit tests for inference and cold-start prediction logic.
"""

import pytest
import pandas as pd
from src.predict import CinemaDemandPredictor


@pytest.fixture(scope="module")
def predictor():
    return CinemaDemandPredictor()


def test_single_prediction_format(predictor):
    show_sample = {
        "movie_name": "Velocity Drift",
        "genre": "Action",
        "movie_language": "English",
        "movie_duration": 130,
        "release_days": 5,
        "show_date": "2024-05-18",
        "show_time": "19:45",
        "day_of_week": "Saturday",
        "is_weekend": 1,
        "is_holiday": 0,
        "screen_capacity": 250,
        "historical_average_sales": 180.0,
        "previous_show_occupancy": 0.75,
        "ticket_price": 16.0,
        "advance_booking_count": 120,
        "marketing_spend": 28000.0,
        "competitor_show_count": 2
    }
    result = predictor.predict_single(show_sample)
    
    assert "predicted_tickets_sold" in result
    assert "expected_occupancy_pct" in result
    assert "demand_category" in result
    assert "recommended_staffing" in result
    assert "capacity_action" in result

    # Capacity constraints check
    assert result["predicted_tickets_sold"] >= show_sample["advance_booking_count"]
    assert result["predicted_tickets_sold"] <= show_sample["screen_capacity"]
    assert 0.0 <= result["expected_occupancy_pct"] <= 100.0


def test_cold_start_fallback(predictor):
    # Cold start: newly released movie, missing historical_average_sales and previous_show_occupancy
    cold_start_sample = {
        "movie_name": "New Unreleased Blockbuster",
        "genre": "Sci-Fi",
        "movie_language": "English",
        "movie_duration": 140,
        "release_days": 0,
        "show_date": "2024-10-10",
        "show_time": "19:45",
        "screen_capacity": 200,
        "advance_booking_count": 60,
        "ticket_price": 15.0
        # Notice historical_average_sales & previous_show_occupancy are omitted
    }
    result = predictor.predict_single(cold_start_sample)
    assert result["predicted_tickets_sold"] >= 60
    assert result["predicted_tickets_sold"] <= 200


def test_batch_prediction(predictor):
    df_sample = pd.DataFrame([
        {
            "movie_name": "Laugh Out Loud",
            "genre": "Comedy",
            "movie_language": "English",
            "movie_duration": 105,
            "release_days": 14,
            "show_date": "2024-06-01",
            "show_time": "13:15",
            "screen_capacity": 150,
            "advance_booking_count": 40,
            "historical_average_sales": 70.0,
            "previous_show_occupancy": 0.45,
            "ticket_price": 12.0,
            "marketing_spend": 10000.0,
            "competitor_show_count": 1
        },
        {
            "movie_name": "Midnight Shadows",
            "genre": "Horror",
            "movie_language": "Spanish",
            "movie_duration": 95,
            "release_days": 2,
            "show_date": "2024-06-01",
            "show_time": "22:30",
            "screen_capacity": 200,
            "advance_booking_count": 95,
            "historical_average_sales": 130.0,
            "previous_show_occupancy": 0.70,
            "ticket_price": 14.0,
            "marketing_spend": 18000.0,
            "competitor_show_count": 0
        }
    ])
    batch_res = predictor.predict_batch(df_sample)
    assert len(batch_res) == 2
    assert "predicted_tickets_sold" in batch_res.columns
    assert "expected_occupancy_pct" in batch_res.columns
    assert "demand_category" in batch_res.columns


def test_edge_case_advance_equals_or_exceeds_capacity(predictor):
    # Edge case 1: Pre-sold out screening
    sample_sold_out = {
        "movie_name": "Blockbuster Premiere",
        "genre": "Action",
        "screen_capacity": 250,
        "advance_booking_count": 250,
        "ticket_price": 18.0
    }
    res1 = predictor.predict_single(sample_sold_out)
    assert res1["predicted_tickets_sold"] == 250
    assert res1["expected_occupancy_pct"] == 100.0
    assert res1["remaining_seats"] == 0

    # Edge case 2: Advance booking exceeds capacity due to invalid user input
    sample_overflow = {
        "movie_name": "Overflow Show",
        "genre": "Sci-Fi",
        "screen_capacity": 200,
        "advance_booking_count": 280,
        "ticket_price": 15.0
    }
    res2 = predictor.predict_single(sample_overflow)
    assert res2["predicted_tickets_sold"] == 200
    assert res2["expected_occupancy_pct"] == 100.0


def test_confidence_interval_bounds(predictor):
    """Verify empirical 95% confidence interval calculations and bounds."""
    sample = {
        "movie_name": "Galactic Horizon",
        "genre": "Sci-Fi",
        "screen_capacity": 300,
        "advance_booking_count": 100,
        "ticket_price": 16.0
    }
    res = predictor.predict_single(sample)
    assert "confidence_interval_95" in res
    ci = res["confidence_interval_95"]
    assert "lower" in ci
    assert "upper" in ci
    
    # Check logical bounds
    assert ci["lower"] <= res["predicted_tickets_sold"]
    assert res["predicted_tickets_sold"] <= ci["upper"]
    assert ci["lower"] >= sample["advance_booking_count"]
    assert ci["upper"] <= sample["screen_capacity"]


def test_operational_recommendations_and_disclaimer(predictor):
    """Verify decision-support rules, model metadata, and regulatory disclaimers."""
    sample = {
        "movie_name": "Family Fun",
        "genre": "Animation",
        "screen_capacity": 200,
        "advance_booking_count": 180,  # High occupancy > 85%
        "ticket_price": 12.0
    }
    res = predictor.predict_single(sample)
    
    assert res["demand_category"] == "High Demand"
    assert "Full Staffing Required" in res["recommended_staffing"]
    assert "reallocation" in res["capacity_action"].lower() or "surge" in res["capacity_action"].lower() or "screen" in res["capacity_action"].lower()
    assert "model_used" in res
    assert res["model_used"] == "Gradient Boosting Regressor"
    assert "disclaimer" in res
    assert "decision-support" in res["disclaimer"].lower()



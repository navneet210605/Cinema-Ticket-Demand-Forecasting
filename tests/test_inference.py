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

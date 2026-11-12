"""
Unit tests for data generation.
Validates synthetic dataset integrity, business logic constraints, and reproducibility.
"""

import pytest
import pandas as pd
from src.data_generation import generate_cinema_dataset


def test_dataset_record_count_and_columns():
    df = generate_cinema_dataset(num_records=1200, seed=42)
    assert len(df) == 1200
    expected_cols = [
        "show_id", "movie_name", "genre", "movie_language", "movie_duration",
        "release_days", "show_date", "show_time", "day_of_week", "is_weekend",
        "is_holiday", "screen_capacity", "historical_average_sales",
        "previous_show_occupancy", "ticket_price", "advance_booking_count",
        "marketing_spend", "competitor_show_count", "tickets_sold"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing required column: {col}"


def test_reproducibility():
    df1 = generate_cinema_dataset(num_records=500, seed=42)
    df2 = generate_cinema_dataset(num_records=500, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_no_duplicate_shows():
    df = generate_cinema_dataset(num_records=1000, seed=42)
    assert df["show_id"].nunique() == 1000


def test_capacity_and_advance_booking_constraints():
    df = generate_cinema_dataset(num_records=1500, seed=42)
    
    # 1. Tickets sold cannot exceed screen capacity
    exceeding_capacity = df[df["tickets_sold"] > df["screen_capacity"]]
    assert len(exceeding_capacity) == 0, "Found shows where tickets_sold > screen_capacity"
    
    # 2. Advance booking count cannot exceed tickets sold
    exceeding_adv = df[df["advance_booking_count"] > df["tickets_sold"]]
    assert len(exceeding_adv) == 0, "Found shows where advance_booking_count > tickets_sold"

    # 3. All tickets sold are non-negative
    assert (df["tickets_sold"] >= 0).all()
    assert (df["advance_booking_count"] >= 0).all()

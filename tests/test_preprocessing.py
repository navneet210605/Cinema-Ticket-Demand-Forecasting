"""
Unit tests for preprocessing and feature engineering.
"""

import pytest
import pandas as pd
from src.data_generation import generate_cinema_dataset
from src.preprocessing import (
    engineer_features,
    split_chronologically,
    create_preprocessor,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)
from src.utils import categorize_showtime


def test_showtime_categorization():
    assert categorize_showtime("10:00") == "Morning"
    assert categorize_showtime("13:15") == "Afternoon"
    assert categorize_showtime("19:45") == "Evening"
    assert categorize_showtime("22:30") == "Night"


def test_feature_engineering_columns():
    df = generate_cinema_dataset(num_records=100, seed=42)
    engineered_df = engineer_features(df)
    
    expected_new_cols = [
        "month", "day", "showtime_category", "advance_booking_ratio",
        "price_per_minute", "marketing_spend_log", "competitor_density",
        "hist_sales_ratio", "release_decay_factor"
    ]
    for col in expected_new_cols:
        assert col in engineered_df.columns, f"Engineered column {col} is missing"

    assert (engineered_df["advance_booking_ratio"] <= 1.0).all()
    assert (engineered_df["price_per_minute"] > 0).all()


def test_chronological_split_no_leakage():
    df = generate_cinema_dataset(num_records=500, seed=42)
    train_df, val_df, test_df = split_chronologically(df, 0.7, 0.15)

    assert len(train_df) == 350
    assert len(val_df) == 75
    assert len(test_df) == 75

    # Check temporal ordering
    assert train_df["show_date"].max() <= val_df["show_date"].min()
    assert val_df["show_date"].max() <= test_df["show_date"].min()


def test_preprocessor_transformation():
    df = generate_cinema_dataset(num_records=200, seed=42)
    df_eng = engineer_features(df)
    
    features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df_eng[features]
    
    preprocessor = create_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    
    assert X_transformed.shape[0] == 200
    assert not pd.isna(X_transformed).any()

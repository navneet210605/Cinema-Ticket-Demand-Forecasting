"""
Preprocessing and Feature Engineering Module.

Defines the exact forecast cutoff (6 hours prior to showtime) and engineers
time-aware, demand-specific features without target leakage.
Provides Scikit-Learn transformers for numerical scaling and categorical encoding,
and chronological train/validation/test split utilities.
"""

import os
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.utils import categorize_showtime


# Define the operational forecast horizon
FORECAST_CUTOFF_DESCRIPTION = (
    "Forecast Cutoff: Strictly 6 hours prior to showtime. "
    "At this moment, advance bookings up to the cutoff, schedule, theater capacity, "
    "ticket price, and competitor shows are known. Walk-in sales are unknown."
)

CATEGORICAL_FEATURES = [
    "genre",
    "movie_language",
    "day_of_week",
    "showtime_category"
]

NUMERICAL_FEATURES = [
    "movie_duration",
    "release_days",
    "month",
    "day",
    "is_weekend",
    "is_holiday",
    "screen_capacity",
    "historical_average_sales",
    "previous_show_occupancy",
    "ticket_price",
    "advance_booking_count",
    "marketing_spend",
    "competitor_show_count",
    "advance_booking_ratio",
    "price_per_minute",
    "marketing_spend_log",
    "competitor_density",
    "hist_sales_ratio",
    "release_decay_factor"
]

TARGET_COL = "tickets_sold"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate domain-specific features from raw cinema show records.
    Ensures safe, leakage-free feature computation.
    """
    data = df.copy()
    
    # 1. Parse date elements
    if "show_date" in data.columns:
        show_dates = pd.to_datetime(data["show_date"])
        data["month"] = show_dates.dt.month
        data["day"] = show_dates.dt.day
    else:
        if "month" not in data.columns:
            data["month"] = 1
        if "day" not in data.columns:
            data["day"] = 1

    # 2. Derive showtime category
    if "showtime_category" not in data.columns:
        if "show_time" in data.columns:
            data["showtime_category"] = data["show_time"].apply(categorize_showtime)
        else:
            data["showtime_category"] = "Evening"

    # 3. Advance booking ratio (share of auditorium pre-booked at cutoff)
    capacity = np.maximum(data["screen_capacity"].values, 1)
    adv_count = np.maximum(data["advance_booking_count"].values, 0)
    data["advance_booking_ratio"] = np.round(adv_count / capacity, 4)

    # 4. Price per minute (perceived customer value density)
    duration = np.maximum(data["movie_duration"].values, 1)
    data["price_per_minute"] = np.round(data["ticket_price"].values / duration, 4)

    # 5. Log-transformed marketing spend (captures diminishing marginal returns)
    mktg = np.maximum(data["marketing_spend"].values, 0)
    data["marketing_spend_log"] = np.round(np.log1p(mktg), 4)

    # 6. Competitor density normalized by screen size
    data["competitor_density"] = np.round(
        data["competitor_show_count"].values / np.maximum(capacity / 100.0, 1.0), 4
    )

    # 7. Historical sales to capacity ratio
    hist_sales = np.maximum(data["historical_average_sales"].values, 0)
    data["hist_sales_ratio"] = np.round(hist_sales / capacity, 4)

    # 8. Exponential theatrical release decay factor
    rel_days = np.maximum(data["release_days"].values, 0)
    data["release_decay_factor"] = np.round(np.exp(-0.045 * rel_days), 4)

    return data


def create_preprocessor() -> ColumnTransformer:
    """
    Build Scikit-Learn ColumnTransformer for numerical scaling and one-hot encoding.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor


def split_chronologically(df: pd.DataFrame, 
                          train_ratio: float = 0.70, 
                          val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform strict chronological train, validation, and test split.
    Prevents temporal data leakage.
    """
    # Sort chronologically by date and showtime
    df_sorted = df.sort_values(by=["show_date", "show_time"]).reset_index(drop=True)
    
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df_sorted.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df_sorted.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[val_end:].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df


def save_processed_splits(train_df: pd.DataFrame, 
                          val_df: pd.DataFrame, 
                          test_df: pd.DataFrame, 
                          output_dir: str = "data/processed") -> Dict[str, str]:
    """Save processed splits into CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    return {
        "train": train_path,
        "val": val_path,
        "test": test_path
    }


if __name__ == "__main__":
    from src.data_generation import generate_cinema_dataset
    
    raw_path = "data/raw/cinema_ticket_sales.csv"
    if os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path)
    else:
        df_raw = generate_cinema_dataset(num_records=2500, seed=42)
        
    df_engineered = engineer_features(df_raw)
    tr, va, te = split_chronologically(df_engineered)
    paths = save_processed_splits(tr, va, te)
    
    print("Preprocessing & Chronological split completed successfully:")
    print(f"  Train: {tr.shape} ({tr['show_date'].min()} to {tr['show_date'].max()})")
    print(f"  Validation: {va.shape} ({va['show_date'].min()} to {va['show_date'].max()})")
    print(f"  Test: {test_df.shape if 'test_df' in locals() else te.shape} ({te['show_date'].min()} to {te['show_date'].max()})")

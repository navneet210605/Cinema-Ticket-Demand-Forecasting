"""
Inference Module for Cinema Ticket Demand Forecasting.

Handles single-show and batch demand predictions using the serialized Joblib model.
Features automatic cold-start imputation, dynamic feature engineering,
domain-constrained clipping (advance_booking <= predicted <= screen_capacity),
occupancy calculation, and operational business recommendations.
"""

import os
from typing import Dict, Any, Union, List
import numpy as np
import pandas as pd
import joblib

from src.preprocessing import (
    engineer_features,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES
)
from src.utils import categorize_demand, categorize_showtime


class CinemaDemandPredictor:
    """
    Production-ready Cinema Demand Predictor.
    Loads trained ML pipeline and outputs demand forecasts and operational guidelines.
    """
    def __init__(self, model_path: str = "models/best_model.joblib"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. "
                "Please run `python -m src.train` first."
            )
        self.model = joblib.load(model_path)
        self.feature_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    def _prepare_input_df(self, input_data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """
        Validate input format, handle cold-start missing values, and engineer features.
        """
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise ValueError("Input data must be a dictionary or a Pandas DataFrame.")

        # Handle Cold-Start defaults if parameters are omitted or NaN
        for idx in df.index:
            cap = df.at[idx, "screen_capacity"] if "screen_capacity" in df.columns and pd.notna(df.at[idx, "screen_capacity"]) else 200
            df.at[idx, "screen_capacity"] = int(cap)

            # Metadata defaults
            if "genre" not in df.columns or pd.isna(df.at[idx, "genre"]):
                df.at[idx, "genre"] = "Action"
            if "movie_language" not in df.columns or pd.isna(df.at[idx, "movie_language"]):
                df.at[idx, "movie_language"] = "English"
            if "movie_duration" not in df.columns or pd.isna(df.at[idx, "movie_duration"]):
                df.at[idx, "movie_duration"] = 120
            if "release_days" not in df.columns or pd.isna(df.at[idx, "release_days"]):
                df.at[idx, "release_days"] = 7
            if "show_time" not in df.columns or pd.isna(df.at[idx, "show_time"]):
                df.at[idx, "show_time"] = "19:45"
            if "show_date" not in df.columns or pd.isna(df.at[idx, "show_date"]):
                df.at[idx, "show_date"] = "2024-06-15"

            # Holiday & Weekend defaults
            if "is_holiday" not in df.columns or pd.isna(df.at[idx, "is_holiday"]):
                df.at[idx, "is_holiday"] = 0

            # Day of week & weekend fallback if show_date exists
            if "day_of_week" not in df.columns or pd.isna(df.at[idx, "day_of_week"]):
                dt = pd.to_datetime(df.at[idx, "show_date"])
                df.at[idx, "day_of_week"] = dt.strftime("%A")
            
            if "is_weekend" not in df.columns or pd.isna(df.at[idx, "is_weekend"]):
                dt = pd.to_datetime(df.at[idx, "show_date"])
                df.at[idx, "is_weekend"] = 1 if dt.weekday() in [4, 5, 6] else 0

            # 1. Historical average sales fallback
            if "historical_average_sales" not in df.columns or pd.isna(df.at[idx, "historical_average_sales"]):
                # Cold start: Estimate 55% of auditorium capacity
                df.at[idx, "historical_average_sales"] = round(cap * 0.55, 1)

            # 2. Previous show occupancy fallback
            if "previous_show_occupancy" not in df.columns or pd.isna(df.at[idx, "previous_show_occupancy"]):
                df.at[idx, "previous_show_occupancy"] = 0.50

            # 3. Advance booking fallback
            if "advance_booking_count" not in df.columns or pd.isna(df.at[idx, "advance_booking_count"]):
                df.at[idx, "advance_booking_count"] = 0

            # 4. Competitor shows fallback
            if "competitor_show_count" not in df.columns or pd.isna(df.at[idx, "competitor_show_count"]):
                df.at[idx, "competitor_show_count"] = 1

            # 5. Marketing spend fallback
            if "marketing_spend" not in df.columns or pd.isna(df.at[idx, "marketing_spend"]):
                df.at[idx, "marketing_spend"] = 15000.0

            # 6. Ticket price fallback
            if "ticket_price" not in df.columns or pd.isna(df.at[idx, "ticket_price"]):
                df.at[idx, "ticket_price"] = 14.50

        # Apply feature engineering
        df_engineered = engineer_features(df)
        return df_engineered

    def predict_single(self, show_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate demand prediction and business advice for a single show.
        """
        df_prepared = self._prepare_input_df(show_params)
        
        # Ensure all feature columns exist
        X = df_prepared[self.feature_columns]
        raw_pred = self.model.predict(X)[0]

        capacity = int(df_prepared["screen_capacity"].iloc[0])
        adv_booking = int(df_prepared["advance_booking_count"].iloc[0])

        # Enforce physical reality:
        # Final tickets cannot be less than tickets already sold in advance
        # Final tickets cannot exceed screen capacity
        clamped_tickets = int(np.clip(np.round(raw_pred), adv_booking, capacity))
        occupancy_pct = round((clamped_tickets / capacity) * 100.0, 1)
        remaining_seats = max(0, capacity - clamped_tickets)
        demand_cat = categorize_demand(occupancy_pct)

        # Operational business logic
        if occupancy_pct >= 80.0:
            staffing = "Full Staffing Required (4 Ushes, 3 Concession Counters, 2 Ticket Desks)"
            capacity_action = "High Demand Surge: Consider adding an extra midnight show or shifting to a 400+ seat IMAX screen."
            pricing_recommendation = "Optimal demand: Maintain premium pricing or apply weekend surge (+10-15%)."
        elif occupancy_pct >= 50.0:
            staffing = "Standard Staffing (2 Ushers, 2 Concession Counters, 1 Ticket Desk)"
            capacity_action = "Healthy Demand: Screen capacity is well-utilized."
            pricing_recommendation = "Standard pricing structure is performing well."
        else:
            staffing = "Light Staffing (1 Usher, 1 Concession Counter, Kiosk check-in)"
            capacity_action = "Underutilized Screen: Consider shifting to smaller screen (120 seats) to reduce HVAC and projection costs."
            pricing_recommendation = "Low Demand Warning: Launch flash combo discounts (Popcorn + Ticket) to stimulate walk-in sales."

        return {
            "predicted_tickets_sold": clamped_tickets,
            "screen_capacity": capacity,
            "advance_booking_count": adv_booking,
            "expected_occupancy_pct": occupancy_pct,
            "remaining_seats": remaining_seats,
            "demand_category": demand_cat,
            "recommended_staffing": staffing,
            "capacity_action": capacity_action,
            "pricing_recommendation": pricing_recommendation,
            "raw_model_prediction": round(float(raw_pred), 1)
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for an entire schedule of shows.
        """
        df_prepared = self._prepare_input_df(df)
        X = df_prepared[self.feature_columns]
        raw_preds = self.model.predict(X)

        capacities = df_prepared["screen_capacity"].values
        adv_bookings = df_prepared["advance_booking_count"].values

        clamped = np.clip(np.round(raw_preds), adv_bookings, capacities).astype(int)
        occupancies = np.round((clamped / capacities) * 100.0, 1)

        result_df = df.copy()
        result_df["predicted_tickets_sold"] = clamped
        result_df["expected_occupancy_pct"] = occupancies
        result_df["remaining_seats"] = capacities - clamped
        result_df["demand_category"] = [categorize_demand(occ) for occ in occupancies]

        return result_df


if __name__ == "__main__":
    predictor = CinemaDemandPredictor()
    sample_show = {
        "movie_name": "Galactic Odyssey",
        "genre": "Sci-Fi",
        "movie_language": "English",
        "movie_duration": 152,
        "release_days": 3,
        "show_date": "2024-09-20",
        "show_time": "19:45",
        "day_of_week": "Friday",
        "is_weekend": 1,
        "is_holiday": 0,
        "screen_capacity": 350,
        "historical_average_sales": 240.0,
        "previous_show_occupancy": 0.82,
        "ticket_price": 18.50,
        "advance_booking_count": 195,
        "marketing_spend": 32000.0,
        "competitor_show_count": 1
    }
    
    pred = predictor.predict_single(sample_show)
    print("Inference Test:")
    for k, v in pred.items():
        print(f"  {k}: {v}")

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
        # Support running from root or from subdirectories like app/ or notebooks/
        resolved_path = model_path
        if not os.path.exists(resolved_path):
            alt_path = os.path.join("..", model_path)
            if os.path.exists(alt_path):
                resolved_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Trained model not found at '{model_path}' or '{alt_path}'. "
                    "Please run `python -m src.train` first."
                )
        self.model = joblib.load(resolved_path)
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

        capacity = int(max(1, df_prepared["screen_capacity"].iloc[0]))
        # Advance booking cannot exceed physical screen capacity
        adv_booking = int(min(capacity, max(0, df_prepared["advance_booking_count"].iloc[0])))

        # Enforce physical reality:
        # Final tickets cannot be less than tickets already sold in advance
        # Final tickets cannot exceed screen capacity
        clamped_tickets = int(np.clip(np.round(raw_pred), adv_booking, capacity))
        occupancy_pct = round((clamped_tickets / capacity) * 100.0, 1)
        remaining_seats = max(0, capacity - clamped_tickets)
        demand_cat = categorize_demand(occupancy_pct)

        # Calculate empirical prediction uncertainty interval (95% confidence based on Test RMSE: 15.20 tickets)
        test_rmse = 15.20
        margin = 1.96 * test_rmse
        ci_lower = int(max(adv_booking, np.round(clamped_tickets - margin)))
        ci_upper = int(min(capacity, np.round(clamped_tickets + margin)))

        # Operational business logic & promotional recommendations
        if occupancy_pct >= 75.0:
            staffing = "Full Staffing Required: 4 Ushers, 3 Concession Counters, 2 Ticket Desk Attendants"
            capacity_action = "High Demand Surge: Screen is near full capacity. Consider upgrading to a 350+ seat premium auditorium or adding a late-night screening."
            pricing_recommendation = "Optimal Demand: Maintain premium ticket pricing; apply weekend surge pricing (+10-15%)."
            promotional_recommendation = "High organic demand: No general admission discounts needed. Promote premium concession combos (Gourmet Popcorn + Beverage)."
        elif occupancy_pct >= 40.0:
            staffing = "Standard Staffing: 2 Ushers, 2 Concession Counters, 1 Ticket Desk Attendant"
            capacity_action = "Healthy Demand: Screen capacity is well-utilized. Auditorium sizing matches demand."
            pricing_recommendation = "Stable Demand: Standard ticket pricing structure is performing optimally."
            promotional_recommendation = "Steady attendance: Offer loyalty member rewards and mid-tier concession packages."
        else:
            staffing = "Light Staffing: 1 Usher, 1 Concession Counter, Self-service kiosk check-in"
            capacity_action = "Underutilized Screen: Consider shifting to a smaller 120-seat auditorium to save projection and HVAC energy costs."
            pricing_recommendation = "Low Demand Warning: Implement flash discount pricing or matinee promotional rates to stimulate walk-in sales."
            promotional_recommendation = "Low foot-traffic alert: Launch combo deals (Free Soda with Popcorn, 2-for-1 student/senior tickets) to maximize walk-in conversions."

        return {
            "predicted_tickets_sold": clamped_tickets,
            "screen_capacity": capacity,
            "advance_booking_count": adv_booking,
            "expected_occupancy_pct": occupancy_pct,
            "remaining_seats": remaining_seats,
            "available_seats": remaining_seats,
            "demand_category": demand_cat,
            "model_used": "Gradient Boosting Regressor",
            "confidence_interval_95": {
                "lower": ci_lower,
                "upper": ci_upper
            },
            "prediction_interval_str": f"[{ci_lower} - {ci_upper} tickets]",
            "recommended_staffing": staffing,
            "capacity_action": capacity_action,
            "pricing_recommendation": pricing_recommendation,
            "promotional_recommendation": promotional_recommendation,
            "raw_model_prediction": round(float(raw_pred), 1),
            "occupancy_formula": f"({clamped_tickets} / {capacity}) × 100 = {occupancy_pct}%",
            "disclaimer": "Notice: Business recommendations are decision-support guidelines based on statistical demand heuristics, not guaranteed operational outcomes."
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for an entire schedule of shows.
        """
        df_prepared = self._prepare_input_df(df)
        X = df_prepared[self.feature_columns]
        raw_preds = self.model.predict(X)

        capacities = np.maximum(df_prepared["screen_capacity"].values, 1)
        adv_bookings = np.clip(df_prepared["advance_booking_count"].values, 0, capacities)

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

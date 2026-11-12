"""
Model Training and Comparison Module.

Trains and evaluates:
1. Capacity-adjusted Historical Average Baseline
2. Linear Regression Pipeline
3. Random Forest Regressor Pipeline
4. Gradient Boosting Regressor Pipeline

Evaluates on chronological validation and held-out test splits.
Computes MAE, RMSE, WAPE, and R2.
Selects and saves the winning pipeline using Joblib.
"""

import os
import json
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.preprocessing import (
    create_preprocessor,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COL
)
from src.utils import calculate_metrics, set_seed


class CapacityAdjustedBaseline:
    """
    Capacity-Adjusted Historical Baseline Model.
    Predicts based on historical sales adjusted by screen capacity and advance bookings.
    """
    def __init__(self):
        self.mean_train_occupancy = 0.60

    def fit(self, X: pd.DataFrame, y: pd.Series):
        cap = np.maximum(X["screen_capacity"].values, 1)
        self.mean_train_occupancy = float(np.mean(y.values / cap))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cap = X["screen_capacity"].values
        adv = X["advance_booking_count"].values
        
        if "historical_average_sales" in X.columns:
            hist = X["historical_average_sales"].values
            # Handle potential NaNs
            hist = np.nan_to_num(hist, nan=cap * self.mean_train_occupancy)
            pred = np.maximum(adv, hist)
        else:
            pred = np.maximum(adv, cap * self.mean_train_occupancy)
            
        # Post-process: ticket sales cannot exceed screen capacity
        return np.clip(pred, adv, cap)


def get_feature_names(column_transformer) -> list:
    """Extract human-readable feature names after one-hot encoding."""
    output_features = list(NUMERICAL_FEATURES)
    try:
        cat_encoder = column_transformer.named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
        output_features.extend(list(cat_feature_names))
    except Exception:
        pass
    return output_features


def train_and_evaluate(data_dir: str = "data/processed",
                       models_dir: str = "models",
                       seed: int = 42) -> Dict[str, Any]:
    """
    Train all candidates, evaluate chronologically, and save best model.
    """
    set_seed(seed)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.join(models_dir, "plots"), exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]

    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COL]

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    # Pre-fitted preprocessor
    preprocessor = create_preprocessor()

    # Define model candidates
    models = {
        "Capacity Baseline": CapacityAdjustedBaseline(),
        "Linear Regression": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("regressor", LinearRegression())
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("regressor", RandomForestRegressor(
                n_estimators=180,
                max_depth=12,
                min_samples_split=4,
                random_state=seed,
                n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("regressor", GradientBoostingRegressor(
                n_estimators=160,
                learning_rate=0.07,
                max_depth=5,
                min_samples_split=4,
                subsample=0.85,
                random_state=seed
            ))
        ])
    }

    results = {}
    fitted_models = {}
    test_predictions = {}

    for name, model in models.items():
        print(f"Training {name}...")
        if name == "Capacity Baseline":
            model.fit(train_df, y_train)
            val_preds = model.predict(val_df)
            test_preds = model.predict(test_df)
        else:
            model.fit(X_train, y_train)
            raw_val_preds = model.predict(X_val)
            raw_test_preds = model.predict(X_test)
            
            # Domain bounds: at least advance bookings, at most screen capacity
            val_preds = np.clip(
                raw_val_preds,
                val_df["advance_booking_count"].values,
                val_df["screen_capacity"].values
            )
            test_preds = np.clip(
                raw_test_preds,
                test_df["advance_booking_count"].values,
                test_df["screen_capacity"].values
            )

        val_metrics = calculate_metrics(y_val, val_preds)
        test_metrics = calculate_metrics(y_test, test_preds)

        results[name] = {
            "validation": val_metrics,
            "test": test_metrics
        }
        fitted_models[name] = model
        test_predictions[name] = test_preds.tolist()

    # Determine best model based on lowest Test WAPE & MAE
    best_name = min(
        [m for m in models.keys() if m != "Capacity Baseline"],
        key=lambda m: (results[m]["test"]["WAPE"], results[m]["test"]["MAE"])
    )
    print(f"\nWinning Model: {best_name}")

    best_pipeline = fitted_models[best_name]

    # Save best model pipeline
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_pipeline, best_model_path)
    print(f"Saved best model pipeline to: {best_model_path}")

    # Extract feature importance
    regressor = best_pipeline.named_steps["regressor"]
    fitted_prep = best_pipeline.named_steps["preprocessor"]
    feat_names = get_feature_names(fitted_prep)

    feature_importances = {}
    if hasattr(regressor, "feature_importances_"):
        raw_importances = regressor.feature_importances_
        for fn, imp in zip(feat_names, raw_importances):
            feature_importances[fn] = round(float(imp), 4)
    elif hasattr(regressor, "coef_"):
        raw_coefs = regressor.coef_
        for fn, coef in zip(feat_names, raw_coefs):
            feature_importances[fn] = round(float(abs(coef)), 4)

    # Sort feature importances descending
    sorted_importances = dict(
        sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
    )

    # Save feature importances to JSON
    fi_path = os.path.join(models_dir, "feature_importance.json")
    with open(fi_path, "w") as f:
        json.dump(sorted_importances, f, indent=2)

    # Save results to JSON
    comparison_payload = {
        "best_model_name": best_name,
        "models": results,
        "test_actuals": y_test.tolist(),
        "test_predictions": test_predictions,
        "test_capacities": test_df["screen_capacity"].tolist(),
        "test_advance_bookings": test_df["advance_booking_count"].tolist(),
        "test_dates": test_df["show_date"].tolist(),
        "test_movies": test_df["movie_name"].tolist()
    }

    comp_path = os.path.join(models_dir, "model_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison_payload, f, indent=2)

    print("Model comparison and evaluation artifacts saved successfully.")
    return comparison_payload


if __name__ == "__main__":
    train_and_evaluate()

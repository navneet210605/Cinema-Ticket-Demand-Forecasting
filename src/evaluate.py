"""
Model Evaluation and Diagnostic Reporting Module.

Provides comprehensive diagnostic evaluation:
- Actual vs Predicted regression plots
- Residual analysis (distribution and heteroscedasticity)
- Model comparison tables & charts
- Segmented demand performance (by Genre, Showtime, Weekend)
- Cold-start handling analysis
"""

import os
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_evaluation_plots(comparison_json_path: str = "models/model_comparison.json",
                              output_dir: str = "models/plots") -> Dict[str, str]:
    """
    Generate static publication-quality diagnostic charts for technical reporting.
    """
    if not os.path.exists(comparison_json_path) and os.path.exists(os.path.join("..", comparison_json_path)):
        comparison_json_path = os.path.join("..", comparison_json_path)
    if not os.path.exists(output_dir) and os.path.exists(os.path.join("..", output_dir)):
        output_dir = os.path.join("..", output_dir)

    os.makedirs(output_dir, exist_ok=True)
    
    with open(comparison_json_path, "r") as f:
        data = json.load(f)

    actuals = np.array(data["test_actuals"])
    models = data["models"]
    best_model_name = data["best_model_name"]
    best_preds = np.array(data["test_predictions"][best_model_name])
    residuals = actuals - best_preds

    sns.set_theme(style="darkgrid")
    plot_paths = {}

    # 1. Model Comparison Chart (MAE, RMSE, WAPE)
    model_names = list(models.keys())
    maes = [models[m]["test"]["MAE"] for m in model_names]
    rmses = [models[m]["test"]["RMSE"] for m in model_names]
    wapes = [models[m]["test"]["WAPE"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(10, 5))
    rects1 = ax1.bar(x - width, maes, width, label='MAE (Tickets)', color='#4A90E2')
    rects2 = ax1.bar(x, rmses, width, label='RMSE (Tickets)', color='#E94E77')

    ax2 = ax1.twinx()
    rects3 = ax2.bar(x + width, wapes, width, label='WAPE (%)', color='#F5A623')

    ax1.set_ylabel('Absolute Error (Tickets)', color='#4A90E2', fontsize=12)
    ax2.set_ylabel('Percentage Error (%)', color='#F5A623', fontsize=12)
    ax1.set_title('Cinema Ticket Demand: Model Performance Comparison (Held-out Test Set)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=15, ha='right', fontsize=11)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.tight_layout()
    comp_plot_path = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(comp_plot_path, dpi=300)
    plt.close(fig)
    plot_paths["model_comparison"] = comp_plot_path

    # 2. Actual vs Predicted Plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actuals, best_preds, alpha=0.6, color="#00C9A7", edgecolors='k', linewidth=0.5)
    min_val = min(actuals.min(), best_preds.min())
    max_val = max(actuals.max(), best_preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Forecast Line (y=x)")
    ax.set_title(f'Actual vs Predicted Tickets Sold ({best_model_name})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Actual Tickets Sold', fontsize=11)
    ax.set_ylabel('Predicted Tickets Sold', fontsize=11)
    ax.legend()
    plt.tight_layout()
    act_pred_path = os.path.join(output_dir, "actual_vs_predicted.png")
    fig.savefig(act_pred_path, dpi=300)
    plt.close(fig)
    plot_paths["actual_vs_predicted"] = act_pred_path

    # 3. Residual Distribution & Heteroscedasticity Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(residuals, kde=True, ax=ax1, color="#845EC2", bins=25)
    ax1.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax1.set_title('Residual Distribution (Actual - Predicted)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Prediction Error (Tickets)')
    ax1.set_ylabel('Show Count')

    ax2.scatter(best_preds, residuals, alpha=0.6, color="#FF9671", edgecolors='k', linewidth=0.5)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax2.set_title('Residuals vs Fitted Values', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Predicted Tickets Sold')
    ax2.set_ylabel('Residual (Tickets)')

    plt.tight_layout()
    resid_path = os.path.join(output_dir, "residual_analysis.png")
    fig.savefig(resid_path, dpi=300)
    plt.close(fig)
    plot_paths["residual_analysis"] = resid_path

    # 4. Feature Importance Top 12 Plot
    fi_json_path = "models/feature_importance.json"
    if not os.path.exists(fi_json_path) and os.path.exists("../models/feature_importance.json"):
        fi_json_path = "../models/feature_importance.json"
    if os.path.exists(fi_json_path):
        with open(fi_json_path, "r") as f:
            fi = json.load(f)
        top_features = list(fi.items())[:12]
        feat_labels = [item[0].replace("_", " ").title() for item in top_features][::-1]
        feat_scores = [item[1] for item in top_features][::-1]

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(feat_labels, feat_scores, color="#008B8B")
        ax.set_title('Top 12 Drivers of Cinema Ticket Demand (Feature Importance)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Relative Importance Weight')
        plt.tight_layout()
        fi_plot_path = os.path.join(output_dir, "feature_importance.png")
        fig.savefig(fi_plot_path, dpi=300)
        plt.close(fig)
        plot_paths["feature_importance"] = fi_plot_path

    return plot_paths


def evaluate_cold_start_cohort(data_dir: str = "data/processed",
                               comparison_json_path: str = "models/model_comparison.json") -> Dict[str, Any]:
    """
    Quantitatively evaluate model performance on Cold-Start vs Mature movie screenings.
    Cold-start is defined as Day 0 (Premiere) and Opening Weekend (<= 3 days since release).
    """
    from src.utils import calculate_metrics

    if not os.path.exists(data_dir) and os.path.exists(os.path.join("..", data_dir)):
        data_dir = os.path.join("..", data_dir)
    if not os.path.exists(comparison_json_path) and os.path.exists(os.path.join("..", comparison_json_path)):
        comparison_json_path = os.path.join("..", comparison_json_path)

    test_csv_path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(test_csv_path) or not os.path.exists(comparison_json_path):
        return {}

    test_df = pd.read_csv(test_csv_path)
    with open(comparison_json_path, "r") as f:
        comp_data = json.load(f)

    best_model_name = comp_data.get("best_model_name", "Gradient Boosting")
    preds = np.array(comp_data["test_predictions"][best_model_name])
    actuals = test_df["tickets_sold"].values

    mask_mature = test_df["release_days"] > 3
    mask_cold_start = test_df["release_days"] <= 3
    mask_day0 = test_df["release_days"] == 0

    cohorts = {
        "overall_test": {
            "description": "All 375 held-out future screenings",
            "show_count": int(len(test_df)),
            "metrics": calculate_metrics(actuals, preds)
        },
        "mature_shows": {
            "description": "Mature screenings with established historical track record (> 3 days)",
            "show_count": int(mask_mature.sum()),
            "metrics": calculate_metrics(actuals[mask_mature], preds[mask_mature])
        },
        "cold_start_shows": {
            "description": "Opening window screenings (<= 3 days since theatrical release)",
            "show_count": int(mask_cold_start.sum()),
            "metrics": calculate_metrics(actuals[mask_cold_start], preds[mask_cold_start])
        },
        "day_0_premieres": {
            "description": "Day 0 Premieres with zero prior show history at the theater",
            "show_count": int(mask_day0.sum()),
            "metrics": calculate_metrics(actuals[mask_day0], preds[mask_day0])
        }
    }

    # Save to comparison JSON
    comp_data["cold_start_cohorts"] = cohorts
    with open(comparison_json_path, "w") as f:
        json.dump(comp_data, f, indent=2)

    return cohorts


def evaluate_cold_start_strategy() -> Dict[str, Any]:
    """
    Explain and validate cold-start scenario:
    When a movie is in Day 0 (Premiere) with no historical average sales or previous occupancy.
    """
    explanation = {
        "definition": (
            "Cold start occurs when a brand-new movie premiere is scheduled (release_days = 0) "
            "with no prior show records for this specific title at the theater."
        ),
        "solution_mechanisms": [
            "1. Genre & Language cluster anchoring: The model learns baseline demand for Action/Sci-Fi/Drama releases.",
            "2. Advance Booking Cutoff: At 6 hours prior to premiere, advance booking volume provides strong empirical intent.",
            "3. Marketing Spend signal: Big-budget releases exhibit higher initial marketing spend which lifts base demand.",
            "4. Heuristic Fallback imputation: When historical_average_sales is missing, theater slot average occupancy (e.g. 55%) is imputed.",
            "5. Theatrical Decay Curve: release_decay_factor starts at 1.0 (maximum premiere hype)."
        ]
    }
    return explanation


if __name__ == "__main__":
    paths = generate_evaluation_plots()
    print("Evaluation charts generated successfully:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    
    cohorts = evaluate_cold_start_cohort()
    print("\nCold-Start Cohort Analysis:")
    for cohort_name, data in cohorts.items():
        print(f"  [{cohort_name}] (N={data['show_count']}): WAPE={data['metrics']['WAPE']}%, MAE={data['metrics']['MAE']}")


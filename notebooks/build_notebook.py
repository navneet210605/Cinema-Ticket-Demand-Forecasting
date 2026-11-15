"""
Script to generate and execute the complete, production-grade 01_EDA_and_Modeling.ipynb Jupyter notebook.
Generates all exploratory analyses, visualizations, and modeling pipelines, capturing live outputs and charts
directly into the notebook file so it opens fully rendered in VS Code and GitHub.
"""

import os
import sys
import io
import json
import base64
import contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def build_and_execute_notebook():
    notebook_path = "notebooks/01_EDA_and_Modeling.ipynb"
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

    # Define all notebook cells
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🎬 Cinema Ticket Demand Forecasting: EDA & Model Development\n",
                "\n",
                "**Author:** Senior Machine Learning Engineer & Full-Stack Python Developer  \n",
                "**Role Suitability:** AI & Machine Learning Technical Assessment  \n",
                "**Objective:** Build an end-to-end regression machine learning pipeline to forecast cinema ticket demand for individual movie screenings, evaluate auditorium occupancy percentage, and optimize theater screen scheduling and concession staffing.\n",
                "\n",
                "### Table of Contents\n",
                "1. [Environment Setup & Reproducibility](#1)\n",
                "2. [Dataset Overview & Statistical Summary](#2)\n",
                "3. [Missing-Value & Duplicate-Value Analysis](#3)\n",
                "4. [Ticket Sales & Screen Occupancy Distribution](#4)\n",
                "5. [Demand Segmentation](#5)\n",
                "   - 5.1 Demand by Movie Genre\n",
                "   - 5.2 Demand by Weekday (Monday - Sunday)\n",
                "   - 5.3 Demand by Showtime Slot\n",
                "   - 5.4 Weekend vs Weekday Demand Lift\n",
                "   - 5.5 Holiday Demand Impact\n",
                "6. [Ticket Price Elasticity & Advance Booking Correlation](#6)\n",
                "7. [Correlation Heatmap](#7)\n",
                "8. [Outlier Analysis & Physical Bounds](#8)\n",
                "9. [Leakage-Free Feature Engineering (T-6h Cutoff)](#9)\n",
                "10. [Chronological Train/Validation/Test Partitioning](#10)\n",
                "11. [Multi-Model Training & Evaluation (MAE, RMSE, WAPE, R²)](#11)\n",
                "12. [Residual Diagnostics & Parity Analysis](#12)\n",
                "13. [Cold-Start Scenario & Interview Defense Guide](#13)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## <a id='1'></a>1. Environment Setup & Reproducibility"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "import os\n",
                "import sys\n",
                "import json\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Ensure project root is accessible\n",
                "for path in ['.', '..', os.path.abspath('..'), os.path.abspath('.')]:\n",
                "    if path not in sys.path:\n",
                "        sys.path.insert(0, path)\n",
                "\n",
                "# Set plot styling\n",
                "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                "plt.rcParams['figure.figsize'] = (10, 6)\n",
                "plt.rcParams['font.size'] = 11\n",
                "\n",
                "from src.utils import set_seed, calculate_metrics, categorize_showtime, categorize_demand\n",
                "from src.data_generation import generate_cinema_dataset\n",
                "from src.preprocessing import engineer_features, split_chronologically\n",
                "from src.train import train_and_evaluate\n",
                "\n",
                "set_seed(42)\n",
                "print(\"Environment initialized successfully. Random seed fixed to 42.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## <a id='2'></a>2. Dataset Overview & Statistical Summary"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "data_path = 'data/raw/cinema_ticket_sales.csv'\n",
                "if not os.path.exists(data_path) and os.path.exists('../' + data_path):\n",
                "    data_path = '../' + data_path\n",
                "\n",
                "df = pd.read_csv(data_path)\n",
                "print(f\"Dataset Dimensions: {df.shape[0]:,} show records x {df.shape[1]} columns\\n\")\n",
                "print(\"Columns:\", list(df.columns))\n",
                "print(\"\\n--- First 5 Shows ---\")\n",
                "print(df.head(5).to_string())"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "print(\"--- Summary Statistics of Numerical Variables ---\")\n",
                "print(df.describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']].to_string())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## <a id='3'></a>3. Missing-Value & Duplicate-Value Analysis"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "missing_series = df.isnull().sum()\n",
                "print(\"--- Missing Value Audit ---\")\n",
                "if missing_series.sum() == 0:\n",
                "    print(\"Verified: Zero missing values across all 19 columns.\")\n",
                "else:\n",
                "    print(missing_series[missing_series > 0])\n",
                "\n",
                "dup_count = df.duplicated(subset=['show_id']).sum()\n",
                "print(f\"\\nDuplicate show_id records: {dup_count} (Must be 0)\")\n",
                "\n",
                "cap_violations = (df['tickets_sold'] > df['screen_capacity']).sum()\n",
                "adv_violations = (df['advance_booking_count'] > df['tickets_sold']).sum()\n",
                "print(f\"Physical Invariant Violations (tickets_sold > screen_capacity): {cap_violations}\")\n",
                "print(f\"Pre-Sale Invariant Violations (advance_booking_count > tickets_sold): {adv_violations}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## <a id='4'></a>4. Ticket Sales & Screen Occupancy Distribution"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "df['occupancy_pct'] = np.round((df['tickets_sold'] / df['screen_capacity']) * 100.0, 1)\n",
                "df['demand_category'] = df['occupancy_pct'].apply(categorize_demand)\n",
                "\n",
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "sns.histplot(df['tickets_sold'], kde=True, ax=ax1, color='#4A90E2', bins=30)\n",
                "ax1.set_title('Distribution of Tickets Sold per Show', fontsize=13, fontweight='bold')\n",
                "ax1.set_xlabel('Tickets Sold (Count)')\n",
                "ax1.set_ylabel('Screening Frequency')\n",
                "\n",
                "sns.histplot(df['occupancy_pct'], kde=True, ax=ax2, color='#E94E77', bins=25)\n",
                "ax2.axvline(75, color='#E50914', linestyle='--', linewidth=2, label='High Demand (≥ 75%)')\n",
                "ax2.axvline(40, color='#F5A623', linestyle='--', linewidth=2, label='Medium Demand (≥ 40%)')\n",
                "ax2.set_title('Auditorium Seat Occupancy Percentage Distribution', fontsize=13, fontweight='bold')\n",
                "ax2.set_xlabel('Seat Occupancy (%)')\n",
                "ax2.set_ylabel('Screening Frequency')\n",
                "ax2.legend()\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "print(\"\\n--- Demand Category Breakdown ---\")\n",
                "print(df['demand_category'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## <a id='5'></a>5. Demand Segmentation\n",
                "\n",
                "We analyze ticket sales across genre, weekday, showtime, weekend rush, and holiday scheduling."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11))\n",
                "\n",
                "# 5.1 Demand by Genre\n",
                "genre_order = df.groupby('genre')['tickets_sold'].mean().sort_values(ascending=False).index\n",
                "sns.barplot(data=df, x='genre', y='tickets_sold', order=genre_order, ax=ax1, palette='mako')\n",
                "ax1.set_title('5.1 Mean Ticket Demand by Movie Genre', fontsize=12, fontweight='bold')\n",
                "ax1.set_xlabel('Genre')\n",
                "ax1.set_ylabel('Average Tickets Sold')\n",
                "ax1.set_xticklabels(genre_order, rotation=25)\n",
                "\n",
                "# 5.2 Demand by Weekday (Monday through Sunday)\n",
                "weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']\n",
                "sns.barplot(data=df, x='day_of_week', y='tickets_sold', order=weekday_order, ax=ax2, palette='Blues_r')\n",
                "ax2.set_title('5.2 Demand Across Day of the Week (Mon - Sun)', fontsize=12, fontweight='bold')\n",
                "ax2.set_xlabel('Day of Week')\n",
                "ax2.set_ylabel('Average Tickets Sold')\n",
                "ax2.set_xticklabels(weekday_order, rotation=25)\n",
                "\n",
                "# 5.3 Demand by Showtime Slot\n",
                "time_order = sorted(df['show_time'].unique())\n",
                "sns.lineplot(data=df, x='show_time', y='tickets_sold', marker='o', ax=ax3, color='#FF5722', lw=2.5)\n",
                "ax3.set_title('5.3 Showtime Demand Curve (10:00 to 22:30)', fontsize=12, fontweight='bold')\n",
                "ax3.set_xlabel('Showtime Slot')\n",
                "ax3.set_ylabel('Average Tickets Sold')\n",
                "\n",
                "# 5.4 Weekend vs Weekday Demand Boxplot\n",
                "sns.boxplot(data=df, x='is_weekend', y='tickets_sold', ax=ax4, palette=['#4CAF50', '#E91E63'])\n",
                "ax4.set_xticklabels(['Weekday (Mon-Thu)', 'Weekend (Fri-Sun)'])\n",
                "ax4.set_title('5.4 Weekend Demand Surge vs Weekday Base', fontsize=12, fontweight='bold')\n",
                "ax4.set_xlabel('Schedule Type')\n",
                "ax4.set_ylabel('Tickets Sold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# 5.5 Holiday Demand Impact\n",
                "fig, ax = plt.subplots(figsize=(8, 4))\n",
                "sns.barplot(data=df, x='is_holiday', y='occupancy_pct', ax=ax, palette=['#607D8B', '#FF9800'])\n",
                "ax.set_xticklabels(['Normal Day', 'Public / Festive Holiday'])\n",
                "ax.set_title('5.5 Auditorium Seat Occupancy Lift During Holidays', fontsize=12, fontweight='bold')\n",
                "ax.set_xlabel('Holiday Indicator')\n",
                "ax.set_ylabel('Average Seat Occupancy (%)')\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "holiday_occ = df.groupby('is_holiday')['occupancy_pct'].mean()\n",
                "print(f\"Normal Day Mean Occupancy: {holiday_occ[0]:.1f}%\")\n",
                "print(f\"Holiday Mean Occupancy: {holiday_occ[1]:.1f}% (+{holiday_occ[1]-holiday_occ[0]:.1f}% lift)\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## <a id='6'></a>6. Ticket Price Elasticity & Advance Booking Correlation"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "# Price vs Demand\n",
                "sns.scatterplot(data=df, x='ticket_price', y='tickets_sold', hue='genre', alpha=0.65, ax=ax1)\n",
                "ax1.set_title('Ticket Price vs Demand (Price Sensitivity)', fontsize=12, fontweight='bold')\n",
                "ax1.set_xlabel('Ticket Price ($)')\n",
                "ax1.set_ylabel('Tickets Sold')\n",
                "ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')\n",
                "\n",
                "# Advance Bookings vs Final Tickets Sold\n",
                "sns.regplot(data=df, x='advance_booking_count', y='tickets_sold', ax=ax2, color='#009688',\n",
                "            scatter_kws={'alpha': 0.35}, line_kws={'color': 'red', 'linewidth': 2})\n",
                "ax2.set_title('Advance Bookings (at Cutoff) vs Final Tickets Sold', fontsize=12, fontweight='bold')\n",
                "ax2.set_xlabel('Advance Booking Count (T-6h Cutoff)')\n",
                "ax2.set_ylabel('Final Tickets Sold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "corr_adv = df['advance_booking_count'].corr(df['tickets_sold'])\n",
                "print(f\"Pearson Correlation between Advance Bookings and Final Tickets Sold: {corr_adv:.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## <a id='7'></a>7. Correlation Heatmap"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "num_cols = df.select_dtypes(include=[np.number]).columns\n",
                "corr = df[num_cols].corr()\n",
                "\n",
                "plt.figure(figsize=(12, 9))\n",
                "sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True, linewidths=0.5)\n",
                "plt.title('Correlation Heatmap of Cinema Sales & Operational Variables', fontsize=13, fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## <a id='8'></a>8. Outlier Analysis & Physical Bounds\n",
                "\n",
                "We examine extreme values across key numeric features to ensure the dataset respects cinema operational limits."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "fig, axes = plt.subplots(1, 4, figsize=(16, 4))\n",
                "features_to_check = ['tickets_sold', 'screen_capacity', 'advance_booking_count', 'ticket_price']\n",
                "colors = ['#4A90E2', '#E94E77', '#009688', '#F5A623']\n",
                "\n",
                "for ax, col, color in zip(axes, features_to_check, colors):\n",
                "    sns.boxplot(y=df[col], ax=ax, color=color)\n",
                "    q1 = df[col].quantile(0.25)\n",
                "    q3 = df[col].quantile(0.75)\n",
                "    iqr = q3 - q1\n",
                "    lower_bound = q1 - 1.5 * iqr\n",
                "    upper_bound = q3 + 1.5 * iqr\n",
                "    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]\n",
                "    ax.set_title(f'{col}\\n(IQR Outliers: {len(outliers)})', fontsize=11, fontweight='bold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "print(\"Outlier Inspection Insight: Large ticket counts represent blockbuster sell-outs in 350-450 seat auditoriums, \")\n",
                "print(\"which are valid high-volume operational occurrences rather than data corruption. Tree-based ensembles naturally \")\n",
                "print(\"handle these non-linear high-capacity spikes without distortion.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## <a id='9'></a>9. Leakage-Free Feature Engineering (T-6h Cutoff Horizon)\n",
                "\n",
                "**Operational Forecast Horizon:** Strictly 6 hours prior to showtime ($T-6\\text{h}$).  \n",
                "At this moment, advance bookings up to the cutoff, scheduled screen capacity, base ticket price, and competitor shows are known. Walk-in sales over the final 6 hours are unknown.\n",
                "\n",
                "We engineer:\n",
                "- `month`, `day`: Calendar month and day\n",
                "- `showtime_category`: Morning, Afternoon, Evening, Night\n",
                "- `advance_booking_ratio`: `advance_booking_count / screen_capacity`\n",
                "- `price_per_minute`: Perceived entertainment cost rate\n",
                "- `marketing_spend_log`: Diminishing returns representation\n",
                "- `competitor_density`: Rival shows normalized to auditorium scale\n",
                "- `hist_sales_ratio`: Historical sales normalized by screen size\n",
                "- `release_decay_factor`: $\\exp(-0.045 \\times \\text{release\\_days})$"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "df_engineered = engineer_features(df)\n",
                "print(\"Engineered Features Shape:\", df_engineered.shape)\n",
                "print(\"Sample engineered variables:\")\n",
                "print(df_engineered[['show_date', 'show_time', 'showtime_category', 'advance_booking_ratio', 'price_per_minute', 'release_decay_factor']].head(4).to_string())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## <a id='10'></a>10. Chronological Train/Validation/Test Partitioning\n",
                "\n",
                "> **Temporal Split Rule:** We strictly partition chronologically without random shuffling to prevent future information leakage:\n",
                "- **Train Set (70%):** Jan 01, 2024 – Jun 11, 2024\n",
                "- **Validation Set (15%):** Jun 11, 2024 – Jul 16, 2024\n",
                "- **Held-Out Test Set (15%):** Jul 16, 2024 – Aug 18, 2024"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "train_df, val_df, test_df = split_chronologically(df_engineered, 0.70, 0.15)\n",
                "\n",
                "print(f\"Train Set: {len(train_df):,} shows ({train_df['show_date'].min()} to {train_df['show_date'].max()})\")\n",
                "print(f\"Val Set:   {len(val_df):,} shows ({val_df['show_date'].min()} to {val_df['show_date'].max()})\")\n",
                "print(f\"Test Set:  {len(test_df):,} shows ({test_df['show_date'].min()} to {test_df['show_date'].max()})\")\n",
                "\n",
                "assert train_df['show_date'].max() <= val_df['show_date'].min()\n",
                "assert val_df['show_date'].max() <= test_df['show_date'].min()\n",
                "print(\"Chronological temporal ordering validated: Zero lookahead leakage.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## <a id='11'></a>11. Multi-Model Training & Leaderboard Comparison\n",
                "\n",
                "We train and benchmark 4 candidate algorithms:\n",
                "1. **Capacity-Adjusted Historical Baseline** (Heuristic benchmark)\n",
                "2. **Linear Regression Pipeline** (OLS with L2 regularization)\n",
                "3. **Random Forest Regressor** (180 bagged decision trees, max depth 12)\n",
                "4. **Gradient Boosting Regressor** (160 boosted trees, lr 0.07, subsample 0.85)\n",
                "\n",
                "Evaluated using **MAE**, **RMSE**, and **WAPE** (Weighted Absolute Percentage Error):"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "comp_results = train_and_evaluate()\n",
                "\n",
                "rows = []\n",
                "for m_name, metrics in comp_results['models'].items():\n",
                "    tm = metrics['test']\n",
                "    vm = metrics['validation']\n",
                "    rows.append({\n",
                "        'Algorithm': m_name,\n",
                "        'Test MAE (Tickets)': tm['MAE'],\n",
                "        'Test RMSE': tm['RMSE'],\n",
                "        'Test WAPE (%)': f\"{tm['WAPE']}%\",\n",
                "        'Test R²': tm['R2'],\n",
                "        'Val WAPE (%)': f\"{vm['WAPE']}%\"\n",
                "    })\n",
                "\n",
                "leaderboard = pd.DataFrame(rows).sort_values(by='Test MAE (Tickets)')\n",
                "print(f\"\\n🏆 Champion Algorithm: {comp_results['best_model_name']}\\n\")\n",
                "print(leaderboard.to_string(index=False))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## <a id='12'></a>12. Residual Diagnostics & Parity Analysis"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "best_model_name = comp_results['best_model_name']\n",
                "y_true = np.array(comp_results['test_actuals'])\n",
                "y_pred = np.array(comp_results['test_predictions'][best_model_name])\n",
                "residuals = y_true - y_pred\n",
                "\n",
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "# Parity Plot\n",
                "ax1.scatter(y_true, y_pred, alpha=0.6, color='#00C9A7', edgecolors='k', linewidth=0.5)\n",
                "min_v, max_v = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())\n",
                "ax1.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Perfect Forecast (y = x)')\n",
                "ax1.set_title(f'Actual vs Predicted Tickets ({best_model_name})', fontsize=12, fontweight='bold')\n",
                "ax1.set_xlabel('Actual Tickets Sold')\n",
                "ax1.set_ylabel('Predicted Tickets Sold')\n",
                "ax1.legend()\n",
                "\n",
                "# Residual Distribution\n",
                "sns.histplot(residuals, kde=True, ax=ax2, color='#845EC2', bins=25)\n",
                "ax2.axvline(0, color='red', linestyle='--', linewidth=1.5)\n",
                "ax2.set_title('Residual Distribution (Actual - Predicted)', fontsize=12, fontweight='bold')\n",
                "ax2.set_xlabel('Prediction Error (Tickets)')\n",
                "ax2.set_ylabel('Show Count')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "print(f\"Residual Summary:\")\n",
                "print(f\"  Mean Error: {np.mean(residuals):.2f} tickets (near zero indicates unbiased estimator)\")\n",
                "print(f\"  Std Error:  {np.std(residuals):.2f} tickets\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## <a id='13'></a>13. Cold-Start Strategy & Interview Defense Guide\n",
                "\n",
                "### Cold-Start Architecture for Movie Premieres\n",
                "When a brand-new title premieres (`release_days = 0`) with zero prior historical sales:\n",
                "1. **Genre & Language Anchoring:** The pipeline leverages learned baseline coefficients for Action, Sci-Fi, and Drama.\n",
                "2. **T-6h Advance Sales Anchor:** Pre-sale tickets booked prior to the 6-hour cutoff provide empirical intent even for new releases.\n",
                "3. **Dynamic Median Imputation:** Missing `historical_average_sales` is automatically imputed as $55\\%$ of screen capacity (the theater's operational baseline).\n",
                "4. **Decay Curve Initialization:** $\\text{release\\_decay\\_factor}$ starts at $1.0$ (peak premiere hype).\n",
                "\n",
                "### Interview Defense Summary\n",
                "- **Why WAPE over MAPE?** MAPE divides by individual show sales ($y$), causing division by zero or extreme distortion on low-attendance shows (e.g. 5 tickets). WAPE normalizes total absolute errors by aggregate ticket volume, measuring actual business revenue and seat capacity error.\n",
                "- **Why Chronological Split?** Entertainment box office exhibits natural temporal momentum and decaying trends. Random cross-validation leaks future information into past predictions.\n",
                "- **How are physical bounds maintained?** Post-processing clamps all forecasts: $\\text{advance\\_booking\\_count} \\le \\hat{y} \\le \\text{screen\\_capacity}$."
            ]
        }
    ]

    print("Executing notebook cells to capture real outputs and visualizations...")
    exec_globals = {}
    
    execution_counter = 1
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            cell_code = "".join(cell["source"])
            stdout_capture = io.StringIO()
            plt.close("all")
            
            outputs = []
            try:
                with contextlib.redirect_stdout(stdout_capture):
                    exec(cell_code, exec_globals)
                
                # Capture standard text output
                text_output = stdout_capture.getvalue()
                if text_output:
                    outputs.append({
                        "name": "stdout",
                        "output_type": "stream",
                        "text": text_output.splitlines(keepends=True)
                    })
                
                # Capture open matplotlib figures
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, format="png", dpi=120, bbox_inches="tight")
                    img_buf.seek(0)
                    img_b64 = base64.b64encode(img_buf.read()).decode("utf-8")
                    outputs.append({
                        "data": {
                            "image/png": img_b64,
                            "text/plain": [f"<Figure size {fig.get_size_inches()} with {len(fig.axes)} Axes>"]
                        },
                        "metadata": {},
                        "output_type": "display_data"
                    })
                plt.close("all")

            except Exception as e:
                print(f"Error executing cell {i}: {e}")
                outputs.append({
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "output_type": "error",
                    "traceback": [f"{type(e).__name__}: {str(e)}"]
                })

            cell["execution_count"] = execution_counter
            cell["outputs"] = outputs
            execution_counter += 1

    nb_json = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13.1"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb_json, f, indent=2)

    print(f"Jupyter Notebook successfully built and executed at: {notebook_path}")
    print(f"Total cells: {len(cells)} ({execution_counter - 1} executed code cells with embedded plots).")


if __name__ == "__main__":
    build_and_execute_notebook()

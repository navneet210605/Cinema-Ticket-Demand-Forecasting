# 🎬 Cinema Ticket Demand Forecasting

An end-to-end, industry-grade Machine Learning system that forecasts cinema ticket demand and auditorium seat occupancy for individual movie screenings. Built with **Python, Scikit-Learn, Streamlit, and Plotly**.

---

## 🌟 Key Features

- **Empirical Regression Modeling:** Trains and compares 4 algorithms:
  - Capacity-Adjusted Historical Baseline
  - Linear Regression Pipeline
  - Random Forest Regressor
  - **Gradient Boosting Regressor (Champion: 8.87% WAPE, 11.06 MAE)**
- **Leakage-Free Architecture:** Strictly enforces a **T-6 Hour Forecast Cutoff** and **chronological train/val/test splits** (no random shuffling or temporal lookahead).
- **Physical & Pre-Sale Invariants:** Enforces physical bounds: $\text{advance\_bookings} \le \hat{y} \le \text{screen\_capacity}$.
- **Cold-Start Resilience:** Gracefully handles brand-new movie premieres with zero historical sales via genre anchoring and dynamic imputation.
- **Cinema Dark UI Dashboard:** Interactive Streamlit web app with 4 pages:
  - 🎬 **Executive Dashboard:** High-level KPIs, historical attendance trajectories, and capacity utilization.
  - 🔮 **Demand Prediction Studio:** Live inference form with immediate occupancy gauges and managerial action items (staffing, screen sizing, dynamic pricing).
  - 📊 **Deep Analytics & Trends:** Interactive Plotly charts exploring genre elasticity, showtime curves, and price sensitivity.
  - 🧠 **Model Performance & Explainability:** Test parity plots, residual histograms, feature importances, and interview defense guides.
- **Production Codebase:** 100% automated test coverage with Pytest, serialized Joblib pipelines, and modular separation of concerns.

---

## 📁 Project Directory Structure

```
Cinema Ticket Demand Forecasting/
├── .gitignore                      # Git exclusion rules
├── README.md                       # Comprehensive setup and project guide
├── requirements.txt                # Pinned production dependencies
├── data/
│   ├── raw/
│   │   └── cinema_ticket_sales.csv # 2,500 synthetic show records (reproducible seed 42)
│   └── processed/
│       ├── train.csv               # 70% Chronological train split (Jan 01 - Jun 11)
│       ├── val.csv                 # 15% Chronological validation split (Jun 11 - Jul 16)
│       └── test.csv                # 15% Chronological held-out test split (Jul 16 - Aug 18)
├── src/
│   ├── __init__.py
│   ├── data_generation.py          # Synthetic dataset simulation engine
│   ├── preprocessing.py            # Feature engineering transformer & chronological splitter
│   ├── train.py                    # Multi-model training and Joblib serialization
│   ├── evaluate.py                 # Diagnostic chart generation & cold-start evaluation
│   ├── predict.py                  # Production inference class with boundary clamping
│   └── utils.py                    # Metric calculators (MAE, RMSE, WAPE) & classifiers
├── models/
│   ├── best_model.joblib           # Serialized Gradient Boosting pipeline
│   ├── model_comparison.json       # Metric evaluation JSON
│   ├── feature_importance.json     # Feature weight attribution JSON
│   └── plots/                      # Static high-res diagnostic plots
│       ├── model_comparison.png
│       ├── actual_vs_predicted.png
│       ├── residual_analysis.png
│       └── feature_importance.png
├── notebooks/
│   ├── 01_EDA_and_Modeling.ipynb   # Interactive EDA & Modeling Jupyter Notebook
│   └── build_notebook.py           # Programmatic notebook generator
├── app/
│   ├── app.py                      # Multi-page Streamlit dashboard entrypoint
│   ├── styles.py                   # Cinema Velvet dark theme CSS & Plotly layouts
│   └── components.py               # Reusable metric cards and HTML badges
├── tests/
│   ├── __init__.py
│   ├── test_data_generation.py     # Data integrity & capacity constraint tests
│   ├── test_preprocessing.py       # Feature transformation & split ordering tests
│   ├── test_model.py               # Metric calculation & threshold tests
│   └── test_inference.py           # Prediction bounds, cold-start & batch tests
└── reports/
    ├── architecture_diagram.md     # Mermaid pipeline diagrams and workflow
    ├── dataset_documentation.md    # Complete 19-column data dictionary & simulation physics
    ├── model_evaluation_report.md  # Detailed leaderboard, WAPE derivation, & diagnostics
    └── technical_report.md         # Comprehensive report & interview defense guide
```

---

## 🏆 Model Leaderboard (Held-Out Chronological Test Set)

| Rank | Model Candidate | Test MAE (Tickets) | Test RMSE (Tickets) | Test WAPE (%) | Test $R^2$ |
|:---:|:---|:---:|:---:|:---:|:---:|
| 🥇 | **Gradient Boosting Regressor** | **11.06** | **15.20** | **8.87%** | **0.9476** |
| 🥈 | **Linear Regression (Pipeline)** | 11.58 | 15.53 | 9.29% | 0.9453 |
| 🥉 | **Random Forest Regressor** | 13.09 | 18.03 | 10.50% | 0.9263 |
| 4 | **Capacity-Adjusted Baseline** | 29.21 | 38.74 | 23.43% | 0.6598 |

---

## 🚀 Setup & Execution Instructions

### 1. Prerequisites
Ensure you have **Python 3.10+** installed.

### 2. Clone / Open Directory & Install Dependencies
```bash
# Navigate to project root
cd "Cinema Ticket Demand Forecasting"

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate Dataset (Optional: Pre-generated file included)
```bash
python -m src.data_generation
```

### 4. Run Preprocessing & Chronological Splitting
```bash
python -m src.preprocessing
```

### 5. Train & Evaluate Models
```bash
python -m src.train
python -m src.evaluate
```

### 6. Run Automated Pytest Suite
```bash
python -m pytest tests/
```
All 13 unit tests will execute and validate data constraints, preprocessing logic, model accuracy, and cold-start handling.

### 7. Launch Interactive Streamlit Application
```bash
streamlit run app/app.py
```
Open your browser to `http://localhost:8501` to access the live dashboard!

---

## 🎓 Technical Interview Defense Checklist

1. **Why WAPE over MAPE?**  
   MAPE divides by actual show sales $y$, inflating to infinity on low-attendance shows (e.g. 5 tickets). **WAPE** normalizes by total ticket volume sold ($\sum |y - \hat{y}| / \sum y$), directly measuring overall cinema revenue and seat forecasting accuracy.
2. **Why Chronological Split over $K$-Fold?**  
   Time-series data has natural temporal ordering and hype decay. Random cross-validation leaks future knowledge into the past. Chronological splitting simulates real-world deployment.
3. **How is Cold Start solved?**  
   New movies on premiere night use genre/language baseline weights, the $T-6\text{h}$ advance booking tally, and median occupancy fallbacks.
4. **How are physical constraints maintained?**  
   A post-processing layer clamps all predictions between $\text{advance\_booking\_count}$ and $\text{screen\_capacity}$.

---

## 📜 License
MIT License. Built for technical assessments and educational AI demonstrations.

# 🎬 CineForecast: Cinema Ticket Demand Forecasting

> **Technical Assessment Project for ALGIONS AI LABS PRIVATE LIMITED**  
> **Role:** AI & Machine Learning Engineer  
> **Official Assessment Reference:** `05_Cinema_Ticket_Demand_Forecasting.docx`  
> **Live Deployed Web Application:** [https://splendorous-meerkat-ef61cf.netlify.app/](https://splendorous-meerkat-ef61cf.netlify.app/)  

---

## 🌟 Executive Overview

**CineForecast** is an industry-grade, end-to-end Machine Learning system that predicts ticket demand and auditorium seat occupancy for individual cinema screenings. The system forecasts show attendance strictly **6 hours prior to showtime ($T-6\text{h}$)**, enabling cinema operators to:
1. **Optimize Staffing:** Schedule ushers, box office cashiers, and concession counter attendants based on forecasted foot traffic.
2. **Dynamic Screen Allocation:** Reallocate trending blockbusters to 350+ seat premium auditoriums while shifting underperforming shows to smaller 120-seat screens to save HVAC and projection costs.
3. **Revenue Management:** Apply dynamic surge pricing on high-demand slots (+10-15%) and trigger combo discounts on low-demand matinees to stimulate walk-in sales.

---

## 🏆 Model Leaderboard (Held-Out Chronological Test Set: 375 Shows)

Models were evaluated on an unseen chronological test set (July 16 – August 18, 2024) with strict leakage prevention:

| Rank | Model Candidate | Test MAE (Tickets) | Test RMSE (Tickets) | Test WAPE (%) | Test $R^2$ | Validation WAPE (%) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Gradient Boosting Regressor** | **11.06** | **15.20** | **8.87%** | **0.9476** | **8.13%** |
| 🥈 | Linear Regression (Pipeline) | 11.58 | 15.53 | 9.29% | 0.9453 | 7.01% |
| 🥉 | Random Forest Regressor | 13.09 | 18.03 | 10.50% | 0.9263 | 9.97% |
| 4 | Capacity-Adjusted Baseline | 29.21 | 38.74 | 23.43% | 0.6598 | 23.33% |

*The champion **Gradient Boosting Regressor** delivers a **62.1% error reduction** over the naive Capacity Baseline, predicting attendance within $\approx 11$ tickets across auditoriums up to 450 seats.*

---

## ❄️ Cold-Start Premiere Evaluation

To evaluate performance on newly released films without historical track records:

| Cohort | Definition | Shows ($N$) | MAE (Tickets) | RMSE | WAPE (%) | $R^2$ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Overall Test Set** | All held-out future screenings | 375 | 11.06 | 15.20 | 8.87% | 0.9476 |
| **Mature Shows** | Established movies ($\text{release\_days} > 3$) | 349 | 11.20 | 15.28 | 9.17% | 0.9458 |
| **Cold-Start Shows** | Opening window ($\text{release\_days} \le 3$) | 26 | 9.18 | 14.14 | **5.77%** | 0.9562 |
| **Day 0 Premieres** | Day 0 screenings with zero prior history | 7 | 12.40 | 18.38 | **6.89%** | 0.9072 |

---

## 🛠️ System Architecture & Dual-Mode Deployment

```
                    ┌──────────────────────────────────────────────┐
                    │            CineForecast Frontends            │
                    ├──────────────────────┬───────────────────────┤
                    │   Streamlit Web UI   │  Netlify Web Portal   │
                    │  (Port 8501: 6 Pages)│  (Vanilla JS / HTML5) │
                    └──────────┬───────────┴───────────┬───────────┘
                               │                       │
                HTTP/Internal  │                       │ REST API (Port 8000)
                               ▼                       ▼
                    ┌──────────────────────────────────────────────┐
                    │         CineForecast REST API (FastAPI)      │
                    │         src/api.py (Starlette + Uvicorn)     │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │      Joblib Serialized Pipeline Engine       │
                    │   - Imputation & One-Hot Encoding Transformer│
                    │   - Gradient Boosting Regressor (160 trees)  │
                    │   - Physical Bounds Clamping Layer           │
                    │   - 95% Confidence Interval Estimator        │
                    └──────────────────────────────────────────────┘
```

1. **Production REST API (`src/api.py`):** High-performance ASGI server built with Starlette/FastAPI core and Uvicorn. Serves predictions, metrics, and dataset previews via `/api/predict`, `/api/metrics`, and `/health`.
2. **Streamlit Multi-Page App (`app/app.py`):** 6-page interactive executive suite with Plotly charts, 7 KPI cards, live prediction studio, model explainability, and dataset explorer.
3. **Netlify Web Portal (`web/index.html` / `index.html`):** Deployed on Netlify with **Dual-Mode Inference**—connects dynamically to the local REST API when running, and provides a mathematically calibrated offline fallback engine when accessed purely statically.

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup
```powershell
# Navigate to project directory
cd "Cinema Ticket Demand Forecasting"

# Install production dependencies
pip install -r requirements.txt
```

### 2. Run Automated Pytest Test Suite
```powershell
python -m pytest tests/ -v
```
*Validates 19 unit tests across 10 categories (bounds clamping, chronological ordering, cold start, imputer, etc.) with 100% pass rate.*

### 3. Launch the REST API Backend
```powershell
python -m src.api
```
*Available at:* `http://127.0.0.1:8000`  
*Health Check:* `http://127.0.0.1:8000/health`

### 4. Launch the Streamlit Multi-Page Dashboard
```powershell
streamlit run app/app.py
```
*Or double-click `run_app.bat` on Windows. Access at `http://localhost:8501`.*

### 5. Open the Netlify Web Portal
Open `web/index.html` or visit the live deployment at [https://splendorous-meerkat-ef61cf.netlify.app/](https://splendorous-meerkat-ef61cf.netlify.app/).

---

## 📁 Repository Structure

```
Cinema Ticket Demand Forecasting/
├── index.html                      # Root web entrypoint (sync with web/index.html)
├── netlify.toml                    # Netlify publish & header configuration
├── README.md                       # Master documentation & setup guide
├── requirements.txt                # Pinned production dependencies
├── run_app.bat                     # Windows one-click Streamlit launcher
├── app/
│   ├── app.py                      # Multi-page Streamlit dashboard (6 pages)
│   ├── components.py               # Dynamic KPI cards and HTML badges
│   └── styles.py                   # Cinema Velvet theme styling & Plotly layouts
├── data/
│   ├── raw/cinema_ticket_sales.csv # 2,500 synthetic show records (seed=42)
│   └── processed/                  # Chronological splits (train 70%, val 15%, test 15%)
├── models/
│   ├── best_model.joblib           # Champion Gradient Boosting pipeline
│   ├── model_comparison.json       # Metrics and cold-start cohort JSON
│   ├── feature_importance.json     # Feature weight rankings
│   └── plots/                      # Publication-quality diagnostic plots
├── notebooks/
│   ├── 01_EDA_and_Modeling.ipynb   # 28-cell pre-rendered EDA & Modeling notebook
│   └── build_notebook.py           # Programmatic notebook builder script
├── reports/
│   ├── architecture_diagram.md     # Pipeline and workflow diagrams
│   ├── dataset_documentation.md    # 19-column data dictionary & simulation physics
│   ├── model_evaluation_report.md  # Detailed leaderboard & WAPE derivation
│   └── technical_report.md         # Full assessment report & interview defense guide
├── src/
│   ├── api.py                      # ASGI REST API backend (Starlette/FastAPI)
│   ├── data_generation.py          # Synthetic dataset generator (seed=42)
│   ├── evaluate.py                 # Diagnostic plotting & cold-start cohort evaluation
│   ├── predict.py                  # Production inference & confidence intervals
│   ├── preprocessing.py            # Feature engineering & chronological split
│   ├── train.py                    # Multi-model training and Joblib serialization
│   └── utils.py                    # WAPE, MAE, RMSE metrics & classifiers
├── tests/                          # 19 automated unit tests across 10 categories
└── web/
    └── index.html                  # Netlify production web portal (Dual-Mode)
```

---

## 🎓 Technical Interview Defense Checklist

1. **Why WAPE instead of MAPE?**  
   MAPE divides by individual actuals ($y$), creating division-by-zero or infinite distortion on low-attendance shows (e.g., 4 tickets sold). **WAPE** divides total absolute error by total actual volume ($\frac{\sum |y - \hat{y}|}{\sum y}$), directly measuring total cinema turnover and seat capacity utilization.
2. **Why Chronological Splitting?**  
   Cinema ticket demand is non-stationary; box-office decay and seasonal trends evolve over time. Random cross-validation introduces lookahead leakage (learning from the future to predict the past). Chronological splitting evaluates true out-of-sample forecasting.
3. **How does Cold-Start prediction work?**  
   When a newly released film has zero historical sales (`release_days = 0`), the pipeline anchors on genre/runtime cluster coefficients, uses the cumulative advance booking volume at $T-6\text{h}$, and imputes missing history with dynamic median slot occupancy (55%).
4. **How are impossible predictions prevented?**  
   A dedicated post-processing layer clamps all predictions between $\text{advance\_booking\_count}$ and $\text{screen\_capacity}$, ensuring no model predicts unsold pre-orders or attendance exceeding auditorium seats.

---

## 📜 Compliance & License
Developed for **ALGIONS AI LABS PRIVATE LIMITED** (Role: AI & Machine Learning assessment).  
Code released under the MIT License.

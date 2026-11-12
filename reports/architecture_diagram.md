# 🏗️ Cinema Ticket Demand Forecasting: System Architecture

This document outlines the complete architectural design of the Cinema Ticket Demand Forecasting system, from raw show scheduling and feature engineering to model training, evaluation, and interactive Streamlit deployment.

---

## 1. End-to-End System Pipeline

```mermaid
flowchart TD
    subgraph Data_Layer ["1. Data Ingestion & Generation Layer"]
        A["Movie Scheduling & Catalog<br/>(18 Titles, 8 Genres, Languages)"] --> B["Synthetic Simulation Engine<br/>(Seed: 42, 2,500 Show Records)"]
        B --> C["Raw Dataset Storage<br/>(data/raw/cinema_ticket_sales.csv)"]
    end

    subgraph Feature_Engineering ["2. Leakage-Free Preprocessing Layer (T-6h Cutoff)"]
        C --> D["Feature Engineering Transformer<br/>• Month & Day Extraction<br/>• Showtime Categorization<br/>• Advance Booking Ratio<br/>• Price per Minute & Marketing Log<br/>• Release Decay Factor"]
        D --> E["Chronological Splitter<br/>• Train (70%): Jan 01 - Jun 11<br/>• Val (15%): Jun 11 - Jul 16<br/>• Test (15%): Jul 16 - Aug 18"]
    end

    subgraph Modeling_Layer ["3. Machine Learning Pipeline Layer"]
        E --> F["Scikit-Learn ColumnTransformer<br/>• Numerical: SimpleImputer + StandardScaler<br/>• Categorical: SimpleImputer + OneHotEncoder"]
        F --> G1["Capacity Baseline<br/>(Empirical Slot Heuristic)"]
        F --> G2["Linear Regression<br/>(L2 Regularized OLS)"]
        F --> G3["Random Forest Regressor<br/>(180 Trees, Max Depth 12)"]
        F --> G4["Gradient Boosting Regressor<br/>(Champion Model, 160 Estimators)"]
        G1 & G2 & G3 & G4 --> H["Model Evaluator<br/>(MAE, RMSE, WAPE, R²)"]
        H --> I["Model Serialization (Joblib)<br/>models/best_model.joblib"]
    end

    subgraph Inference_Layer ["4. Inference & Business Logic Layer"]
        I --> J["CinemaDemandPredictor Engine"]
        K["Cold-Start Handler & Input Imputer"] --> J
        J --> L["Physical Boundary Enforcement<br/>advance_booking ≤ y_pred ≤ screen_capacity"]
        L --> M["Decision Support Recommendations<br/>• Staffing (Ushers, Concessions)<br/>• Screen Capacity Action<br/>• Dynamic Pricing Guidance"]
    end

    subgraph UI_Layer ["5. Streamlit Presentation Layer"]
        M --> N["Executive Dashboard (KPIs & Trends)"]
        M --> O["Demand Prediction Studio (Live Inference)"]
        M --> P["Deep Analytics & Trends (Plotly Charts)"]
        M --> Q["Model Performance & Explainability"]
    end
```

---

## 2. Forecast Horizon & Temporal Integrity

A critical failure mode in demand forecasting is **target leakage** (using features that would not exist when the forecast is made). In our system, the forecast cutoff is strictly defined as **T-6 Hours**:

```mermaid
sequenceDiagram
    autonumber
    actor Admin as Cinema Operations Team
    participant System as Cine-Forecast Engine
    participant Model as Gradient Boosting Pipeline
    
    Note over Admin,Model: T-72h to T-6h: Advance Booking Phase
    Admin->>System: Advance bookings logged up to cutoff window
    Note over Admin,Model: T-6h: Forecast Cutoff Point
    Admin->>System: Submit show parameters & pre-sale tally
    System->>System: Impute cold-start values if movie is premiere
    System->>System: Apply non-leaking feature transformations
    System->>Model: Execute pipeline.predict(X)
    Model-->>System: Raw ticket sales estimate
    System->>System: Clamp: max(advance_bookings, min(screen_capacity, pred))
    System-->>Admin: Final Forecast, Expected Occupancy %, Staffing & Screen Recommendations
    Note over Admin,Model: T-6h to Showtime: Walk-in ticket sales occur up to capacity
```

---

## 3. Directory Layout & Module Responsibilities

| Path | Purpose |
|---|---|
| `data/raw/` | Holds the immutable raw cinema show records. |
| `data/processed/` | Holds strictly chronologically partitioned train, validation, and test sets. |
| `src/data_generation.py` | Reproducible synthetic cinema data generator enforcing business constraints. |
| `src/preprocessing.py` | Feature engineering transformers and temporal split logic. |
| `src/train.py` | Training runner for baseline and Scikit-Learn regressors with artifact export. |
| `src/evaluate.py` | Diagnostic chart generation (residuals, actual vs predicted, feature weights). |
| `src/predict.py` | Production inference class with cold-start fallbacks and operational guardrails. |
| `src/utils.py` | Core calculation metrics (WAPE, MAE, RMSE) and categorizers. |
| `app/app.py` | Streamlit dark-mode multi-page user application. |
| `app/styles.py` | Cinema velvet dark CSS design and Plotly templates. |
| `app/components.py` | Metric cards and UI visual badges. |
| `models/` | Serialized model (`best_model.joblib`), metrics JSON, and diagnostic plots. |
| `tests/` | Pytest automated test suite covering generation, transformers, models, and inference. |
| `notebooks/` | Interactive Jupyter Notebook for exploratory data analysis and visual validation. |

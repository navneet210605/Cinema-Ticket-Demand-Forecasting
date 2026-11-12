# 📘 Cinema Ticket Demand Forecasting: Technical Project Report

**Author:** Senior Machine Learning Engineer & Full-Stack Python Developer  
**Project:** Cinema Ticket Demand Forecasting  
**Role Suitability:** AI & Machine Learning Internship / Junior-to-Mid ML Engineer Assessment  

---

## 1. Executive Summary

Cinema management faces a daily operational dilemma: how many tickets will each scheduled screening sell? Accurate demand forecasting enables cinemas to:
1. **Optimize Staffing:** Schedule the right number of ushers, box office cashiers, and concession counter attendants.
2. **Dynamic Screen Allocation:** Shift high-demand blockbusters to 400+ seat premium auditoriums (IMAX/Dolby Cinema) while downsizing slow performers to 120-seat screens to save HVAC and projection costs.
3. **Revenue Management:** Implement targeted dynamic pricing, combo discounts, or promotional pushes for shows at risk of low occupancy.

This project delivers an end-to-end, production-grade Machine Learning solution. Using a strictly chronological training, validation, and testing split, we developed and compared a heuristic Capacity-Adjusted Baseline against Linear Regression, Random Forest, and Gradient Boosting Regressor. The champion **Gradient Boosting Regressor** achieved a **WAPE of 8.87%** and a **Test MAE of 11.06 tickets**, outperforming the baseline by **62.1%**. The pipeline is serialized via Joblib and integrated into an interactive, dark-themed Streamlit application.

---

## 2. Business & Technical Formulation

### 2.1 The Problem as Regression
The task is formulated as a supervised regression problem:
$$\hat{y} = f(X)$$
Where:
- $y \in \mathbb{Z}_{\ge 0}$: Final verified count of tickets sold for a scheduled screening.
- $X \in \mathbb{R}^d$: Feature vector representing movie attributes, schedule timing, calendar indicators, pricing, pre-sale volume, and competitor pressure.

### 2.2 Domain-Specific Invariants & Physical Constraints
In the real world, machine learning models can output impossible numbers (e.g. negative sales, or predicting 350 tickets in a 200-seat theater). Our pipeline enforces two non-negotiable physical bounds:
1. **Advance Booking Floor:**
   $$\hat{y}_{\text{clamped}} \ge \text{advance\_booking\_count}$$
   Tickets already purchased before the forecast cutoff cannot be unsold.
2. **Screen Capacity Ceiling:**
   $$\hat{y}_{\text{clamped}} \le \text{screen\_capacity}$$
   A theater cannot physically seat more people than auditorium capacity.

---

## 3. Forecast Horizon & Leakage Prevention

> [!IMPORTANT]
> **Definition of Prediction Cutoff:** The forecast is generated strictly **6 hours prior to showtime** ($T-6\text{h}$).

### Why $T-6\text{h}$?
- **Operational Feasibility:** 6 hours provides management sufficient lead time to adjust staffing shifts, prepare concession inventory (popcorn batch prep), or adjust digital marquee signage.
- **Predictive Signal:** At $T-6\text{h}$, advance ticket sales have largely accumulated (representing 45%–80% of final sales), while walk-in traffic over the final 6 hours remains the primary variance to be estimated.
- **Leakage Prevention:** Any post-show actuals, concession revenue, or post-cutoff sales are strictly barred from the feature set.

---

## 4. Chronological Data Partitioning

A common mistake in ML projects is using random $k$-fold cross-validation on time-dependent data. Random splits introduce **temporal lookahead bias** (training on future shows to predict past shows).

In our system:
- **Total Dataset:** 2,500 shows sorted chronologically by `show_date` and `show_time`.
- **Training Set (70%):** 1,750 shows (January 1, 2024 to June 11, 2024).
- **Validation Set (15%):** 375 shows (June 11, 2024 to July 16, 2024).
- **Held-Out Test Set (15%):** 375 shows (July 16, 2024 to August 18, 2024).

Models are fitted strictly on the training set, hyperparameter-validated on the validation set, and final business performance is verified only on the unseen future test set.

---

## 5. Model Comparison & Experimental Results

### 5.1 Evaluated Algorithms
1. **Capacity-Adjusted Historical Baseline:** An empirical heuristic combining historical slot average sales clipped to auditorium capacity and advance bookings.
2. **Linear Regression (Pipeline):** Scikit-Learn standard scaled and one-hot encoded regularized linear model.
3. **Random Forest Regressor:** An ensemble of 180 bagged decision trees with maximum depth of 12.
4. **Gradient Boosting Regressor (Champion):** 160 sequentially boosted trees with learning rate $\eta = 0.07$ and subsampling $0.85$.

### 5.2 Held-Out Test Metrics

| Model Candidate | Test MAE | Test RMSE | Test WAPE (%) | Test $R^2$ |
|---|:---:|:---:|:---:|:---:|
| **Gradient Boosting** | **11.06** | **15.20** | **8.87%** | **0.9476** |
| Linear Regression | 11.58 | 15.53 | 9.29% | 0.9453 |
| Random Forest | 13.09 | 18.03 | 10.50% | 0.9263 |
| Capacity Baseline | 29.21 | 38.74 | 23.43% | 0.6598 |

---

## 6. Technical Interview Defense Guide

Be prepared to answer these technical interview questions:

### Q1: Why did you choose WAPE instead of MAPE?
**Answer:**  
In demand forecasting, MAPE is calculated as:
$$\text{MAPE} = \frac{1}{n}\sum \left|\frac{y - \hat{y}}{y}\right| \times 100\%$$
If a morning show sells only 5 tickets and our model predicts 10, the absolute error is just 5 tickets, but MAPE is $100\%$. MAPE divides by individual actuals, which severely penalizes low-volume shows and produces infinite errors if $y=0$. **WAPE (Weighted Absolute Percentage Error)** divides total absolute errors by total actual sales:
$$\text{WAPE} = \frac{\sum |y - \hat{y}|}{\sum y}$$
This aligns directly with overall cinema business turnover and seat utilization.

### Q2: How does your system handle the "Cold-Start" problem for new movies?
**Answer:**  
When a brand-new movie premieres (`release_days = 0`), there is no historical average sales data for that movie. Our system resolves cold-start through four mechanisms:
1. **Categorical Embeddings / Encoding:** The model leverages the historical performance of the film's genre, language, and runtime duration.
2. **Advance Booking Signal:** At $T-6\text{h}$, advance ticket sales are already logged, providing empirical demand confirmation.
3. **Dynamic Median Imputation:** If `historical_average_sales` is missing, the system imputes 55% of the screen capacity (the cinema's empirical baseline occupancy).
4. **Decay Curve Initialization:** The exponential decay feature is initialized at maximum value ($1.0$).

### Q3: Why didn't you use standard $K$-Fold Cross Validation?
**Answer:**  
Cinema ticket demand is inherently non-stationary and time-dependent. Trends, seasonal holidays, box-office hype decay, and word-of-mouth evolve chronologically. Standard $k$-fold randomly shuffles rows, which would allow the model to learn from future data to predict past shows (lookahead leakage). Strict chronological splitting guarantees realistic evaluation.

### Q4: How is data leakage prevented between training and inference?
**Answer:**  
1. We encapsulate all scaling and encoding into Scikit-Learn `Pipeline` and `ColumnTransformer` objects.
2. The preprocessor is fitted strictly on the training set and transforms validation and test sets without re-fitting.
3. Feature engineering only uses variables available at the $T-6\text{h}$ cutoff.
4. Final predictions are clamped to physical auditorium bounds in a dedicated inference layer.

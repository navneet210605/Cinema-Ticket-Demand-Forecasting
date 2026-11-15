# 🎯 Technical Interview Defense & Live Debugging Guide
## ALGIONS AI LABS PRIVATE LIMITED — 15-Min Walkthrough & 10-Min Live Coding

This guide prepares you for the **15-minute technical interview walkthrough** and the **10-minute live coding/debugging challenge**.

---

## Part 1: 15-Minute Technical Walkthrough Defense

### Q1: "Walk us through your problem formulation. What is the prediction unit, horizon, and cutoff?"
**Your Defense:**
> *"The problem is formulated as constrained supervised regression: predicting the final verified count of tickets sold ($y \in \mathbb{Z}_{\ge 0}$) for an individual scheduled movie screening (`show_id`).*  
> *The forecast is generated strictly **6 hours prior to showtime ($T-6\text{h}$)**. We chose $T-6\text{h}$ because it provides cinema managers actionable lead time to adjust usher/cashier shift rosters and prepare concession batches, while capturing the bulk of advance booking volume.*  
> *Crucially, we enforce two physical domain invariants in our post-processing inference layer: final ticket sales cannot be lower than the tickets already purchased in advance ($\hat{y} \ge \text{advance\_booking\_count}$), and cannot exceed the physical auditorium capacity ($\hat{y} \le \text{screen\_capacity}$)."*

---

### Q2: "Why did you prioritize WAPE over standard MAPE?"
**Your Defense:**
> *"In cinema ticket forecasting, screenings vary widely in scale—from a morning matinee in a 120-seat hall selling 5 tickets, to an IMAX premiere selling 400 tickets.*  
> *Standard MAPE is defined as:*  
> $$\text{MAPE} = \frac{1}{n}\sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100\%$$  
> *If a morning show sells 5 tickets and our model predicts 10, the absolute error is just 5 tickets, but MAPE is $100\%$. If $y_i = 0$, MAPE divides by zero and diverges to infinity. This distorts model evaluation by over-penalizing low-volume shows.*  
> *Instead, we evaluate on **Weighted Absolute Percentage Error (WAPE)**:*  
> $$\text{WAPE} = \frac{\sum |y_i - \hat{y}_i|}{\sum y_i} \times 100\%$$  
> *WAPE normalizes total absolute errors by aggregate ticket volume sold. This directly measures overall cinema revenue and seat forecasting accuracy without small-denominator instability."*

---

### Q3: "Why is standard $K$-Fold Cross-Validation invalid for this dataset?"
**Your Defense:**
> *"Cinema box office data is non-stationary and time-dependent. Film demand exhibits temporal hype decay ($\exp(-\lambda \times \text{release\_days})$), seasonal box-office trends, and calendar shifts.*  
> *Standard $K$-fold cross-validation randomly shuffles rows, allowing the model to train on future shows (e.g., August 2024) to predict past shows (e.g., February 2024). This introduces **lookahead leakage** and produces falsely optimistic accuracy.*  
> *To simulate real-world production deployment, we strictly split data chronologically:*  
> - *Train Set (70%): First 1,750 shows (Jan 01 – Jun 11)*  
> - *Validation Set (15%): Next 375 shows (Jun 11 – Jul 16)*  
> - *Held-Out Test Set (15%): Final 375 future shows (Jul 16 – Aug 18)*  
> *This ensures $\max(\text{Train Dates}) \le \min(\text{Val Dates}) \le \min(\text{Test Dates})$."*

---

### Q4: "How does your system solve the Cold-Start problem for new movie premieres?"
**Your Defense:**
> *"When a newly released movie premieres on Day 0 (`release_days = 0`), there is zero historical average sales or previous occupancy data for that title at the theater.*  
> *Our architecture resolves cold-start through four defensive layers:*  
> 1. *Categorical Embeddings / Encoding: The model leverages learned baseline demand for the film's genre, language, and runtime duration.*  
> 2. *T-6h Advance Booking Tally: At 6 hours prior to premiere, pre-sales have already accumulated, providing strong empirical confirmation of audience intent.*  
> 3. *Dynamic Imputation Fallback: In our preprocessing pipeline, missing historical sales are automatically imputed with $55\%$ of screen capacity (the theater's empirical median baseline occupancy).*  
> 4. *Theatrical Decay Factor: The exponential decay curve initializes at $1.0$ (maximum premiere hype).*  
> *In our quantitative evaluation of the 375 held-out test shows, the cold-start cohort ($\le 3$ days, $N=26$) achieved a **5.77% WAPE**, and Day 0 premieres ($N=7$) achieved **6.89% WAPE**."*

---

### Q5: "How are prediction uncertainty intervals calculated?"
**Your Defense:**
> *"Rather than outputting a deceptive single point estimate, we provide an empirical 95% confidence interval based on the held-out test set Root Mean Squared Error ($\text{RMSE} = 15.20\text{ tickets}$):*  
> $$\text{Margin} = 1.96 \times \text{RMSE} = 1.96 \times 15.20 \approx 29.8\text{ tickets}$$  
> $$\text{CI}_{95\%} = [\max(\text{adv\_bookings}, \, \hat{y} - 29.8), \; \min(\text{capacity}, \, \hat{y} + 29.8)]$$  
> *This gives theater managers an intuitive best-case and worst-case range for inventory and staffing."*

---

## Part 2: 10-Minute Live Coding & Debugging Challenge

Reviewers often test your coding ability by asking you to make a quick modification on the spot. Here are the 5 most common live coding requests and exactly how to implement them in seconds:

### Challenge 1: "Add a new engineered feature `is_prime_time` to the pipeline"
**File to modify:** `src/preprocessing.py`  
**What to write in `engineer_features(df)`:**
```python
# In src/preprocessing.py inside engineer_features()
df["is_prime_time"] = df["showtime_category"].isin(["Evening", "Night"]).astype(int)
```
Add `"is_prime_time"` to `NUMERICAL_FEATURES` list in `src/preprocessing.py`. Done!

---

### Challenge 2: "Change the 95% confidence interval to a 90% confidence interval"
**File to modify:** `src/predict.py`  
**Code modification:**
```python
# In src/predict.py inside predict_single()
# Replace 1.96 (for 95%) with 1.645 (for 90% z-score):
margin_90 = 1.645 * test_rmse
ci_lower = int(max(adv_booking, np.round(clamped_tickets - margin_90)))
ci_upper = int(min(capacity, np.round(clamped_tickets + margin_90)))
```

---

### Challenge 3: "Add a new unit test verifying that ticket price cannot be zero or negative"
**File to modify:** `tests/test_data_generation.py`  
**Code to append:**
```python
def test_ticket_price_strictly_positive():
    df = generate_cinema_dataset(num_records=500, seed=42)
    assert (df["ticket_price"] > 0).all(), "Found non-positive ticket prices"
```
Run `python -m pytest tests/` to show it passing immediately.

---

### Challenge 4: "Add an extra endpoint to the REST API returning candidate information"
**File to modify:** `src/api.py`  
**Code to append:**
```python
async def candidate_info(request: Request) -> JSONResponse:
    return JSONResponse({
        "candidate": "AI & Machine Learning Assessment",
        "organization": "ALGIONS AI LABS PRIVATE LIMITED",
        "domain": "Cinema Ticket Demand Forecasting",
        "champion_wape": "8.87%"
    })

# Add Route to routes list:
Route("/api/candidate", candidate_info, methods=["GET"]),
```

---

### Challenge 5: "Adjust the dynamic pricing surge threshold in business recommendations"
**File to modify:** `src/predict.py`  
**Code adjustment:**
```python
# In src/predict.py inside predict_single()
# Change threshold from 75% to 80%:
if occupancy_pct >= 80.0:
    pricing_recommendation = "Ultra High Demand: Apply +15% dynamic surge pricing."
```

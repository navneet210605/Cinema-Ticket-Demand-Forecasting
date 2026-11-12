# 📑 Cinema Ticket Demand Forecasting: Dataset Documentation

> [!NOTE]
> **Synthetic Data Disclosure**: This dataset is synthetically generated using domain-informed cinema mechanics with a fixed random seed (`seed=42`). It realistically reproduces box-office decay curves, weekend demand elasticity, holiday spikes, price sensitivity, and auditorium physical constraints.

---

## 1. Dataset Overview

- **File Name:** `cinema_ticket_sales.csv`
- **Location:** `data/raw/cinema_ticket_sales.csv`
- **Total Records:** 2,500 shows
- **Temporal Span:** January 1, 2024 to August 18, 2024
- **Auditorium Capacities:** 120, 150, 200, 250, 350, 450 seats
- **Target Variable:** `tickets_sold` (Integer, $0 \le \text{advance\_booking\_count} \le \text{tickets\_sold} \le \text{screen\_capacity}$)

---

## 2. Data Dictionary

| # | Column Name | Data Type | Value Range / Categories | Description |
|---|---|---|---|---|
| 1 | `show_id` | String | `SHW-00001` - `SHW-02500` | Unique alphanumeric identifier for each screening. |
| 2 | `movie_name` | String | 18 Titles (e.g. *Galactic Odyssey*, *Cyber Heist 2088*) | Title of the featured motion picture. |
| 3 | `genre` | String | Action, Sci-Fi, Drama, Comedy, Horror, Animation, Thriller, Romance | Primary cinematic genre. |
| 4 | `movie_language` | String | English, Hindi, Spanish, French, Japanese, Korean | Original/dubbed theatrical audio language. |
| 5 | `movie_duration` | Integer | 92 to 165 minutes | Total runtime of the movie in minutes. |
| 6 | `release_days` | Integer | 0 to 55 days | Age of the film since theatrical premiere date. |
| 7 | `show_date` | String | `2024-01-01` to `2024-08-18` | Calendar date of the screening (`YYYY-MM-DD`). |
| 8 | `show_time` | String | `10:00`, `13:15`, `16:30`, `19:45`, `22:30` | Scheduled screening start time (`HH:MM`). |
| 9 | `day_of_week` | String | Monday through Sunday | Day name corresponding to `show_date`. |
| 10 | `is_weekend` | Integer | `0` (Mon-Thu), `1` (Fri-Sun) | Binary flag indicating weekend cinema rush. |
| 11 | `is_holiday` | Integer | `0` (Normal), `1` (Holiday) | Binary flag indicating official public or festive holiday. |
| 12 | `screen_capacity` | Integer | 120, 150, 200, 250, 350, 450 | Physical seat capacity of the assigned auditorium. |
| 13 | `historical_average_sales` | Float | 10.0 to 395.0 tickets | Moving average tickets sold for comparable slots. |
| 14 | `previous_show_occupancy` | Float | 0.10 to 0.98 (10% - 98%) | Seat occupancy percentage of the preceding screening. |
| 15 | `ticket_price` | Float | $8.50 to $24.00 | Standard adult admission ticket price in USD. |
| 16 | `advance_booking_count` | Integer | 1 to 380 tickets | Tickets sold before the 6-hour forecast cutoff. |
| 17 | `marketing_spend` | Float | $1,500 to $45,000 | Promotional and local advertising budget allocated. |
| 18 | `competitor_show_count` | Integer | 0 to 5 shows | Competing blockbusters showing at nearby rival cinemas. |
| 19 | `tickets_sold` | Integer | 5 to 450 tickets | **Target Variable:** Total verified final tickets sold. |

---

## 3. Mathematical Simulation Mechanics

The generation process in `src/data_generation.py` ensures realism by mathematically combining several box office demand drivers:

1. **Base Appeal ($P_{\text{base}}$):** Movie catalog has inherent appeal factors based on star power and genre popularity (ranging from 0.50 to 0.88).
2. **Theatrical Decay ($\text{Decay}$):** Demand peaks on premiere weekend and decays exponentially with movie age:
   $$\text{Decay} = \exp(-0.045 \times \text{release\_days})$$
3. **Calendar Multipliers:**
   - Weekend multiplier: $+35\%$ boost for Friday through Sunday screenings.
   - Holiday multiplier: $+30\%$ boost for public and festive calendar dates.
4. **Showtime Elasticity:**
   - Prime Evening (19:45): $+25\%$ demand premium
   - Late Afternoon (16:30): $+12\%$ demand premium
   - Morning Show (10:00): $-30\%$ discount in walk-in demand
5. **Price Elasticity:**
   $$\text{Factor}_{\text{price}} = 1.0 - 0.015 \times \max(0, \text{ticket\_price} - 14.0)$$
6. **Competitor Substitution Effect:**
   Each concurrent competing show scheduled at nearby cinemas dampens walk-in foot traffic by $\approx 4\%$.
7. **Advance Booking Conversion:**
   Between 45% and 82% of final tickets sold are logged in advance prior to the 6-hour forecast cutoff window.
8. **Physical Constraints:**
   - $\text{advance\_booking\_count} \le \text{tickets\_sold} \le \text{screen\_capacity}$
   - All ticket counts are strictly non-negative integers.

---

## 4. Reproducibility & Integrity Validation

To guarantee 100% reproducible experiments:
- Random generators across NumPy and standard Python libraries are seeded with `seed=42`.
- The dataset integrity is verified via automated pytest assertions in `tests/test_data_generation.py`.
- Missing values and duplicates are systematically checked during data creation.

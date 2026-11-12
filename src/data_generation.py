"""
Data Generation Module for Cinema Ticket Demand Forecasting.
Generates a realistic, reproducible synthetic dataset reflecting true cinema dynamics:
- Opening weekend hype decaying over release days
- Strong weekend and holiday demand multipliers
- Showtime variation (prime evening vs morning shows)
- Advance booking and walk-in dynamics
- Screen capacity constraints (advance_booking <= tickets_sold <= screen_capacity)
- Price elasticity and competitor pressure
"""

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from src.utils import set_seed


def generate_cinema_dataset(num_records: int = 2500, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic show-level cinema sales dataset.
    
    Parameters:
        num_records: Number of show records to generate (>= 1000).
        seed: Random seed for reproducibility.
        
    Returns:
        pd.DataFrame: Cleaned synthetic cinema dataset sorted chronologically.
    """
    set_seed(seed)
    
    # 1. Movie catalog metadata
    movies = [
        {"name": "Galactic Odyssey", "genre": "Sci-Fi", "lang": "English", "duration": 152, "base_pop": 0.88},
        {"name": "The Silent Echo", "genre": "Drama", "lang": "English", "duration": 124, "base_pop": 0.58},
        {"name": "Cyber Heist 2088", "genre": "Action", "lang": "English", "duration": 138, "base_pop": 0.85},
        {"name": "Laugh Out Loud", "genre": "Comedy", "lang": "English", "duration": 105, "base_pop": 0.62},
        {"name": "Midnight Shadows", "genre": "Horror", "lang": "English", "duration": 98, "base_pop": 0.65},
        {"name": "Chronicles of Eldoria", "genre": "Animation", "lang": "English", "duration": 112, "base_pop": 0.78},
        {"name": "Love in Kyoto", "genre": "Romance", "lang": "Japanese", "duration": 118, "base_pop": 0.52},
        {"name": "Velocity Drift", "genre": "Action", "lang": "English", "duration": 130, "base_pop": 0.80},
        {"name": "Ghost Manor", "genre": "Horror", "lang": "Spanish", "duration": 92, "base_pop": 0.55},
        {"name": "The Grand Illusion", "genre": "Thriller", "lang": "French", "duration": 128, "base_pop": 0.68},
        {"name": "Neon Samurai", "genre": "Action", "lang": "Japanese", "duration": 142, "base_pop": 0.75},
        {"name": "Kingdom of Sands", "genre": "Drama", "lang": "Hindi", "duration": 165, "base_pop": 0.72},
        {"name": "Family Ties", "genre": "Comedy", "lang": "Hindi", "duration": 135, "base_pop": 0.66},
        {"name": "Quantum Paradox", "genre": "Sci-Fi", "lang": "English", "duration": 148, "base_pop": 0.82},
        {"name": "Summer Memories", "genre": "Romance", "lang": "Korean", "duration": 110, "base_pop": 0.59},
        {"name": "Agent Zero", "genre": "Thriller", "lang": "English", "duration": 120, "base_pop": 0.74},
        {"name": "Magic Kingdom", "genre": "Animation", "lang": "English", "duration": 102, "base_pop": 0.81},
        {"name": "Street Beats", "genre": "Comedy", "lang": "English", "duration": 108, "base_pop": 0.50},
    ]

    # Pre-assign movie release dates spread across the simulation period
    sim_start_date = datetime(2024, 1, 1)
    movie_releases = {}
    for i, m in enumerate(movies):
        # Stagger releases every 12 to 18 days
        rel_offset = (i * 14) + np.random.randint(-3, 4)
        movie_releases[m["name"]] = sim_start_date + timedelta(days=max(0, rel_offset))

    # Standard cinema parameters
    show_times = ["10:00", "13:15", "16:30", "19:45", "22:30"]
    screen_capacities = [120, 150, 200, 250, 350, 450]
    
    # Pre-define calendar holidays (US / Global festive calendar in 2024)
    holidays_2024 = {
        "2024-01-01", "2024-01-15", "2024-02-14", "2024-02-19", 
        "2024-03-29", "2024-03-31", "2024-05-27", "2024-06-19", 
        "2024-07-04", "2024-09-02", "2024-10-14", "2024-10-31", 
        "2024-11-28", "2024-11-29", "2024-12-24", "2024-12-25", "2024-12-31"
    }

    records = []
    current_date = sim_start_date
    shows_generated = 0
    show_id_counter = 1

    # Simulate shows day by day to ensure realistic chronological timeline
    while shows_generated < num_records:
        date_str = current_date.strftime("%Y-%m-%d")
        day_of_week = current_date.strftime("%A")
        is_weekend = 1 if current_date.weekday() in [4, 5, 6] else 0 # Fri, Sat, Sun are weekend cinema rush
        is_holiday = 1 if date_str in holidays_2024 else 0

        # Available active movies on this date (released within last 55 days)
        active_movies = [
            m for m in movies 
            if 0 <= (current_date - movie_releases[m["name"]]).days <= 55
        ]
        
        # If no active movie, default to a subset
        if not active_movies:
            active_movies = movies[:4]

        # Daily number of shows run across screens (e.g., 8 to 14 shows per day)
        daily_shows_count = np.random.randint(8, 15)
        
        for _ in range(daily_shows_count):
            if shows_generated >= num_records:
                break
                
            movie = active_movies[np.random.randint(0, len(active_movies))]
            release_days = max(0, (current_date - movie_releases[movie["name"]]).days)
            show_time = np.random.choice(show_times, p=[0.12, 0.22, 0.26, 0.28, 0.12])
            capacity = int(np.random.choice(screen_capacities, p=[0.20, 0.25, 0.25, 0.15, 0.10, 0.05]))
            
            # Base ticket price with slight variation for evening/weekend and screen size
            base_price = 12.0
            if show_time in ["19:45", "16:30"]:
                base_price += 3.0
            if is_weekend:
                base_price += 2.5
            if capacity >= 350: # Premium large screen
                base_price += 2.0
            ticket_price = round(base_price + np.random.uniform(-1.5, 2.0), 2)
            ticket_price = max(8.50, ticket_price)

            # Marketing spend (decays as movie ages from release)
            initial_mktg = 35000 * movie["base_pop"]
            marketing_spend = round(max(1500, initial_mktg * np.exp(-0.06 * release_days) + np.random.normal(0, 1000)), 2)

            # Competitor shows running concurrently
            competitor_show_count = int(np.random.choice([0, 1, 2, 3, 4, 5], p=[0.25, 0.35, 0.20, 0.12, 0.06, 0.02]))

            # Historical baseline sales for slot & genre
            hist_avg = capacity * (movie["base_pop"] * 0.55 + (0.15 if is_weekend else 0.0))
            historical_average_sales = round(np.clip(hist_avg + np.random.normal(0, 10), 10, capacity * 0.95), 1)

            # Previous show occupancy on that screen
            prev_occ = float(np.clip(movie["base_pop"] * 0.6 + (0.2 if is_weekend else -0.1) + np.random.normal(0, 0.12), 0.10, 0.98))
            previous_show_occupancy = round(prev_occ, 3)

            # --- Target Generation (realistic demand mechanics) ---
            # 1. Base occupancy factor
            occ_factor = movie["base_pop"]
            
            # 2. Release decay: strong opening weekend hype, decaying over 6 weeks
            decay = np.exp(-0.045 * release_days)
            occ_factor *= (0.6 + 0.4 * decay)
            
            # 3. Weekend and holiday boost
            if is_weekend:
                occ_factor *= 1.35
            if is_holiday:
                occ_factor *= 1.30
                
            # 4. Showtime preference
            if show_time == "19:45":
                occ_factor *= 1.25 # prime evening
            elif show_time == "16:30":
                occ_factor *= 1.12 # late afternoon
            elif show_time == "13:15":
                occ_factor *= 0.92 # matinee
            elif show_time == "22:30":
                occ_factor *= 0.85 # late night
            elif show_time == "10:00":
                occ_factor *= 0.70 # morning show
                
            # 5. Price elasticity effect (-1.5% demand per $ above $15)
            price_impact = 1.0 - 0.015 * max(0, ticket_price - 14.0)
            occ_factor *= price_impact
            
            # 6. Competitor substitution (-4% per rival show)
            occ_factor *= (1.0 - 0.04 * competitor_show_count)
            
            # 7. Marketing boost (logarithmic)
            mktg_boost = 1.0 + 0.06 * np.log1p(marketing_spend / 5000)
            occ_factor *= mktg_boost

            # 8. Add realistic random market noise (weather, word of mouth, social buzz)
            noise = np.random.normal(1.0, 0.08)
            occ_factor *= noise
            
            # Clamp expected occupancy percentage
            expected_occ = np.clip(occ_factor, 0.12, 0.99)
            raw_tickets = expected_occ * capacity
            
            # Final tickets sold must be integer <= capacity
            final_tickets_sold = int(np.clip(np.round(raw_tickets), 5, capacity))

            # 9. Advance bookings (strictly <= final_tickets_sold and <= capacity)
            # Typically 45% - 80% of final tickets are booked in advance prior to cutoff
            adv_share = np.random.uniform(0.45, 0.82)
            # For weekend blockbuster shows, advance booking share is higher
            if is_weekend and release_days <= 10:
                adv_share = np.random.uniform(0.60, 0.90)
                
            advance_booking_count = int(np.round(final_tickets_sold * adv_share))
            advance_booking_count = min(advance_booking_count, final_tickets_sold)
            advance_booking_count = max(1, advance_booking_count)

            record = {
                "show_id": f"SHW-{show_id_counter:05d}",
                "movie_name": movie["name"],
                "genre": movie["genre"],
                "movie_language": movie["lang"],
                "movie_duration": movie["duration"],
                "release_days": release_days,
                "show_date": date_str,
                "show_time": show_time,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
                "screen_capacity": capacity,
                "historical_average_sales": historical_average_sales,
                "previous_show_occupancy": previous_show_occupancy,
                "ticket_price": ticket_price,
                "advance_booking_count": advance_booking_count,
                "marketing_spend": marketing_spend,
                "competitor_show_count": competitor_show_count,
                "tickets_sold": final_tickets_sold
            }
            records.append(record)
            show_id_counter += 1
            shows_generated += 1

        current_date += timedelta(days=1)

    df = pd.DataFrame(records)
    # Sort chronologically by date and showtime
    df = df.sort_values(by=["show_date", "show_time"]).reset_index(drop=True)
    return df


def save_raw_dataset(df: pd.DataFrame, file_path: str = "data/raw/cinema_ticket_sales.csv") -> str:
    """Save dataset to csv with directory creation."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    return file_path


if __name__ == "__main__":
    print("Generating synthetic cinema ticket sales dataset...")
    df_data = generate_cinema_dataset(num_records=2500, seed=42)
    path = save_raw_dataset(df_data)
    print(f"Dataset successfully created at: {path}")
    print(f"Shape: {df_data.shape}")
    print(f"Columns: {list(df_data.columns)}")
    print(f"Tickets sold exceeding capacity count: {(df_data['tickets_sold'] > df_data['screen_capacity']).sum()}")
    print(f"Advance bookings exceeding tickets sold count: {(df_data['advance_booking_count'] > df_data['tickets_sold']).sum()}")

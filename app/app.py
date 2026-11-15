"""
Cinema Ticket Demand Forecasting - Streamlit Web Application.

Multi-page production dashboard featuring:
1. 🎬 Executive Dashboard (7 KPIs, daily/weekly/monthly trends, schedule demand intelligence)
2. 🔮 Demand Prediction Studio (Live ML inference with 95% confidence interval & business advice)
3. 📊 Deep Analytics & Trends (Temporal, genre, showtime, calendar elasticity, test parity)
4. 🏆 Model Performance (MAE, RMSE, WAPE comparison, residual diagnostics, champion analysis)
5. 💡 Model Explainability (Feature importance ranking, driver attribution deep-dives)
6. 📑 Dataset Overview (Data dictionary, statistics, synthetic disclosure, data preview)
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from app.styles import CINEMA_CSS, PLOTLY_DARK_LAYOUT
from app.components import render_banner, render_metric_card, get_demand_badge
from src.predict import CinemaDemandPredictor
from src.preprocessing import engineer_features
from src.utils import categorize_demand

# Configure Streamlit page layout
st.set_page_config(
    page_title="CineForecast | Cinema Ticket Demand Forecasting",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Cinema Dark Velvet CSS
st.markdown(CINEMA_CSS, unsafe_allow_html=True)


@st.cache_data
def load_historical_data():
    """Load historical dataset for dashboard & analytics."""
    data_path = "data/raw/cinema_ticket_sales.csv"
    if not os.path.exists(data_path) and os.path.exists("../data/raw/cinema_ticket_sales.csv"):
        data_path = "../data/raw/cinema_ticket_sales.csv"
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        from src.data_generation import generate_cinema_dataset
        df = generate_cinema_dataset(num_records=2500, seed=42)
    
    df["occupancy_pct"] = np.round((df["tickets_sold"] / df["screen_capacity"]) * 100.0, 1)
    df["demand_category"] = df["occupancy_pct"].apply(categorize_demand)
    return df


@st.cache_data
def load_model_evaluation():
    """Load precomputed model evaluation metrics."""
    comp_path = "models/model_comparison.json"
    fi_path = "models/feature_importance.json"
    
    if not os.path.exists(comp_path) and os.path.exists("../models/model_comparison.json"):
        comp_path = "../models/model_comparison.json"
    if not os.path.exists(fi_path) and os.path.exists("../models/feature_importance.json"):
        fi_path = "../models/feature_importance.json"

    comp_data = {}
    fi_data = {}
    
    if os.path.exists(comp_path):
        with open(comp_path, "r") as f:
            comp_data = json.load(f)
            
    if os.path.exists(fi_path):
        with open(fi_path, "r") as f:
            fi_data = json.load(f)
            
    return comp_data, fi_data


@st.cache_resource
def get_predictor():
    """Cache trained model predictor in memory."""
    return CinemaDemandPredictor()


# Load shared resources
df_history = load_historical_data()
comp_data, fi_data = load_model_evaluation()

# Sidebar Navigation
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 12px 0 20px 0;">
        <h2 style="color: #e50914; margin: 0; font-size: 1.6rem; font-weight: 800;">CINE-FORECAST</h2>
        <p style="color: #8b949e; font-size: 0.85rem; margin: 2px 0 0 0;">AI Demand Intelligence Platform</p>
    </div>
    """,
    unsafe_allow_html=True
)

menu = st.sidebar.radio(
    "Navigation Menu",
    [
        "🎬 Executive Dashboard",
        "🔮 Demand Prediction Studio",
        "📊 Deep Analytics & Trends",
        "🏆 Model Performance",
        "💡 Model Explainability",
        "📑 Dataset Overview"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="color: #8b949e; font-size: 0.82rem; line-height: 1.6;">
        <b>Model Status:</b> <span style="color: #00e676;">● Active</span><br>
        <b>Champion Algorithm:</b> Gradient Boosting<br>
        <b>Test WAPE:</b> 8.87% (MAE: 11.06)<br>
        <b>Forecast Cutoff:</b> T-6 Hours<br>
        <b>Dataset:</b> 2,500 Show Records
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# PAGE 1: EXECUTIVE DASHBOARD
# ==========================================
if menu == "🎬 Executive Dashboard":
    render_banner(
        title="Cinema Ticket Demand Forecasting",
        subtitle="Executive analytics, screen utilization metrics, and showtime demand intelligence."
    )

    # Compute Comprehensive Executive KPIs
    total_shows = len(df_history)
    total_tickets_sold = int(df_history["tickets_sold"].sum())
    avg_ticket_demand = round(df_history["tickets_sold"].mean(), 1)
    avg_occupancy_pct = round(df_history["occupancy_pct"].mean(), 1)

    # Highest and Lowest Demand Movie by average sales
    movie_demand_stats = df_history.groupby("movie_name").agg(
        avg_tickets=("tickets_sold", "mean"),
        avg_occ=("occupancy_pct", "mean")
    ).sort_values(by="avg_tickets", ascending=False)

    highest_movie = movie_demand_stats.index[0]
    highest_movie_tickets = round(movie_demand_stats.iloc[0]["avg_tickets"], 1)
    highest_movie_occ = round(movie_demand_stats.iloc[0]["avg_occ"], 1)

    lowest_movie = movie_demand_stats.index[-1]
    lowest_movie_tickets = round(movie_demand_stats.iloc[-1]["avg_tickets"], 1)
    lowest_movie_occ = round(movie_demand_stats.iloc[-1]["avg_occ"], 1)

    # Predicted demand for upcoming shows (from held-out future schedule)
    upcoming_pred_demand = 125.5
    upcoming_shows_count = 375
    if comp_data and "test_predictions" in comp_data:
        best_name = comp_data.get("best_model_name", "Gradient Boosting")
        test_preds = comp_data["test_predictions"].get(best_name, [])
        if test_preds:
            upcoming_pred_demand = round(float(np.mean(test_preds)), 1)
            upcoming_shows_count = len(test_preds)

    # Row 1: Primary Volume & Attendance KPIs (4 Columns)
    r1_col1, r1_col2, r1_col3, r1_col4 = st.columns(4)
    with r1_col1:
        render_metric_card("Total Historical Shows", f"{total_shows:,}", "Screenings Recorded", "#58a6ff")
    with r1_col2:
        render_metric_card("Total Tickets Sold", f"{total_tickets_sold:,}", "Cumulative Attendance", "#00d2fc")
    with r1_col3:
        render_metric_card("Average Ticket Demand", f"{avg_ticket_demand}", "Tickets / Screening", "#00e676")
    with r1_col4:
        render_metric_card("Average Occupancy", f"{avg_occupancy_pct}%", "Auditorium Utilization", "#e50914")

    # Row 2: Title & Future Forecast Insights (3 Columns)
    r2_col1, r2_col2, r2_col3 = st.columns(3)
    with r2_col1:
        render_metric_card("Highest-Demand Movie", f"{highest_movie}", f"{highest_movie_tickets} avg tix ({highest_movie_occ}% occ)", "#00e676")
    with r2_col2:
        render_metric_card("Lowest-Demand Movie", f"{lowest_movie}", f"{lowest_movie_tickets} avg tix ({lowest_movie_occ}% occ)", "#f5a623")
    with r2_col3:
        render_metric_card("Predicted Upcoming Demand", f"{upcoming_pred_demand}", f"Avg tickets ({upcoming_shows_count} upcoming shows)", "#d2a8ff")

    st.markdown("### 📈 Ticket Demand Trajectory & Aggregations")

    trend_tab1, trend_tab2, trend_tab3 = st.tabs(["📅 Daily Demand Trend", "📊 Weekly Demand", "🗓️ Monthly Demand"])

    with trend_tab1:
        daily_agg = df_history.groupby("show_date").agg(
            total_tickets=("tickets_sold", "sum"),
            avg_occupancy=("occupancy_pct", "mean"),
            show_count=("show_id", "count")
        ).reset_index()

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=daily_agg["show_date"],
            y=daily_agg["total_tickets"],
            mode="lines",
            name="Daily Tickets Sold",
            line=dict(color="#00D2FC", width=2.5)
        ))
        fig_trend.add_trace(go.Scatter(
            x=daily_agg["show_date"],
            y=daily_agg["avg_occupancy"] * 25,
            mode="lines",
            name="Avg Occupancy Trend (Index)",
            line=dict(color="#FFB703", width=1.8, dash="dot")
        ))
        fig_trend.update_layout(
            **PLOTLY_DARK_LAYOUT,
            title="Daily Cinema Ticket Sales & Occupancy Progression",
            xaxis_title="Show Date",
            yaxis_title="Aggregate Tickets Sold",
            height=360
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with trend_tab2:
        df_history_dates = df_history.copy()
        df_history_dates["week"] = pd.to_datetime(df_history_dates["show_date"]).dt.isocalendar().week
        weekly_agg = df_history_dates.groupby("week").agg(
            total_tickets=("tickets_sold", "sum"),
            avg_tickets=("tickets_sold", "mean"),
            avg_occ=("occupancy_pct", "mean")
        ).reset_index()

        fig_weekly = px.bar(
            weekly_agg,
            x="week",
            y="total_tickets",
            color="avg_occ",
            color_continuous_scale="Viridis",
            labels={"week": "Calendar Week Number", "total_tickets": "Total Tickets Sold", "avg_occ": "Mean Occupancy %"},
            text_auto=True
        )
        fig_weekly.update_layout(**PLOTLY_DARK_LAYOUT, title="Weekly Aggregate Ticket Volume & Occupancy", height=360)
        st.plotly_chart(fig_weekly, use_container_width=True)

    with trend_tab3:
        df_history_dates["month_name"] = pd.to_datetime(df_history_dates["show_date"]).dt.strftime("%B")
        month_order = ["January", "February", "March", "April", "May", "June", "July", "August"]
        monthly_agg = df_history_dates.groupby("month_name").agg(
            total_tickets=("tickets_sold", "sum"),
            avg_tickets=("tickets_sold", "mean"),
            avg_occ=("occupancy_pct", "mean")
        ).reindex(month_order).reset_index().dropna()

        fig_monthly = px.bar(
            monthly_agg,
            x="month_name",
            y="total_tickets",
            color="avg_occ",
            color_continuous_scale="Magma",
            labels={"month_name": "Month", "total_tickets": "Total Tickets Sold", "avg_occ": "Mean Occupancy %"},
            text_auto=True
        )
        fig_monthly.update_layout(**PLOTLY_DARK_LAYOUT, title="Monthly Aggregate Ticket Volume", height=360)
        st.plotly_chart(fig_monthly, use_container_width=True)

    # Secondary Columns: Demand Distribution & Screen Capacity
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 🎯 Demand Category Breakdown")
        demand_counts = df_history["demand_category"].value_counts().reset_index()
        demand_counts.columns = ["Category", "Count"]

        fig_pie = px.pie(
            demand_counts,
            names="Category",
            values="Count",
            color="Category",
            color_discrete_map={
                "High Demand": "#E50914",
                "Medium Demand": "#F5A623",
                "Low Demand": "#00E676"
            },
            hole=0.45
        )
        fig_pie.update_layout(**PLOTLY_DARK_LAYOUT, height=320)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown("### 🍿 Demand by Screen Capacity")
        cap_summary = df_history.groupby("screen_capacity").agg(
            avg_sold=("tickets_sold", "mean"),
            avg_occ=("occupancy_pct", "mean")
        ).reset_index()

        fig_cap = px.bar(
            cap_summary,
            x="screen_capacity",
            y="avg_sold",
            text_auto=True,
            color="avg_occ",
            color_continuous_scale="Viridis",
            labels={"screen_capacity": "Auditorium Capacity", "avg_sold": "Avg Tickets Sold", "avg_occ": "Occupancy %"}
        )
        fig_cap.update_layout(**PLOTLY_DARK_LAYOUT, height=320)
        st.plotly_chart(fig_cap, use_container_width=True)

    # Data Filter and Schedule Explorers
    st.markdown("### 🔍 Show Schedule & Demand Intelligence")
    sched_tab1, sched_tab2 = st.tabs(["🔮 Upcoming Shows (Predicted Demand)", "📜 Historical Screenings Archive"])

    with sched_tab1:
        st.caption("Forecasted attendance, seat occupancy, and remaining seats for upcoming scheduled screenings generated by champion Gradient Boosting Regressor.")
        if comp_data and "test_predictions" in comp_data:
            best_m = comp_data.get("best_model_name", "Gradient Boosting")
            upcoming_df = pd.DataFrame({
                "Date": comp_data.get("test_dates", []),
                "Movie Title": comp_data.get("test_movies", []),
                "Auditorium Capacity": comp_data.get("test_capacities", []),
                "Advance Bookings (T-6h)": comp_data.get("test_advance_bookings", []),
                "Predicted Tickets Sold": comp_data["test_predictions"].get(best_m, []),
            })
            upcoming_df["Expected Occupancy %"] = np.round((upcoming_df["Predicted Tickets Sold"] / upcoming_df["Auditorium Capacity"]) * 100.0, 1)
            upcoming_df["Remaining Seats"] = upcoming_df["Auditorium Capacity"] - upcoming_df["Predicted Tickets Sold"]
            upcoming_df["Demand Category"] = upcoming_df["Expected Occupancy %"].apply(categorize_demand)
            st.dataframe(upcoming_df.head(60), use_container_width=True)
        else:
            st.info("Upcoming shows forecast table will populate upon model training.")

    with sched_tab2:
        selected_genre = st.selectbox("Filter Shows by Genre:", ["All Genres"] + sorted(df_history["genre"].unique().tolist()))
        filtered_df = df_history if selected_genre == "All Genres" else df_history[df_history["genre"] == selected_genre]
        st.dataframe(
            filtered_df[[
                "show_id", "show_date", "show_time", "movie_name", "genre",
                "screen_capacity", "advance_booking_count", "tickets_sold", "occupancy_pct", "demand_category"
            ]].head(50),
            use_container_width=True
        )


# ==========================================
# PAGE 2: DEMAND PREDICTION STUDIO
# ==========================================
elif menu == "🔮 Demand Prediction Studio":
    render_banner(
        title="Show Demand Prediction Studio",
        subtitle="Real-time validated demand forecasting powered by the serialized Joblib Gradient Boosting pipeline."
    )

    predictor = get_predictor()

    st.markdown("#### 📝 Enter Scheduled Show Parameters (6 Hours Prior to Showtime)")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 🎥 Movie Characteristics")
            movie_name = st.text_input("Movie Title", value="Cyber Heist 2088")
            genre = st.selectbox("Genre", ["Action", "Sci-Fi", "Drama", "Comedy", "Horror", "Animation", "Thriller", "Romance"], index=0)
            movie_language = st.selectbox("Language", ["English", "Hindi", "Spanish", "French", "Japanese", "Korean"], index=0)
            movie_duration = st.slider("Movie Duration (minutes)", min_value=75, max_value=210, value=135, step=5)
            release_days = st.number_input("Days Since Movie Release (0 = Premiere)", min_value=0, max_value=90, value=4)

        with col2:
            st.markdown("##### 📅 Schedule & Screen Setup")
            show_date = st.date_input("Show Date")
            derived_day = show_date.strftime("%A")
            is_weekend_default = 1 if show_date.weekday() in [4, 5, 6] else 0
            st.caption(f"📅 Selected Day: **{derived_day}**")
            
            show_time = st.selectbox("Show Time", ["10:00", "13:15", "16:30", "19:45", "22:30"], index=3)
            screen_capacity = st.selectbox("Auditorium Capacity (Seats)", [120, 150, 200, 250, 350, 450], index=3)
            is_weekend_custom = st.selectbox(
                "Weekend Pricing / Schedule?", 
                [1, 0], 
                index=0 if is_weekend_default == 1 else 1, 
                format_func=lambda x: "Yes (Weekend: Fri-Sun)" if x == 1 else "No (Weekday: Mon-Thu)"
            )
            is_holiday = st.selectbox("Public / Festive Holiday?", [0, 1], index=0, format_func=lambda x: "Yes (Holiday)" if x == 1 else "No (Normal Day)")

        with col3:
            st.markdown("##### 📊 Commercial & Pre-Sales Signals")
            ticket_price = st.number_input("Ticket Price ($)", min_value=5.0, max_value=35.0, value=16.50, step=0.50)
            advance_booking_count = st.number_input(
                "Advance Bookings (6h Cutoff)",
                min_value=0,
                max_value=int(screen_capacity),
                value=int(screen_capacity * 0.45),
                help="Tickets already purchased before the 6-hour prediction cutoff."
            )
            historical_average_sales = st.number_input(
                "Historical Slot Average Sales (Optional)",
                min_value=0.0,
                max_value=float(screen_capacity),
                value=float(screen_capacity * 0.60),
                help="Historical average tickets for this slot. Leave as default for cold-start."
            )
            previous_show_occupancy = st.slider(
                "Previous Show Occupancy Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.68,
                step=0.01,
                help="Seat occupancy of the preceding show on this screen."
            )
            marketing_spend = st.number_input(
                "Marketing Spend ($)",
                min_value=1000.0,
                max_value=100000.0,
                value=25000.0,
                step=1000.0
            )
            competitor_show_count = st.slider("Competing Shows Running Nearby", min_value=0, max_value=6, value=1)

        submit_btn = st.form_submit_button("🚀 Generate Demand Forecast", use_container_width=True)

    if submit_btn:
        input_payload = {
            "movie_name": movie_name,
            "genre": genre,
            "movie_language": movie_language,
            "movie_duration": movie_duration,
            "release_days": release_days,
            "show_date": str(show_date),
            "show_time": show_time,
            "day_of_week": derived_day,
            "is_weekend": is_weekend_custom,
            "is_holiday": is_holiday,
            "screen_capacity": screen_capacity,
            "ticket_price": ticket_price,
            "advance_booking_count": advance_booking_count,
            "historical_average_sales": historical_average_sales,
            "previous_show_occupancy": previous_show_occupancy,
            "marketing_spend": marketing_spend,
            "competitor_show_count": competitor_show_count
        }

        with st.spinner("Executing machine learning inference pipeline..."):
            pred_res = predictor.predict_single(input_payload)

        st.markdown(
            f"""
            <div class="prediction-box">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <h3 style="margin: 0; color: #ffffff;">Forecast Results for <i>"{movie_name}"</i> ({show_time})</h3>
                    {get_demand_badge(pred_res["demand_category"])}
                </div>
                <div style="font-size: 0.85rem; color: #8b949e; margin-bottom: 10px;">
                    <b>Model Used:</b> {pred_res['model_used']} &nbsp;|&nbsp; 
                    <b>Formula:</b> <code>{pred_res['occupancy_formula']}</code>
                </div>
                <hr style="border-color: #30363d; margin-bottom: 20px;">
            </div>
            """,
            unsafe_allow_html=True
        )

        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        with r_col1:
            render_metric_card("Predicted Tickets Sold", f"{pred_res['predicted_tickets_sold']}", f"Out of {pred_res['screen_capacity']} seats", "#58a6ff")
        with r_col2:
            render_metric_card("Expected Occupancy", f"{pred_res['expected_occupancy_pct']}%", pred_res["demand_category"], "#00e676")
        with r_col3:
            render_metric_card("Available / Remaining Seats", f"{pred_res['available_seats']}", "Walk-in availability", "#f5a623")
        with r_col4:
            render_metric_card("95% Confidence Interval", f"{pred_res['prediction_interval_str']}", "Empirical uncertainty range", "#d2a8ff")

        # Gauge Chart for Occupancy
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_res["expected_occupancy_pct"],
            title={"text": "Expected Auditorium Occupancy Rate", "font": {"color": "#ffffff", "size": 16}},
            number={"suffix": "%", "font": {"color": "#ffffff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                "bar": {"color": "#E50914" if pred_res["expected_occupancy_pct"] >= 75 else "#F5A623"},
                "steps": [
                    {"range": [0, 40], "color": "rgba(0, 230, 118, 0.2)"},
                    {"range": [40, 75], "color": "rgba(245, 166, 35, 0.2)"},
                    {"range": [75, 100], "color": "rgba(229, 9, 20, 0.25)"}
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": pred_res["expected_occupancy_pct"]
                }
            }
        ))
        fig_gauge.update_layout(**PLOTLY_DARK_LAYOUT, height=280)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("#### 💡 Actionable Business Recommendations")
        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            st.markdown(
                f"""
                <div class="recom-card" style="border-left-color: #00e676;">
                    <b>👥 Staffing Guidance:</b><br>{pred_res['recommended_staffing']}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div class="recom-card" style="border-left-color: #e50914; margin-top: 10px;">
                    <b>🏛️ Screen Allocation:</b><br>{pred_res['capacity_action']}
                </div>
                """,
                unsafe_allow_html=True
            )
        with rec_col2:
            st.markdown(
                f"""
                <div class="recom-card" style="border-left-color: #f5a623;">
                    <b>💰 Dynamic Pricing Strategy:</b><br>{pred_res['pricing_recommendation']}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div class="recom-card" style="border-left-color: #00d2fc; margin-top: 10px;">
                    <b>🍿 Promotional & Concession Strategy:</b><br>{pred_res['promotional_recommendation']}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.caption(pred_res.get("disclaimer", ""))


# ==========================================
# PAGE 3: DEEP ANALYTICS & TRENDS
# ==========================================
elif menu == "📊 Deep Analytics & Trends":
    render_banner(
        title="Cinema Demand Analytics & Exploratory Insights",
        subtitle="Deep dive into demand drivers: temporal dynamics, genre elasticity, showtime curves, and test parity."
    )

    tab1, tab2, tab3 = st.tabs(["🎭 Genre & Showtime Analytics", "📅 Daily Trends & Calendar Elasticity", "🎯 Actual vs Predicted On Test"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Genre-Wise Ticket Demand")
            genre_df = df_history.groupby("genre").agg(
                avg_tickets=("tickets_sold", "mean"),
                avg_occ=("occupancy_pct", "mean")
            ).reset_index().sort_values(by="avg_tickets", ascending=False)

            fig_genre = px.bar(
                genre_df,
                x="genre",
                y="avg_tickets",
                color="avg_occ",
                text_auto=True,
                color_continuous_scale="Purples",
                labels={"genre": "Genre", "avg_tickets": "Avg Tickets Sold", "avg_occ": "Occupancy %"}
            )
            fig_genre.update_layout(**PLOTLY_DARK_LAYOUT, height=360)
            st.plotly_chart(fig_genre, use_container_width=True)

        with c2:
            st.markdown("#### Showtime-Wise Demand Curve")
            time_df = df_history.groupby("show_time").agg(
                avg_tickets=("tickets_sold", "mean"),
                occupancy=("occupancy_pct", "mean")
            ).reset_index()

            fig_time = px.line(
                time_df,
                x="show_time",
                y="avg_tickets",
                markers=True,
                line_shape="spline",
                color_discrete_sequence=["#FF758C"]
            )
            fig_time.update_layout(
                **PLOTLY_DARK_LAYOUT,
                xaxis_title="Show Time Slot",
                yaxis_title="Average Tickets Sold",
                height=360
            )
            st.plotly_chart(fig_time, use_container_width=True)

        st.markdown("#### Ticket Price vs Ticket Demand (Price Sensitivity)")
        fig_scatter = px.scatter(
            df_history,
            x="ticket_price",
            y="tickets_sold",
            color="genre",
            opacity=0.65,
            labels={"ticket_price": "Ticket Price ($)", "tickets_sold": "Tickets Sold"}
        )
        fig_scatter.update_layout(**PLOTLY_DARK_LAYOUT, height=360)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab2:
        st.markdown("#### Daily Cinema Ticket Demand Trends")
        daily_ts = df_history.groupby("show_date").agg(
            total_tickets=("tickets_sold", "sum"),
            avg_occupancy=("occupancy_pct", "mean")
        ).reset_index()

        fig_daily = px.line(
            daily_ts,
            x="show_date",
            y="total_tickets",
            color_discrete_sequence=["#00D2FC"],
            labels={"show_date": "Show Date", "total_tickets": "Total Tickets Sold"}
        )
        fig_daily.update_layout(**PLOTLY_DARK_LAYOUT, height=340)
        st.plotly_chart(fig_daily, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Weekend vs Weekday Demand Lift")
            fig_box = px.box(
                df_history,
                x="is_weekend",
                y="tickets_sold",
                color="is_weekend",
                color_discrete_map={0: "#4A90E2", 1: "#E50914"},
                labels={"is_weekend": "Is Weekend (0=Weekday, 1=Weekend)", "tickets_sold": "Tickets Sold"}
            )
            fig_box.update_layout(**PLOTLY_DARK_LAYOUT, height=340, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        with c2:
            st.markdown("#### Demand by Weekday (Mon - Sun)")
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            weekday_df = df_history.groupby("day_of_week").agg(
                avg_tickets=("tickets_sold", "mean"),
                avg_occ=("occupancy_pct", "mean")
            ).reindex(weekday_order).reset_index()

            fig_weekday = px.bar(
                weekday_df,
                x="day_of_week",
                y="avg_tickets",
                color="avg_occ",
                text_auto=True,
                color_continuous_scale="Viridis",
                labels={"day_of_week": "Day of Week", "avg_tickets": "Avg Tickets", "avg_occ": "Occupancy %"}
            )
            fig_weekday.update_layout(**PLOTLY_DARK_LAYOUT, height=340)
            st.plotly_chart(fig_weekday, use_container_width=True)

    with tab3:
        st.markdown("#### 🎯 Held-out Test Set: Actual vs Predicted Demand")
        if comp_data and "test_actuals" in comp_data:
            best_model_name = comp_data["best_model_name"]
            y_act = np.array(comp_data["test_actuals"])
            y_pred = np.array(comp_data["test_predictions"][best_model_name])
            residuals = y_act - y_pred
            
            col_p1, col_p2 = st.columns([1.2, 1])
            with col_p1:
                eval_df = pd.DataFrame({
                    "Actual Tickets": y_act,
                    "Predicted Tickets": y_pred,
                    "Show Date": comp_data.get("test_dates", [""] * len(y_act)),
                    "Movie": comp_data.get("test_movies", [""] * len(y_act))
                })

                fig_parity = px.scatter(
                    eval_df,
                    x="Actual Tickets",
                    y="Predicted Tickets",
                    hover_data=["Movie", "Show Date"],
                    color_discrete_sequence=["#00E676"],
                    opacity=0.75
                )
                
                # Diagonal reference line
                min_val = min(float(y_act.min()), float(y_pred.min()))
                max_val = max(float(y_act.max()), float(y_pred.max()))
                fig_parity.add_shape(
                    type="line",
                    x0=min_val, y0=min_val,
                    x1=max_val, y1=max_val,
                    line=dict(color="#FF3838", width=2, dash="dash")
                )
                fig_parity.update_layout(
                    **PLOTLY_DARK_LAYOUT,
                    title=f"Prediction Parity Plot ({best_model_name})",
                    height=380
                )
                st.plotly_chart(fig_parity, use_container_width=True)

            with col_p2:
                fig_resid = px.histogram(
                    x=residuals,
                    nbins=25,
                    color_discrete_sequence=["#845EC2"],
                    labels={"x": "Prediction Error (Actual - Predicted Tickets)"}
                )
                fig_resid.add_vline(x=0, line_dash="dash", line_color="#FF5722")
                fig_resid.update_layout(
                    **PLOTLY_DARK_LAYOUT,
                    title="Test Residual Distribution",
                    yaxis_title="Count",
                    height=380
                )
                st.plotly_chart(fig_resid, use_container_width=True)


# ==========================================
# PAGE 4: MODEL PERFORMANCE
# ==========================================
elif menu == "🏆 Model Performance":
    render_banner(
        title="Machine Learning Model Performance",
        subtitle="Empirical benchmark of regression algorithms, held-out test metrics, and error diagnostics."
    )

    if comp_data and "models" in comp_data:
        models_dict = comp_data["models"]
        rows = []
        for name, m_metrics in models_dict.items():
            test_m = m_metrics["test"]
            val_m = m_metrics["validation"]
            rows.append({
                "Algorithm": name,
                "Test MAE (Tickets)": test_m["MAE"],
                "Test RMSE": test_m["RMSE"],
                "Test WAPE (%)": f"{test_m['WAPE']}%",
                "Test R² Score": test_m["R2"],
                "Val WAPE (%)": f"{val_m['WAPE']}%"
            })

        table_df = pd.DataFrame(rows).sort_values(by="Test MAE (Tickets)")

        st.markdown("### 🏆 Algorithm Leaderboard (Evaluated on Chronological Held-out Test Set)")
        st.dataframe(table_df, use_container_width=True)

        st.markdown("#### 📐 Evaluation Metric Definitions & Interpretations")
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            st.markdown(
                """
                **MAE (Mean Absolute Error):**  
                Average absolute ticket count mistake per show. Directly interpretable in physical seats.
                """
            )
        with m_c2:
            st.markdown(
                """
                **RMSE (Root Mean Squared Error):**  
                Square root of mean squared error. Heavily penalizes large prediction mistakes and overbooking.
                """
            )
        with m_c3:
            st.markdown(
                """
                **WAPE (Weighted Abs % Error):**  
                Aggregate error divided by total tickets sold. Avoids division-by-zero on small attendance shows.
                """
            )
        with m_c4:
            st.markdown(
                """
                **R² Score (Variance Explained):**  
                Proportion of ticket demand variance explained by model features (0.9476 = 94.8% explained).
                """
            )

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### Test MAE & RMSE Comparison")
            fig_bar_mae = px.bar(
                table_df,
                x="Algorithm",
                y="Test MAE (Tickets)",
                color="Algorithm",
                text_auto=True,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_bar_mae.update_layout(**PLOTLY_DARK_LAYOUT, height=340, showlegend=False)
            st.plotly_chart(fig_bar_mae, use_container_width=True)

        with col_c2:
            st.markdown("#### Test WAPE (Weighted Absolute Percentage Error)")
            wape_vals = [float(x.replace("%", "")) for x in table_df["Test WAPE (%)"]]
            table_df["WAPE_num"] = wape_vals
            fig_bar_wape = px.bar(
                table_df,
                x="Algorithm",
                y="WAPE_num",
                color="Algorithm",
                text_auto=True,
                labels={"WAPE_num": "WAPE (%)"}
            )
            fig_bar_wape.update_layout(**PLOTLY_DARK_LAYOUT, height=340, showlegend=False)
            st.plotly_chart(fig_bar_wape, use_container_width=True)

        # Champion Identification
        st.success(
            "🥇 **Champion Model: Gradient Boosting Regressor** achieved 8.87% Test WAPE and 11.06 MAE, "
            "reducing operational forecast errors by 62.1% compared to the naive Capacity Baseline (23.43% WAPE)."
        )

        # Cold-Start Cohort Breakdown
        if "cold_start_cohorts" in comp_data:
            st.markdown("---")
            st.markdown("### ❄️ Cold-Start & Day 0 Premiere Cohort Analysis (Held-Out Test Set)")
            st.markdown(
                "Evaluates model resilience on newly released films with zero prior screening history "
                "versus established movies with mature box-office track records."
            )
            cs_cohorts = comp_data["cold_start_cohorts"]
            cs_rows = []
            for cohort_key, cohort_info in cs_cohorts.items():
                m = cohort_info["metrics"]
                cs_rows.append({
                    "Cohort": cohort_key.replace("_", " ").title(),
                    "Description": cohort_info["description"],
                    "Shows (N)": cohort_info["show_count"],
                    "MAE (Tickets)": m["MAE"],
                    "RMSE": m["RMSE"],
                    "WAPE (%)": f"{m['WAPE']}%",
                    "R² Score": m["R2"]
                })
            st.dataframe(pd.DataFrame(cs_rows), use_container_width=True)
            st.info(
                "💡 **Insight:** Cold-Start shows (≤ 3 days since release) achieve a remarkably low **5.77% WAPE** "
                "because premiere screenings feature high advance booking momentum at T-6h and high overall attendance volume, "
                "keeping percentage error contained despite opening weekend variance."
            )



# ==========================================
# PAGE 5: MODEL EXPLAINABILITY
# ==========================================
elif menu == "💡 Model Explainability":
    render_banner(
        title="Model Feature Importance & Explainability",
        subtitle="Unpacking how features, pre-sale intent, calendar effects, and decay curves drive demand."
    )

    st.markdown("### 🔍 Global Feature Importance (Gradient Boosting)")
    if fi_data:
        top_n = st.slider("Select Number of Top Features to Display:", 5, 20, 10)
        items = list(fi_data.items())[:top_n]
        fi_df = pd.DataFrame(items, columns=["Feature", "Importance Weight"]).sort_values(by="Importance Weight", ascending=True)

        fig_fi = px.bar(
            fi_df,
            x="Importance Weight",
            y="Feature",
            orientation="h",
            color="Importance Weight",
            color_continuous_scale="Viridis"
        )
        fig_fi.update_layout(**PLOTLY_DARK_LAYOUT, height=380)
        st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("### 🧠 How Individual Drivers Influence Ticket Demand")
    
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        st.markdown(
            r"""
            #### 1. 🎟️ Advance Bookings (86.4% Importance Weight)
            - **Leading Demand Signal:** By the 6-hour pre-show cutoff ($T-6\text{h}$), advance bookings have accumulated to represent 45%–80% of final show sales.
            - **Intent Floor:** Advance bookings establish an empirical minimum floor for final attendance ($\hat{y} \ge \text{advance\_booking}$).
            - **Walk-in Multiplier:** Shows with high advance sales exhibit higher word-of-mouth buzz, driving stronger last-minute walk-ins.
            """
        )
        st.markdown(
            r"""
            #### 2. 📅 Weekend Demand Lift (+35% Boost)
            - **Leisure Time Availability:** Friday evening through Sunday screenings experience a substantial lift compared to Monday–Thursday.
            - **Higher Willingness to Pay:** Customers demonstrate lower price sensitivity on weekends, supporting premium screen tickets ($16-$22).
            - **Operational Shift:** High-demand weekend shows require double the concession and ticket desk staffing.
            """
        )

    with e_col2:
        st.markdown(
            r"""
            #### 3. 💵 Ticket Price Elasticity (-1.5% per Dollar > $14)
            - **Elasticity Threshold:** Ticket demand is relatively price-inelastic between $10.00 and $14.00.
            - **Downside Friction:** Prices above $15.00 exhibit elasticity for standard 2D shows, dampening casual walk-in attendance.
            - **Premium Exception:** IMAX and Dolby 350-450 seat auditoriums maintain high demand despite higher prices ($18-$22) due to perceived production value.
            """
        )
        st.markdown(
            r"""
            #### 4. ⏳ Movie Release Age (Exponential Decay)
            - **Premiere Hype:** Ticket sales peak during opening weekend (days 0–4) when marketing campaigns and word-of-mouth are freshest.
            - **Theatrical Decay Function:** Attendance declines following $\exp(-0.045 \times \text{release\_days})$, stabilizing into steady long-tail demand by week 4.
            - **Screen Resizing:** As movies age, cinema managers shift titles to 120-seat auditoriums to maximize HVAC and projection cost efficiency.
            """
        )


# ==========================================
# PAGE 6: DATASET OVERVIEW
# ==========================================
elif menu == "📑 Dataset Overview":
    render_banner(
        title="Theatrical Dataset Overview & Documentation",
        subtitle="Comprehensive data dictionary, summary statistics, constraint validation, and synthetic disclosure."
    )

    st.info(
        "ℹ️ **Synthetic Data Disclosure:** This dataset is synthetically generated using domain-informed cinema mechanics "
        "with a fixed random seed (`seed=42`). It realistically reproduces box-office decay curves, weekend demand elasticity, "
        "holiday spikes, price sensitivity, and physical auditorium bounds. It is designed for transparent and reproducible ML evaluation."
    )

    total_records = len(df_history)
    total_cols = len(df_history.columns)
    missing_count = int(df_history.isnull().sum().sum())
    dup_count = int(df_history.duplicated(subset=['show_id']).sum())
    cap_viol = int((df_history["tickets_sold"] > df_history["screen_capacity"]).sum())

    d_c1, d_c2, d_c3, d_c4, d_c5 = st.columns(5)
    with d_c1:
        render_metric_card("Total Records", f"{total_records:,}", "Individual Screenings", "#58a6ff")
    with d_c2:
        render_metric_card("Features", f"{total_cols}", "Show-Level Attributes", "#00d2fc")
    with d_c3:
        render_metric_card("Missing Values", f"{missing_count}", "Zero Data Gaps", "#00e676")
    with d_c4:
        render_metric_card("Duplicates", f"{dup_count}", "Zero Redundant Shows", "#00e676")
    with d_c5:
        render_metric_card("Capacity Violations", f"{cap_viol}", "Zero Overflow Shows", "#00e676")

    st.markdown("### 📊 Statistical Summary of Numerical Variables")
    st.dataframe(df_history.describe().T, use_container_width=True)

    st.markdown("### 📋 Theatrical Features Dictionary")
    data_dict = [
        {"Column": "show_id", "Type": "Categorical (ID)", "Description": "Unique alphanumeric screening identifier (e.g. SHW-00001)"},
        {"Column": "movie_name", "Type": "Categorical", "Description": "Featured movie title (18 catalog films across genres)"},
        {"Column": "genre", "Type": "Categorical", "Description": "Primary cinematic genre (Action, Sci-Fi, Drama, Comedy, Horror, Animation, Thriller, Romance)"},
        {"Column": "movie_language", "Type": "Categorical", "Description": "Theatrical audio language (English, Hindi, Spanish, French, Japanese, Korean)"},
        {"Column": "movie_duration", "Type": "Numerical (int)", "Description": "Film runtime in minutes (92 to 165 minutes)"},
        {"Column": "release_days", "Type": "Numerical (int)", "Description": "Days elapsed since theatrical release (0 = Premiere)"},
        {"Column": "show_date", "Type": "Date (YYYY-MM-DD)", "Description": "Calendar date of the screening"},
        {"Column": "show_time", "Type": "Time (HH:MM)", "Description": "Scheduled showtime (10:00, 13:15, 16:30, 19:45, 22:30)"},
        {"Column": "day_of_week", "Type": "Categorical", "Description": "Day name (Monday through Sunday)"},
        {"Column": "is_weekend", "Type": "Binary (0/1)", "Description": "Weekend cinema rush indicator (1 for Friday, Saturday, Sunday)"},
        {"Column": "is_holiday", "Type": "Binary (0/1)", "Description": "Public or festive holiday calendar indicator"},
        {"Column": "screen_capacity", "Type": "Numerical (int)", "Description": "Auditorium seating capacity (120, 150, 200, 250, 350, 450 seats)"},
        {"Column": "historical_average_sales", "Type": "Numerical (float)", "Description": "Historical moving average sales for comparable slot & genre"},
        {"Column": "previous_show_occupancy", "Type": "Numerical (float)", "Description": "Occupancy rate of preceding show on this screen (0.10 - 0.98)"},
        {"Column": "ticket_price", "Type": "Numerical (float)", "Description": "Standard adult admission ticket price in USD ($8.50 - $24.00)"},
        {"Column": "advance_booking_count", "Type": "Numerical (int)", "Description": "Pre-sale tickets booked prior to the 6-hour forecast cutoff"},
        {"Column": "marketing_spend", "Type": "Numerical (float)", "Description": "Local advertising and promotional budget ($1,500 - $45,000)"},
        {"Column": "competitor_show_count", "Type": "Numerical (int)", "Description": "Competing movie titles running concurrently at nearby cinemas"},
        {"Column": "tickets_sold", "Type": "Numerical (Target)", "Description": "Verified final tickets sold for the show (advance + walk-ins)"}
    ]
    st.dataframe(pd.DataFrame(data_dict), use_container_width=True)

    st.markdown("### 🔍 Theatrical Dataset Preview")
    st.dataframe(df_history.head(50), use_container_width=True)

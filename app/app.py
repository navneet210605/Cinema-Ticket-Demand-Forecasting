"""
Cinema Ticket Demand Forecasting - Streamlit Web Application.

Multi-page production dashboard featuring:
1. Executive Dashboard (KPIs, historical show trends, high/low demand breakdown)
2. Demand Prediction Studio (Real-time live ML inference using saved Joblib model)
3. Deep Analytics & Trends (Interactive Plotly visualizations)
4. Model Performance & Explainability (Metrics, residual inspection, feature importance)
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
    page_title="Cinema Ticket Demand Forecasting",
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
        "🧠 Model Performance & Explainability"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="color: #8b949e; font-size: 0.82rem; line-height: 1.5;">
        <b>Model Status:</b> <span style="color: #00e676;">● Active</span><br>
        <b>Champion Algorithm:</b> Gradient Boosting<br>
        <b>Forecast Horizon:</b> T-6 Hours<br>
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
        subtitle="Operational intelligence, screen utilization metrics, and show-level demand forecasting."
    )

    # Top KPI Metrics Cards
    total_shows = len(df_history)
    avg_tickets = round(df_history["tickets_sold"].mean(), 1)
    avg_occ = round(df_history["occupancy_pct"].mean(), 1)
    high_demand_count = (df_history["demand_category"] == "High Demand").sum()
    low_demand_count = (df_history["demand_category"] == "Low Demand").sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric_card("Total Shows", f"{total_shows:,}", "Historical Records")
    with col2:
        render_metric_card("Avg Tickets Sold", f"{avg_tickets}", "Tickets / Show", "#58a6ff")
    with col3:
        render_metric_card("Avg Occupancy", f"{avg_occ}%", "Seat Utilization", "#e50914")
    with col4:
        render_metric_card("High-Demand Shows", f"{high_demand_count:,}", f"{round(high_demand_count/total_shows*100, 1)}% of shows", "#ff6b6b")
    with col5:
        render_metric_card("Low-Demand Shows", f"{low_demand_count:,}", f"{round(low_demand_count/total_shows*100, 1)}% of shows", "#f5a623")

    st.markdown("### 📈 Historical Ticket Sales & Occupancy Trend")

    # Aggregate daily trend
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
        name="Total Tickets Sold",
        line=dict(color="#00D2FC", width=2.5)
    ))
    fig_trend.add_trace(go.Scatter(
        x=daily_agg["show_date"],
        y=daily_agg["avg_occupancy"] * 25, # Scaled for secondary visualization
        mode="lines",
        name="Avg Occupancy Trend (Index)",
        line=dict(color="#FFB703", width=1.8, dash="dot")
    ))
    fig_trend.update_layout(
        **PLOTLY_DARK_LAYOUT,
        title="Daily Cinema Ticket Demand Trajectory",
        xaxis_title="Show Date",
        yaxis_title="Aggregate Tickets Sold",
        height=380
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Secondary Columns: Demand Distribution & Recent Shows
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

    # Data Filter and Table
    st.markdown("### 🔍 Historical Show Schedule Explorer")
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
        subtitle="Generate live, validated demand forecasts using the champion trained Machine Learning pipeline."
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
            show_time = st.selectbox("Show Time", ["10:00", "13:15", "16:30", "19:45", "22:30"], index=3)
            screen_capacity = st.selectbox("Auditorium Capacity (Seats)", [120, 150, 200, 250, 350, 450], index=3)
            is_weekend_custom = st.selectbox("Weekend Pricing / Schedule?", [1, 0], index=0, format_func=lambda x: "Yes (Weekend)" if x == 1 else "No (Weekday)")
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
            render_metric_card("Remaining Seats", f"{pred_res['remaining_seats']}", "Walk-in availability", "#f5a623")
        with r_col4:
            render_metric_card("Pre-Sale Conversion", f"{round(pred_res['advance_booking_count']/pred_res['predicted_tickets_sold']*100, 1)}%", f"{pred_res['advance_booking_count']} advance bookings", "#d2a8ff")

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

        st.markdown("#### 💡 Operational & Managerial Action Items")
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        with rec_col1:
            st.markdown(
                f"""
                <div class="recom-card" style="border-left-color: #00e676;">
                    <b>👥 Staffing Guidance:</b><br>{pred_res['recommended_staffing']}
                </div>
                """,
                unsafe_allow_html=True
            )
        with rec_col2:
            st.markdown(
                f"""
                <div class="recom-card" style="border-left-color: #e50914;">
                    <b>🏛️ Screen Allocation:</b><br>{pred_res['capacity_action']}
                </div>
                """,
                unsafe_allow_html=True
            )
        with rec_col3:
            st.markdown(
                f"""
                <div class="recom-card" style="border-left-color: #f5a623;">
                    <b>💰 Dynamic Pricing:</b><br>{pred_res['pricing_recommendation']}
                </div>
                """,
                unsafe_allow_html=True
            )


# ==========================================
# PAGE 3: DEEP ANALYTICS & TRENDS
# ==========================================
elif menu == "📊 Deep Analytics & Trends":
    render_banner(
        title="Cinema Demand Analytics & Exploratory Insights",
        subtitle="Deep dive into demand drivers: genre elasticity, showtime curves, and temporal dynamics."
    )

    tab1, tab2, tab3 = st.tabs(["🎭 Genre & Schedule Dynamics", "📅 Temporal & Weekend Effects", "🎯 Actual vs Predicted On Test"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Average Demand by Movie Genre")
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
            st.markdown("#### Demand Curve Across Showtimes")
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

    with tab2:
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
            fig_box.update_layout(**PLOTLY_DARK_LAYOUT, height=360, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        with c2:
            st.markdown("#### Price Elasticity: Ticket Price vs Demand")
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

    with tab3:
        st.markdown("#### 🎯 Held-out Test Set: Actual vs Predicted Demand")
        if comp_data and "test_actuals" in comp_data:
            best_model_name = comp_data["best_model_name"]
            y_act = comp_data["test_actuals"]
            y_pred = comp_data["test_predictions"][best_model_name]
            
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
            min_val = min(min(y_act), min(y_pred))
            max_val = max(max(y_act), max(y_pred))
            fig_parity.add_shape(
                type="line",
                x0=min_val, y0=min_val,
                x1=max_val, y1=max_val,
                line=dict(color="#FF3838", width=2, dash="dash")
            )
            fig_parity.update_layout(
                **PLOTLY_DARK_LAYOUT,
                title=f"Prediction Parity Plot ({best_model_name} on Held-Out Test)",
                height=450
            )
            st.plotly_chart(fig_parity, use_container_width=True)
        else:
            st.info("Run model training to populate test evaluation parity chart.")


# ==========================================
# PAGE 4: MODEL PERFORMANCE & EXPLAINABILITY
# ==========================================
elif menu == "🧠 Model Performance & Explainability":
    render_banner(
        title="Model Performance & Explainability",
        subtitle="Empirical comparison of regression algorithms, test error metrics, and feature attribution."
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

    # Feature Importance Section
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

    # Architectural Discussion & Limitations
    st.markdown("---")
    st.markdown("### 📖 Model Architecture, Cold-Start Strategy & System Limitations")

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(
            """
            #### ❄️ Cold-Start Handling Strategy
            When a brand-new movie premiere is scheduled (`release_days = 0`) with zero prior sales at the theater:
            1. **Genre & Language Anchoring:** The pipeline relies on the learned base appeal of Action, Sci-Fi, or Animation titles.
            2. **T-6h Advance Booking Anchor:** Advance booking velocity provides real pre-sales intent even without historical sales.
            3. **Heuristic Imputation:** Missing `historical_average_sales` is dynamically replaced with the theater's baseline slot occupancy (55% of capacity).
            4. **Marketing Spend Coefficient:** Premiere buzz is captured via logarithmic marketing budget features.
            """
        )

    with col_info2:
        st.markdown(
            """
            #### ⚠️ Model Limitations & Boundary Conditions
            1. **Exogenous Shocks:** Extreme weather (blizzards, hurricanes) or transit disruptions are not accounted for in standard sales features.
            2. **Viral Social Media Spikes:** Organic virality (TikTok / meme trends) can cause sudden surges unpredicted by linear marketing spend.
            3. **Auditorium Capacity Ceiling:** Predictions are strictly capped at `screen_capacity`; no overbooking is permitted.
            4. **Chronological Splitting:** The pipeline uses strict temporal train/val/test splits to prevent lookahead bias.
            """
        )

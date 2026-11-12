"""
Cinema-Themed Dark Styling and Plotly Templates for Streamlit UI.
"""

CINEMA_CSS = """
<style>
    /* Dark Velvet & Charcoal Theme */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Top Header Branding */
    .cinema-banner {
        background: linear-gradient(135deg, #1f1b2e 0%, #161b22 50%, #1e1322 100%);
        border: 1px solid #30363d;
        border-left: 6px solid #e50914;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    .cinema-banner h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .cinema-banner p {
        color: #8b949e;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    .metric-title {
        color: #8b949e;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .metric-subtext {
        color: #7ee787;
        font-size: 0.82rem;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Demand Badges */
    .badge-high {
        background-color: rgba(229, 9, 20, 0.18);
        color: #ff6b6b;
        border: 1px solid #e50914;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-medium {
        background-color: rgba(245, 166, 35, 0.18);
        color: #f5a623;
        border: 1px solid #f5a623;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-low {
        background-color: rgba(0, 230, 118, 0.18);
        color: #00e676;
        border: 1px solid #00e676;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }

    /* Result Highlight Box */
    .prediction-box {
        background: linear-gradient(145deg, #1c2128 0%, #161b22 100%);
        border: 1px solid #444c56;
        border-radius: 12px;
        padding: 24px;
        margin-top: 15px;
        margin-bottom: 20px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    }

    /* Recommendation Cards */
    .recom-card {
        background: #161b22;
        border-left: 4px solid #58a6ff;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 10px;
        color: #c9d1d9;
        font-size: 0.92rem;
    }

    /* Sidebar Tweaks */
    section[data-testid="stSidebar"] {
        background-color: #090d13 !important;
        border-right: 1px solid #21262d !important;
    }
</style>
"""

PLOTLY_DARK_LAYOUT = {
    "paper_bgcolor": "#161b22",
    "plot_bgcolor": "#161b22",
    "font": {"color": "#c9d1d9", "family": "Inter, sans-serif"},
    "xaxis": {
        "gridcolor": "#21262d",
        "zerolinecolor": "#30363d",
        "showline": True,
        "linecolor": "#30363d"
    },
    "yaxis": {
        "gridcolor": "#21262d",
        "zerolinecolor": "#30363d",
        "showline": True,
        "linecolor": "#30363d"
    },
    "legend": {
        "bgcolor": "rgba(22, 27, 34, 0.8)",
        "bordercolor": "#30363d",
        "borderwidth": 1
    },
    "margin": {"l": 40, "r": 30, "t": 50, "b": 40}
}

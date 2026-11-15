"""
Reusable UI Components for Cinema Demand Streamlit Dashboard.
"""

import streamlit as st


def render_banner(title: str, subtitle: str):
    """Render cinematic top banner."""
    st.markdown(
        f"""
        <div class="cinema-banner">
            <h1>🎬 {title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_metric_card(title: str, value: str, subtext: str = "", delta_color: str = "#7ee787"):
    """Render styled dark-mode metric card with dynamic font sizing."""
    val_str = str(value)
    is_num = val_str.replace(',', '').replace('%', '').replace('.', '').replace('$', '').isdigit()
    val_size = "1.85rem" if is_num or len(val_str) <= 8 else "1.32rem"
    
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="font-size: {val_size}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{value}">{value}</div>
            <div class="metric-subtext" style="color: {delta_color};">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_demand_badge(category: str) -> str:
    """Return HTML string for demand category badge."""
    if "High" in category:
        return f'<span class="badge-high">🔥 {category}</span>'
    elif "Medium" in category:
        return f'<span class="badge-medium">⚡ {category}</span>'
    else:
        return f'<span class="badge-low">🟢 {category}</span>'

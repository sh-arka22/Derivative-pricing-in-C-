"""
QuantPricer Interactive Dashboard
=================================
Multi-page Streamlit app for real-time derivatives analysis.

Launch: streamlit run dashboard/app.py
   or:  make dashboard
"""

import streamlit as st

st.set_page_config(
    page_title="QuantPricer Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("QuantPricer Dashboard")
st.markdown("""
**Multi-Model Derivatives Pricing Engine** — Interactive Analysis

Select a page from the sidebar to explore:

| Page | Description |
|------|-------------|
| **Vol Surface** | Interactive 3D implied volatility surface (model + real market) |
| **Greeks Explorer** | All 5 Greeks with real-time parameter sliders |
| **Model Comparison** | BS vs Heston vs Merton implied vol smiles |
| **Yield Curve** | Live Treasury curve + Nelson-Siegel fit |
| **Barrier Analysis** | Barrier option pricing with interactive barrier level |
| **Correlation** | Multi-asset correlation trade explorer |
| **Risk Dashboard** | VaR, CVaR, portfolio risk analysis |
| **Market Data** | Real options chains from Yahoo Finance |

---
*Built with C++ (pricing engine) + Python (Streamlit + Plotly)*
""")

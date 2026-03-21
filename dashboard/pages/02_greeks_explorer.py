"""Interactive Greeks Explorer — all 5 Greeks with real-time sliders"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from engine.pricer import (bs_call, bs_put, bs_delta_call, bs_delta_put,
                            bs_gamma, bs_vega, bs_theta_call, bs_theta_put,
                            bs_rho_call, bs_rho_put)

st.header("Greeks Explorer")

col1, col2 = st.columns([1, 4])

with col1:
    st.markdown("**Parameters**")
    K = st.slider("Strike (K)", 50.0, 150.0, 100.0, 1.0)
    r = st.slider("Risk-free rate (%)", 0.0, 15.0, 5.0, 0.5) / 100
    T = st.slider("Time to maturity (years)", 0.05, 3.0, 1.0, 0.05)
    sigma = st.slider("Volatility (%)", 5.0, 80.0, 20.0, 1.0) / 100
    opt_type = st.radio("Option Type", ["Call", "Put"])

with col2:
    S = np.linspace(50, 150, 200)
    is_call = opt_type == "Call"

    # Compute all Greeks
    if is_call:
        price = bs_call(S, K, r, T, sigma)
        delta = bs_delta_call(S, K, r, T, sigma)
        theta = bs_theta_call(S, K, r, T, sigma)
        rho = bs_rho_call(S, K, r, T, sigma)
    else:
        price = bs_put(S, K, r, T, sigma)
        delta = bs_delta_put(S, K, r, T, sigma)
        theta = bs_theta_put(S, K, r, T, sigma)
        rho = bs_rho_put(S, K, r, T, sigma)

    gamma = bs_gamma(S, K, r, T, sigma)
    vega = bs_vega(S, K, r, T, sigma)

    # 2x3 subplot grid
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=("Price", "Delta", "Gamma",
                                        "Vega", "Theta", "Rho"))

    accent = "#00d4ff" if is_call else "#ff6b6b"

    fig.add_trace(go.Scatter(x=S, y=price, line=dict(color=accent), name="Price",
                             fill='tozeroy', fillcolor=f'rgba(0,212,255,0.1)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=S, y=delta, line=dict(color=accent), name="Delta"), row=1, col=2)
    fig.add_trace(go.Scatter(x=S, y=gamma, line=dict(color="#51cf66"), name="Gamma",
                             fill='tozeroy', fillcolor='rgba(81,207,102,0.1)'), row=1, col=3)
    fig.add_trace(go.Scatter(x=S, y=vega, line=dict(color="#ffd43b"), name="Vega",
                             fill='tozeroy', fillcolor='rgba(255,212,59,0.1)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=S, y=theta, line=dict(color="#cc5de8"), name="Theta"), row=2, col=2)
    fig.add_trace(go.Scatter(x=S, y=rho, line=dict(color="#ff922b"), name="Rho"), row=2, col=3)

    # Add strike line to each subplot
    for row in range(1, 3):
        for col in range(1, 4):
            fig.add_vline(x=K, line_dash="dash", line_color="white", opacity=0.2,
                         row=row, col=col)

    fig.update_layout(
        height=600, template='plotly_dark', showlegend=False,
        title=f'{opt_type} Greeks | K={K:.0f}, r={r*100:.1f}%, T={T:.2f}y, σ={sigma*100:.0f}%'
    )
    fig.update_xaxes(title_text="Spot", row=2)
    st.plotly_chart(fig, use_container_width=True)

    # Key values at ATM
    st.markdown("**ATM Values (S = K):**")
    atm_cols = st.columns(6)
    atm_price = bs_call(K, K, r, T, sigma) if is_call else bs_put(K, K, r, T, sigma)
    atm_delta = bs_delta_call(K, K, r, T, sigma) if is_call else bs_delta_put(K, K, r, T, sigma)
    atm_gamma = bs_gamma(K, K, r, T, sigma)
    atm_vega = bs_vega(K, K, r, T, sigma)
    atm_theta = bs_theta_call(K, K, r, T, sigma) if is_call else bs_theta_put(K, K, r, T, sigma)
    atm_rho = bs_rho_call(K, K, r, T, sigma) if is_call else bs_rho_put(K, K, r, T, sigma)

    atm_cols[0].metric("Price", f"${atm_price:.2f}")
    atm_cols[1].metric("Delta", f"{atm_delta:.4f}")
    atm_cols[2].metric("Gamma", f"{atm_gamma:.4f}")
    atm_cols[3].metric("Vega", f"{atm_vega:.2f}")
    atm_cols[4].metric("Theta", f"{atm_theta:.2f}")
    atm_cols[5].metric("Rho", f"{atm_rho:.2f}")

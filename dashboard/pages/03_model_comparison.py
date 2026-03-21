"""Model Comparison — BS vs Heston vs Merton using C++ MC engine"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from engine.pricer import bs_call, cpp_available, heston_smile, merton_smile

st.header("Model Comparison — BS vs Heston vs Merton")

col1, col2 = st.columns([1, 3])

with col1:
    S = st.number_input("Spot", value=100.0)
    r = st.slider("Rate (%)", 0.0, 10.0, 5.0) / 100
    T = st.slider("Maturity", 0.1, 3.0, 1.0, 0.1)
    sigma = st.slider("BS flat vol (%)", 5, 60, 20) / 100
    num_paths = st.select_slider("MC Paths", [10000, 30000, 50000, 80000, 150000], value=50000)

    st.markdown("**Heston Parameters**")
    h_v0 = st.slider("v0", 0.01, 0.16, 0.04, 0.01)
    h_kappa = st.slider("kappa", 0.1, 5.0, 2.0, 0.1)
    h_theta = st.slider("theta", 0.01, 0.16, 0.04, 0.01)
    h_xi = st.slider("xi (vol-of-vol)", 0.05, 0.8, 0.3, 0.05)
    h_rho = st.slider("rho", -0.95, 0.5, -0.7, 0.05)

    st.markdown("**Merton Parameters**")
    m_lambda = st.slider("Jump intensity", 0.0, 5.0, 1.0, 0.1)
    m_mu_j = st.slider("Jump mean", -0.3, 0.1, -0.1, 0.01)
    m_sigma_j = st.slider("Jump vol", 0.01, 0.4, 0.15, 0.01)

with col2:
    use_cpp = cpp_available()

    if st.button("Compute (C++ MC Engine)" if use_cpp else "Compute (C++ not built — run `make release`)"):
        if not use_cpp:
            st.error("C++ pricer not found. Run `make release` in the project root.")
            st.stop()

        with st.spinner("Running Heston + Merton MC in C++..."):
            heston_df = heston_smile(S0=S, K_min=70, K_max=130, K_step=2, r=r, T=T,
                                     v0=h_v0, kappa=h_kappa, theta=h_theta,
                                     xi=h_xi, rho=h_rho, num_paths=num_paths)
            merton_df = merton_smile(S0=S, K_min=70, K_max=130, K_step=2, r=r, T=T,
                                     sigma=sigma, lam=m_lambda, mu_j=m_mu_j,
                                     sigma_j=m_sigma_j, num_paths=num_paths)
            st.session_state['heston_df'] = heston_df
            st.session_state['merton_df'] = merton_df

    if 'heston_df' in st.session_state:
        heston_df = st.session_state['heston_df']
        merton_df = st.session_state['merton_df']

        # Implied Vol Smile
        fig = go.Figure()
        fig.add_hline(y=sigma*100, line_dash="dash", line_color="white", opacity=0.3,
                      annotation_text=f"BS flat vol = {sigma*100:.0f}%")
        fig.add_trace(go.Scatter(x=heston_df['Strike'], y=heston_df['ImpliedVol']*100,
                                 mode='lines+markers', name='Heston MC',
                                 line=dict(color='#00d4ff', width=2.5), marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=merton_df['Strike'], y=merton_df['ImpliedVol']*100,
                                 mode='lines+markers', name='Merton MC',
                                 line=dict(color='#ff6b6b', width=2.5), marker=dict(size=5)))
        fig.add_vline(x=S, line_dash="dash", line_color="white", opacity=0.2, annotation_text="ATM")
        fig.update_layout(
            xaxis_title='Strike', yaxis_title='Implied Volatility (%)',
            title='Implied Vol Smile — C++ Monte Carlo',
            height=500, template='plotly_dark',
        )
        st.plotly_chart(fig, use_container_width=True)

        # Price comparison
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=heston_df['Strike'], y=heston_df['BSPrice'],
                                  mode='lines', name='Black-Scholes',
                                  line=dict(color='white', dash='dash', width=1.5)))
        fig2.add_trace(go.Scatter(x=heston_df['Strike'], y=heston_df['HestonPrice'],
                                  mode='lines+markers', name='Heston',
                                  line=dict(color='#00d4ff'), marker=dict(size=4)))
        fig2.add_trace(go.Scatter(x=merton_df['Strike'], y=merton_df['MertonPrice'],
                                  mode='lines+markers', name='Merton',
                                  line=dict(color='#ff6b6b'), marker=dict(size=4)))
        fig2.update_layout(
            xaxis_title='Strike', yaxis_title='Call Price ($)',
            title='Call Prices — C++ Monte Carlo',
            height=400, template='plotly_dark',
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.success(f"Computed with {num_paths:,} MC paths per strike point via C++ engine")
    else:
        st.info("Click 'Compute' to run the C++ Monte Carlo engine for Heston and Merton pricing.")

"""Correlation Trade Explorer — C++ multi-asset MC engine"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from engine.pricer import bs_call, cpp_available, multi_asset_corr

st.header("Correlation Trade Explorer")
st.markdown("How correlation affects multi-asset option prices — computed via C++ Cholesky MC.")

col1, col2 = st.columns([1, 3])

with col1:
    S = st.number_input("Spot (each asset)", value=100.0)
    K = st.number_input("Strike", value=100.0)
    r = st.slider("Rate (%)", 0.0, 10.0, 5.0) / 100
    T = st.slider("Maturity", 0.1, 3.0, 1.0, 0.1)
    sigma = st.slider("Vol (%) each asset", 5, 60, 20) / 100
    N = st.slider("Number of assets", 2, 10, 2)
    num_paths = st.select_slider("MC Paths", [10000, 30000, 50000, 80000], value=50000)

with col2:
    use_cpp = cpp_available()

    if st.button("Compute via C++ MC" if use_cpp else "C++ not built — run `make release`"):
        if not use_cpp:
            st.error("Run `make release` to build the C++ pricer.")
            st.stop()

        with st.spinner(f"Running {N}-asset correlated MC in C++..."):
            df = multi_asset_corr(S0=S, K=K, r=r, T=T, sigma=sigma, N=N,
                                  rho_min=-0.4, rho_max=0.99, rho_step=0.04,
                                  num_paths=num_paths)
            st.session_state['corr_df'] = df

    if 'corr_df' in st.session_state:
        df = st.session_state['corr_df']
        bs_single = float(bs_call(S, K, r, T, sigma))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Rho'], y=df['BestOfCall'], name='Best-of Call',
                                 mode='lines+markers', line=dict(color='#00d4ff', width=2.5),
                                 marker=dict(size=4)))
        fig.add_trace(go.Scatter(x=df['Rho'], y=df['BasketCall'], name='Basket Call',
                                 mode='lines+markers', line=dict(color='#51cf66', width=2.5),
                                 marker=dict(size=4)))
        fig.add_trace(go.Scatter(x=df['Rho'], y=df['WorstOfCall'], name='Worst-of Call',
                                 mode='lines+markers', line=dict(color='#ff6b6b', width=2.5),
                                 marker=dict(size=4)))
        fig.add_hline(y=bs_single, line_dash="dash", line_color="white", opacity=0.3,
                      annotation_text=f"Single asset BS = {bs_single:.2f}")
        fig.update_layout(
            xaxis_title='Correlation (rho)', yaxis_title='Option Price ($)',
            title=f'Correlation Trade — {N} Assets, C++ Cholesky MC ({num_paths:,} paths)',
            height=500, template='plotly_dark',
        )
        st.plotly_chart(fig, use_container_width=True)

        # Basket vol
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['Rho'], y=df['BasketVol']*100,
                                  line=dict(color='#cc5de8', width=2.5),
                                  fill='tozeroy', fillcolor='rgba(204,93,232,0.1)'))
        fig2.add_hline(y=sigma*100, line_dash="dash", line_color="white", opacity=0.3,
                       annotation_text=f"Individual vol = {sigma*100:.0f}%")
        fig2.update_layout(
            xaxis_title='Correlation (rho)', yaxis_title='Basket Volatility (%)',
            title='Diversification Effect',
            height=400, template='plotly_dark',
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.success(f"Computed with C++ Cholesky-decomposed {N}-asset MC engine ({num_paths:,} paths per rho point)")
    else:
        st.info("Click 'Compute' to run the C++ multi-asset Monte Carlo engine.")

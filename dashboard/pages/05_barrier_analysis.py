"""Barrier Option Analysis — C++ analytic pricing engine"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from engine.pricer import bs_call, cpp_available, barrier_curve

st.header("Barrier Option Analysis")

col1, col2 = st.columns([1, 3])

with col1:
    S = st.number_input("Spot", value=100.0)
    K = st.number_input("Strike", value=100.0)
    r = st.slider("Rate (%)", 0.0, 10.0, 5.0) / 100
    T = st.slider("Maturity", 0.1, 3.0, 1.0, 0.1)
    sigma = st.slider("Vol (%)", 5, 60, 20) / 100

with col2:
    use_cpp = cpp_available()

    if st.button("Compute via C++ Engine" if use_cpp else "C++ not built — run `make release`"):
        if not use_cpp:
            st.error("Run `make release` to build the C++ pricer.")
            st.stop()

        with st.spinner("Computing barrier prices (C++ Haug analytic + MC)..."):
            df = barrier_curve(S0=S, K=K, r=r, T=T, sigma=sigma, H_min=50, H_max=99, H_step=1)
            st.session_state['barrier_df'] = df

    if 'barrier_df' in st.session_state:
        df = st.session_state['barrier_df']
        vanilla = float(df['Vanilla'].iloc[0])

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Barrier Option Prices (C++ Haug Analytic)",
                                            "Knock-Out Probability (C++ MC)"))

        fig.add_trace(go.Scatter(x=df['Barrier'], y=df['DOCall'], name='Down-and-Out Call',
                                 line=dict(color='#00d4ff', width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Barrier'], y=df['DICall'], name='Down-and-In Call',
                                 line=dict(color='#ff6b6b', width=2.5)), row=1, col=1)
        fig.add_hline(y=vanilla, line_dash="dash", line_color="white", opacity=0.3,
                      annotation_text=f"Vanilla={vanilla:.2f}", row=1, col=1)

        fig.add_trace(go.Scatter(x=df['Barrier'], y=df['KnockPct'], name='Knock %',
                                 line=dict(color='#ffd43b', width=2.5),
                                 fill='tozeroy', fillcolor='rgba(255,212,59,0.1)'), row=1, col=2)

        fig.update_layout(height=500, template='plotly_dark',
                          title=f'Down Barrier Analysis | S={S}, K={K}, σ={sigma*100:.0f}%')
        fig.update_xaxes(title_text="Barrier Level")
        st.plotly_chart(fig, use_container_width=True)

        st.info("Prices from C++ Haug (2007) analytic formulas. Knock probability from C++ MC (50k paths, 252 steps).")
    else:
        st.info("Click 'Compute' to price barrier options using the C++ engine.")

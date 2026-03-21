"""Live Yield Curve — Treasury rates + Nelson-Siegel fit"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from engine.pricer import nelson_siegel
from engine.market_data import get_treasury_curve

st.header("Yield Curve Analysis")

tab1, tab2 = st.tabs(["Live Treasury Curve", "Nelson-Siegel Explorer"])

with tab1:
    st.subheader("US Treasury Yield Curve")

    fred_key = st.text_input("FRED API Key (optional — leave blank for sample data)",
                             type="password", key="fred_key")

    if st.button("Fetch Treasury Curve", key="fetch_curve"):
        with st.spinner("Fetching from FRED..."):
            curve = get_treasury_curve(fred_key if fred_key else None)
            st.session_state['treasury_curve'] = curve

    if 'treasury_curve' in st.session_state:
        curve = st.session_state['treasury_curve']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curve['tenor'], y=curve['rate'] * 100,
            mode='lines+markers', name='Treasury Curve',
            line=dict(color='#00d4ff', width=3),
            marker=dict(size=8),
        ))

        # Fit Nelson-Siegel
        from scipy.optimize import minimize
        tenors_arr = curve['tenor'].values
        rates_arr = curve['rate'].values

        def ns_error(params):
            b0, b1, b2, tau = params
            if tau < 0.1: return 1e10
            model = nelson_siegel(tenors_arr, b0, b1, b2, tau)
            return np.sum((model - rates_arr)**2)

        res = minimize(ns_error, [0.04, -0.01, 0.01, 2.0], method='Nelder-Mead')
        b0, b1, b2, tau = res.x

        t_fine = np.linspace(0.05, 30, 200)
        ns_fit = nelson_siegel(t_fine, b0, b1, b2, tau)

        fig.add_trace(go.Scatter(
            x=t_fine, y=ns_fit * 100,
            mode='lines', name=f'Nelson-Siegel fit',
            line=dict(color='#ff6b6b', dash='dash'),
        ))

        fig.update_layout(
            xaxis_title='Maturity (years)', yaxis_title='Yield (%)',
            title='US Treasury Yield Curve',
            height=500, template='plotly_dark',
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("beta0 (level)", f"{b0*100:.2f}%")
        col2.metric("beta1 (slope)", f"{b1*100:.2f}%")
        col3.metric("beta2 (curve)", f"{b2*100:.2f}%")
        col4.metric("tau (decay)", f"{tau:.2f}")

        # Spread analysis
        r_2y = float(np.interp(2, curve['tenor'], curve['rate']))
        r_10y = float(np.interp(10, curve['tenor'], curve['rate']))
        spread = (r_10y - r_2y) * 10000
        st.metric("10Y - 2Y Spread", f"{spread:.0f} bps",
                  delta="Inverted (recession signal)" if spread < 0 else "Normal")
    else:
        st.info("Click 'Fetch Treasury Curve' to load data (works without API key)")

with tab2:
    st.subheader("Nelson-Siegel Parameter Explorer")

    col1, col2 = st.columns([1, 3])
    with col1:
        b0 = st.slider("beta0 (level %)", 1.0, 8.0, 4.5, 0.1) / 100
        b1 = st.slider("beta1 (slope %)", -4.0, 4.0, -1.5, 0.1) / 100
        b2 = st.slider("beta2 (curve %)", -4.0, 4.0, 2.0, 0.1) / 100
        tau = st.slider("tau (decay)", 0.5, 10.0, 2.0, 0.1)

        st.markdown(f"**Short rate**: {(b0+b1)*100:.2f}%")
        st.markdown(f"**Long rate**: {b0*100:.2f}%")

    with col2:
        t = np.linspace(0.05, 30, 200)
        rates = nelson_siegel(t, b0, b1, b2, tau)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=rates * 100, line=dict(color='#00d4ff', width=3),
                                 fill='tozeroy', fillcolor='rgba(0,212,255,0.1)'))
        fig.update_layout(
            xaxis_title='Maturity (years)', yaxis_title='Zero Rate (%)',
            title=f'Nelson-Siegel Curve | r(0)={(b0+b1)*100:.1f}%, r(∞)={b0*100:.1f}%',
            height=450, template='plotly_dark',
        )
        st.plotly_chart(fig, use_container_width=True)

"""Interactive 3D Implied Volatility Surface"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from engine.pricer import bs_call, implied_vol

st.header("3D Implied Volatility Surface")

tab1, tab2 = st.tabs(["Model (Heston-like)", "Market Data"])

# ============================================================================
# Tab 1: Model Volatility Surface
# ============================================================================
with tab1:
    st.subheader("Heston-Style Parametric Vol Surface")

    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Surface Parameters**")
        atm_vol = st.slider("ATM Vol (%)", 10, 60, 20, key="atm") / 100
        skew = st.slider("Skew (negative = put skew)", -30, 10, -8, key="skew") / 100
        smile = st.slider("Smile (curvature)", 0, 20, 5, key="smile") / 100
        term_slope = st.slider("Term slope (vol vs T)", -5, 5, -2, key="term") / 100
        spot = st.number_input("Spot", value=100.0, key="spot_model")
        r = st.slider("Risk-free rate (%)", 0.0, 10.0, 5.0, key="r_model") / 100

    with col2:
        # Generate parametric vol surface
        strikes = np.linspace(0.7 * spot, 1.3 * spot, 30)
        maturities = np.linspace(0.1, 2.0, 20)
        K_grid, T_grid = np.meshgrid(strikes, maturities)

        # Parametric model: IV(K,T) = atm + skew*(K/S - 1) + smile*(K/S - 1)^2 + term*(T - 1)
        moneyness = K_grid / spot - 1.0
        IV_grid = (atm_vol
                   + skew * moneyness
                   + smile * moneyness**2
                   + term_slope * (T_grid - 1.0))
        IV_grid = np.clip(IV_grid, 0.05, 1.0)

        fig = go.Figure(data=[go.Surface(
            x=strikes, y=maturities, z=IV_grid * 100,
            colorscale='Plasma', opacity=0.9,
            contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
            hovertemplate='Strike: %{x:.0f}<br>Maturity: %{y:.2f}y<br>IV: %{z:.1f}%<extra></extra>'
        )])
        fig.update_layout(
            scene=dict(
                xaxis_title='Strike',
                yaxis_title='Maturity (years)',
                zaxis_title='Implied Vol (%)',
                camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
            ),
            title='Implied Volatility Surface',
            height=650,
            template='plotly_dark',
        )
        st.plotly_chart(fig, use_container_width=True)

    # Vol smile slices
    st.subheader("Volatility Smile Slices")
    fig2 = go.Figure()
    for i, T in enumerate([0.25, 0.5, 1.0, 1.5, 2.0]):
        idx = np.argmin(np.abs(maturities - T))
        fig2.add_trace(go.Scatter(
            x=strikes, y=IV_grid[idx] * 100,
            mode='lines', name=f'T={T:.2f}y',
        ))
    fig2.add_vline(x=spot, line_dash="dash", line_color="white", opacity=0.3, annotation_text="ATM")
    fig2.update_layout(
        xaxis_title='Strike', yaxis_title='Implied Vol (%)',
        title='Volatility Smile by Maturity',
        height=400, template='plotly_dark',
    )
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================================
# Tab 2: Real Market Vol Surface
# ============================================================================
with tab2:
    st.subheader("Market Implied Vol Surface (Yahoo Finance)")

    ticker = st.text_input("Ticker", value="AAPL", key="mkt_ticker")
    if st.button("Fetch Options Data", key="fetch_btn"):
        with st.spinner(f"Fetching options chain for {ticker}..."):
            try:
                from engine.market_data import get_stock_price, get_options_chain, compute_iv_surface

                spot_mkt = get_stock_price(ticker)
                chain = get_options_chain(ticker, max_expiries=6)

                if chain.empty:
                    st.error("No options data available")
                else:
                    st.success(f"{ticker} spot: ${spot_mkt:.2f} | {len(chain)} options loaded")

                    iv_df = compute_iv_surface(chain, spot_mkt)

                    if not iv_df.empty:
                        # Calls only for cleaner surface
                        calls_iv = iv_df[iv_df['type'] == 'call']

                        # Scatter plot (raw data points)
                        fig3 = go.Figure(data=[go.Scatter3d(
                            x=calls_iv['moneyness'],
                            y=calls_iv['T'],
                            z=calls_iv['iv'] * 100,
                            mode='markers',
                            marker=dict(size=4, color=calls_iv['iv']*100,
                                       colorscale='Plasma', showscale=True,
                                       colorbar=dict(title='IV%')),
                            hovertemplate='K/S: %{x:.2f}<br>T: %{y:.2f}y<br>IV: %{z:.1f}%<extra></extra>'
                        )])
                        fig3.update_layout(
                            scene=dict(
                                xaxis_title='Moneyness (K/S)',
                                yaxis_title='Maturity (years)',
                                zaxis_title='Implied Vol (%)',
                            ),
                            title=f'{ticker} Market Implied Volatility',
                            height=650, template='plotly_dark',
                        )
                        st.plotly_chart(fig3, use_container_width=True)

                        # Smile by expiry (2D)
                        fig4 = go.Figure()
                        for exp in calls_iv['expiry'].unique()[:6]:
                            subset = calls_iv[calls_iv['expiry'] == exp].sort_values('strike')
                            fig4.add_trace(go.Scatter(
                                x=subset['moneyness'], y=subset['iv'] * 100,
                                mode='lines+markers', name=exp, marker=dict(size=4)
                            ))
                        fig4.add_vline(x=1.0, line_dash="dash", line_color="white",
                                      opacity=0.3, annotation_text="ATM")
                        fig4.update_layout(
                            xaxis_title='Moneyness (K/S)',
                            yaxis_title='Implied Vol (%)',
                            title=f'{ticker} Vol Smile by Expiry',
                            height=400, template='plotly_dark',
                        )
                        st.plotly_chart(fig4, use_container_width=True)
                    else:
                        st.warning("Could not compute implied vols")
            except Exception as e:
                st.error(f"Error: {e}")

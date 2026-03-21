"""Risk Dashboard — VaR, CVaR, portfolio risk"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

st.header("Risk Dashboard")

col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("**Portfolio Allocation**")
    w_equity = st.slider("Equity (%)", 0, 100, 60)
    w_bonds = st.slider("Bonds (%)", 0, 100, 30)
    w_commod = 100 - w_equity - w_bonds
    st.markdown(f"Commodities: **{w_commod}%**")

    if w_commod < 0:
        st.error("Weights exceed 100%")
        st.stop()

    st.markdown("**Risk Parameters**")
    vol_eq = st.slider("Equity vol (%)", 5, 50, 15) / 100
    vol_bd = st.slider("Bond vol (%)", 1, 20, 5) / 100
    vol_cm = st.slider("Commodity vol (%)", 5, 50, 20) / 100
    rho_eb = st.slider("Equity-Bond corr", -0.5, 0.9, 0.3, 0.05)

    portfolio_val = st.number_input("Portfolio Value ($)", value=10_000_000, step=1_000_000)
    confidence = st.slider("VaR Confidence (%)", 90.0, 99.9, 99.0, 0.5) / 100

with col2:
    w = np.array([w_equity, w_bonds, w_commod]) / 100
    sigma = np.array([vol_eq, vol_bd, vol_cm])

    # Correlation matrix
    corr = np.array([[1.0, rho_eb, 0.2],
                     [rho_eb, 1.0, -0.1],
                     [0.2, -0.1, 1.0]])

    # Covariance matrix
    cov = np.outer(sigma, sigma) * corr

    # Portfolio vol: sqrt(w' * Sigma * w)
    port_var = w @ cov @ w
    port_vol = np.sqrt(port_var)

    # Parametric VaR & CVaR
    z = norm.ppf(confidence)
    var_pct = z * port_vol
    cvar_pct = port_vol * norm.pdf(z) / (1 - confidence)
    var_dollar = var_pct * portfolio_val
    cvar_dollar = cvar_pct * portfolio_val

    # Display metrics
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Portfolio Vol (daily)", f"{port_vol*100:.2f}%")
    mcol2.metric("Annual Vol", f"{port_vol*np.sqrt(252)*100:.1f}%")
    mcol3.metric(f"VaR ({confidence*100:.1f}%)", f"${var_dollar:,.0f}")
    mcol4.metric(f"CVaR ({confidence*100:.1f}%)", f"${cvar_dollar:,.0f}")

    # P&L distribution
    np.random.seed(42)
    n_sim = 100000
    L = np.linalg.cholesky(cov)
    Z = np.random.randn(n_sim, 3)
    returns = Z @ L.T
    pnl = (returns @ w) * portfolio_val

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pnl, nbinsx=200, name='P&L Distribution',
                               marker_color='#00d4ff', opacity=0.7))
    fig.add_vline(x=-var_dollar, line_color='#ff6b6b', line_width=2,
                  annotation_text=f'VaR: -${var_dollar:,.0f}')
    fig.add_vline(x=-cvar_dollar, line_color='#ffd43b', line_width=2,
                  annotation_text=f'CVaR: -${cvar_dollar:,.0f}')
    fig.update_layout(
        xaxis_title='Daily P&L ($)', yaxis_title='Frequency',
        title=f'Monte Carlo P&L Distribution ({n_sim:,} scenarios)',
        height=450, template='plotly_dark',
    )
    st.plotly_chart(fig, use_container_width=True)

    # VaR sensitivity to confidence level
    alphas = np.linspace(0.9, 0.999, 50)
    vars_pct = [norm.ppf(a) * port_vol * 100 for a in alphas]
    cvars_pct = [port_vol * norm.pdf(norm.ppf(a)) / (1 - a) * 100 for a in alphas]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=alphas * 100, y=vars_pct, name='VaR',
                              line=dict(color='#ff6b6b', width=2.5)))
    fig2.add_trace(go.Scatter(x=alphas * 100, y=cvars_pct, name='CVaR',
                              line=dict(color='#ffd43b', width=2.5)))
    fig2.update_layout(
        xaxis_title='Confidence Level (%)', yaxis_title='Risk (% of portfolio)',
        title='VaR & CVaR vs Confidence Level',
        height=400, template='plotly_dark',
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Risk decomposition
    sigma_w = cov @ w
    marginal_var = z * sigma_w / port_vol
    component_var = w * marginal_var
    total_component = component_var.sum()

    st.subheader("Risk Decomposition (Euler)")
    decomp_data = {
        "Asset": ["Equity", "Bonds", "Commodities", "**Total**"],
        "Weight": [f"{x*100:.0f}%" for x in w] + ["100%"],
        "Component VaR": [f"{x*100:.3f}%" for x in component_var] + [f"{total_component*100:.3f}%"],
        "% of Total": [f"{x/total_component*100:.1f}%" for x in component_var] + ["100%"],
    }
    st.table(decomp_data)

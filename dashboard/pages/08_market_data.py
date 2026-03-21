"""Market Data Explorer — real options chains from Yahoo Finance"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

st.header("Market Data Explorer")
st.markdown("Fetch real options chains and analyze live market data.")

ticker = st.text_input("Ticker Symbol", value="SPY")

if st.button("Fetch Data"):
    with st.spinner(f"Loading {ticker} options data..."):
        try:
            from engine.market_data import get_stock_price, get_options_chain, get_historical_prices

            spot = get_stock_price(ticker)
            chain = get_options_chain(ticker, max_expiries=8)
            hist = get_historical_prices(ticker, period="6mo")

            st.session_state['mkt_spot'] = spot
            st.session_state['mkt_chain'] = chain
            st.session_state['mkt_hist'] = hist
            st.session_state['mkt_ticker'] = ticker
            st.success(f"{ticker}: ${spot:.2f} | {len(chain)} options loaded")
        except Exception as e:
            st.error(f"Error fetching data: {e}")

if 'mkt_chain' in st.session_state:
    spot = st.session_state['mkt_spot']
    chain = st.session_state['mkt_chain']
    hist = st.session_state['mkt_hist']
    tkr = st.session_state['mkt_ticker']

    tab1, tab2, tab3 = st.tabs(["Stock Chart", "Options Chain", "Vol Analysis"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist['Open'], high=hist['High'],
            low=hist['Low'], close=hist['Close'], name=tkr
        ))
        fig.update_layout(
            title=f'{tkr} — 6 Month Price Chart',
            yaxis_title='Price ($)',
            height=500, template='plotly_dark',
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Key stats
        c1, c2, c3, c4 = st.columns(4)
        returns = hist['Close'].pct_change().dropna()
        c1.metric("Current Price", f"${spot:.2f}")
        c2.metric("6M Return", f"{(hist['Close'].iloc[-1]/hist['Close'].iloc[0]-1)*100:.1f}%")
        c3.metric("Realized Vol (ann)", f"{returns.std()*np.sqrt(252)*100:.1f}%")
        c4.metric("6M High", f"${hist['High'].max():.2f}")

    with tab2:
        expiry = st.selectbox("Expiration", chain['expiry'].unique())
        opt_type = st.radio("Type", ["call", "put"], horizontal=True)

        filtered = chain[(chain['expiry'] == expiry) & (chain['type'] == opt_type)]
        filtered = filtered.sort_values('strike')

        st.dataframe(
            filtered[['strike', 'bid', 'ask', 'mid', 'impliedVolatility',
                      'volume', 'openInterest', 'days_to_expiry']].reset_index(drop=True),
            use_container_width=True,
            height=400,
        )

        # Bid-ask spread analysis
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=filtered['strike'], y=(filtered['ask'] - filtered['bid']),
                              marker_color='#ff6b6b', name='Bid-Ask Spread'))
        fig2.add_vline(x=spot, line_dash="dash", line_color="white", opacity=0.3,
                       annotation_text="Spot")
        fig2.update_layout(
            xaxis_title='Strike', yaxis_title='Bid-Ask Spread ($)',
            title=f'{tkr} {expiry} {opt_type.title()} Bid-Ask Spread',
            height=350, template='plotly_dark',
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Implied Vol by Strike")
        calls = chain[(chain['type'] == 'call') & (chain['impliedVolatility'] > 0.01)]

        fig3 = go.Figure()
        for exp in calls['expiry'].unique()[:5]:
            sub = calls[calls['expiry'] == exp].sort_values('strike')
            fig3.add_trace(go.Scatter(
                x=sub['strike'], y=sub['impliedVolatility'] * 100,
                mode='lines+markers', name=exp, marker=dict(size=4)
            ))
        fig3.add_vline(x=spot, line_dash="dash", line_color="white", opacity=0.3,
                       annotation_text="Spot")
        fig3.update_layout(
            xaxis_title='Strike', yaxis_title='Implied Vol (%)',
            title=f'{tkr} Implied Volatility Smile',
            height=450, template='plotly_dark',
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Open Interest analysis
        st.subheader("Open Interest by Strike")
        for exp in chain['expiry'].unique()[:3]:
            exp_data = chain[chain['expiry'] == exp]
            call_oi = exp_data[exp_data['type'] == 'call'][['strike', 'openInterest']].rename(
                columns={'openInterest': 'Call OI'})
            put_oi = exp_data[exp_data['type'] == 'put'][['strike', 'openInterest']].rename(
                columns={'openInterest': 'Put OI'})

            fig4 = go.Figure()
            fig4.add_trace(go.Bar(x=call_oi['strike'], y=call_oi['Call OI'],
                                  name='Call OI', marker_color='#00d4ff', opacity=0.7))
            fig4.add_trace(go.Bar(x=put_oi['strike'], y=put_oi['Put OI'],
                                  name='Put OI', marker_color='#ff6b6b', opacity=0.7))
            fig4.add_vline(x=spot, line_dash="dash", line_color="white", opacity=0.3)
            fig4.update_layout(
                xaxis_title='Strike', yaxis_title='Open Interest',
                title=f'{tkr} Open Interest — {exp}',
                height=350, template='plotly_dark', barmode='group',
            )
            st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("Enter a ticker and click 'Fetch Data' to load real market options data.")

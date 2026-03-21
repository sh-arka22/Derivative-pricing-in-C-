"""
Market Data Module — fetches real market data from free APIs.

Sources:
  - yfinance: stock prices, options chains, historical data
  - FRED: Treasury yield curves (requires free API key)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

# ============================================================================
# Stock & Options Data (yfinance)
# ============================================================================

def get_stock_price(ticker: str) -> float:
    """Get the latest stock price."""
    import yfinance as yf
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d")
    if hist.empty:
        raise ValueError(f"No data for {ticker}")
    return float(hist['Close'].iloc[-1])

def get_options_chain(ticker: str, max_expiries: int = 8) -> pd.DataFrame:
    """
    Fetch options chain for a ticker across multiple expirations.
    Returns DataFrame with columns: strike, expiry, type, bid, ask, mid,
    impliedVolatility, volume, openInterest, days_to_expiry
    """
    import yfinance as yf
    stock = yf.Ticker(ticker)
    expirations = stock.options[:max_expiries]

    all_data = []
    today = datetime.now()

    for exp_str in expirations:
        try:
            chain = stock.option_chain(exp_str)
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            dte = max((exp_date - today).days, 1)

            for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
                subset = df[['strike', 'bid', 'ask', 'impliedVolatility',
                             'volume', 'openInterest']].copy()
                subset['type'] = opt_type
                subset['expiry'] = exp_str
                subset['days_to_expiry'] = dte
                subset['T'] = dte / 365.0
                subset['mid'] = (subset['bid'] + subset['ask']) / 2
                all_data.append(subset)
        except Exception:
            continue

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    # Filter out zero-bid options (illiquid)
    result = result[result['bid'] > 0.01]
    return result

def get_historical_prices(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch historical OHLCV data."""
    import yfinance as yf
    stock = yf.Ticker(ticker)
    return stock.history(period=period)

# ============================================================================
# Treasury Yield Curve (FRED)
# ============================================================================

# FRED series IDs for Treasury constant maturity rates
TREASURY_SERIES = {
    1/12:  "DGS1MO",
    3/12:  "DGS3MO",
    6/12:  "DGS6MO",
    1.0:   "DGS1",
    2.0:   "DGS2",
    3.0:   "DGS3",
    5.0:   "DGS5",
    7.0:   "DGS7",
    10.0:  "DGS10",
    20.0:  "DGS20",
    30.0:  "DGS30",
}

def get_treasury_curve(api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch the latest Treasury yield curve from FRED.
    Returns DataFrame with columns: tenor, rate

    If no API key, returns a representative sample curve.
    """
    if api_key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=api_key)

            tenors = []
            rates = []
            for tenor, series_id in TREASURY_SERIES.items():
                try:
                    data = fred.get_series(series_id)
                    latest = data.dropna().iloc[-1]
                    tenors.append(tenor)
                    rates.append(float(latest) / 100.0)  # Convert % to decimal
                except Exception:
                    continue

            return pd.DataFrame({"tenor": tenors, "rate": rates})
        except ImportError:
            pass

    # Fallback: representative US Treasury curve (approximate)
    return pd.DataFrame({
        "tenor": [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30],
        "rate": [0.0525, 0.0540, 0.0530, 0.0500, 0.0470, 0.0450,
                 0.0435, 0.0430, 0.0425, 0.0450, 0.0445]
    })

# ============================================================================
# Compute Implied Vol Surface from Real Options Data
# ============================================================================

def compute_iv_surface(chain: pd.DataFrame, spot: float, r: float = 0.05) -> pd.DataFrame:
    """
    Compute implied volatility for each option in the chain.
    Uses the mid price and Newton-Raphson IV solver.
    """
    from .pricer import implied_vol

    iv_data = []
    for _, row in chain.iterrows():
        if row['mid'] <= 0.01 or row['T'] < 0.01:
            continue
        try:
            is_call = row['type'] == 'call'
            iv = implied_vol(row['mid'], spot, row['strike'], r, row['T'], is_call=is_call)
            if iv is not None and 0.01 < iv < 3.0 and not np.isnan(iv):
                iv_data.append({
                    'strike': row['strike'],
                    'T': row['T'],
                    'expiry': row['expiry'],
                    'iv': iv,
                    'moneyness': row['strike'] / spot,
                    'type': row['type'],
                    'mid': row['mid'],
                })
        except Exception:
            continue

    return pd.DataFrame(iv_data)

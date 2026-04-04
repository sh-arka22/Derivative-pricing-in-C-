#!/usr/bin/env python3
"""
Download historical OHLCV data from Yahoo Finance.

Usage:
    python3 tools/download_data.py                    # default: AAPL SPY TSLA, 1 year
    python3 tools/download_data.py --symbols AAPL MSFT NVDA
    python3 tools/download_data.py --period 2y        # 2 years of data
    python3 tools/download_data.py --start 2024-01-01 --end 2025-01-01

Output:
    data/AAPL.csv, data/SPY.csv, data/TSLA.csv  (Yahoo Finance format)

CSV columns:
    Date,Open,High,Low,Close,Adj Close,Volume

Requirements:
    pip install yfinance
"""

import argparse
import os
import sys

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)


def download_symbol(symbol: str, output_dir: str, period: str = None,
                    start: str = None, end: str = None) -> bool:
    """Download OHLCV data for one symbol and save as CSV."""
    filepath = os.path.join(output_dir, f"{symbol}.csv")
    print(f"  {symbol}: downloading...", end="", flush=True)

    try:
        ticker = yf.Ticker(symbol)

        if start and end:
            df = ticker.history(start=start, end=end, auto_adjust=False)
        else:
            df = ticker.history(period=period or "1y", auto_adjust=False)

        if df.empty:
            print(f" FAILED (no data returned)")
            return False

        # Drop the 'Dividends' and 'Stock Splits' columns if present
        drop_cols = [c for c in ["Dividends", "Stock Splits", "Capital Gains"] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # Ensure column order matches Yahoo Finance CSV download format
        expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        for col in expected_cols:
            if col not in df.columns:
                if col == "Adj Close":
                    df["Adj Close"] = df["Close"]
                else:
                    print(f" FAILED (missing column: {col})")
                    return False

        df = df[expected_cols]

        # Round prices to 2 decimal places, volume to integer
        for col in ["Open", "High", "Low", "Close", "Adj Close"]:
            df[col] = df[col].round(2)
        df["Volume"] = df["Volume"].astype(int)

        # Normalize dates to clean YYYY-MM-DD format
        df.index = pd.to_datetime(df.index, utc=True).strftime("%Y-%m-%d")
        # Deduplicate dates (rare edge case with yfinance)
        df = df[~df.index.duplicated(keep="first")]

        # Save — index is the date
        df.to_csv(filepath, index=True, index_label="Date")

        bars = len(df)
        date_range = f"{df.index[0]} to {df.index[-1]}"
        last_close = df["Close"].iloc[-1]
        print(f" OK ({bars} bars, {date_range}, last=${last_close:.2f})")
        return True

    except Exception as e:
        print(f" FAILED ({e})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download historical OHLCV data from Yahoo Finance")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "SPY", "TSLA"],
                        help="Ticker symbols to download (default: AAPL SPY TSLA)")
    parser.add_argument("--period", default="1y",
                        help="Data period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max (default: 1y)")
    parser.add_argument("--start", default=None,
                        help="Start date YYYY-MM-DD (overrides --period)")
    parser.add_argument("--end", default=None,
                        help="End date YYYY-MM-DD (overrides --period)")
    parser.add_argument("--output", default="data",
                        help="Output directory (default: data)")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("=== Yahoo Finance Data Downloader ===")
    if args.start and args.end:
        print(f"  Range:   {args.start} to {args.end}")
    else:
        print(f"  Period:  {args.period}")
    print(f"  Symbols: {', '.join(args.symbols)}")
    print(f"  Output:  {args.output}/")
    print()

    success = 0
    failed = 0
    for symbol in args.symbols:
        ok = download_symbol(symbol, args.output, args.period, args.start, args.end)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\nDone: {success} downloaded, {failed} failed.")

    if failed > 0:
        print("\nTip: If downloads fail, check your internet connection")
        print("     or try: pip install --upgrade yfinance")
        sys.exit(1)

    # Verify files match config expectations
    config_symbols = ["AAPL", "SPY", "TSLA"]
    missing = [s for s in config_symbols if not os.path.exists(os.path.join(args.output, f"{s}.csv"))]
    if missing:
        print(f"\nWARNING: config/paper_trading.json expects: {', '.join(missing)}")
        print("         Run with: --symbols " + " ".join(config_symbols))


if __name__ == "__main__":
    main()

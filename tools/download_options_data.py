#!/usr/bin/env python3
"""
Download historical options chain data from philippdubach/options-data.

Source: https://github.com/philippdubach/options-data
Format: Parquet files hosted on Cloudflare R2 (~9.4 GB total, per-symbol)

Fields in parquet:
    contract_id, symbol, expiration, strike, type, last, mark, bid, bid_size,
    ask, ask_size, volume, open_interest, date, implied_volatility, delta,
    gamma, theta, vega, rho, in_the_money

Usage:
    python3 tools/download_options_data.py                          # default 8 symbols
    python3 tools/download_options_data.py --symbols AAPL SPY       # specific symbols
    python3 tools/download_options_data.py --start 2024-01-01       # filter date range
    python3 tools/download_options_data.py --max-dte 90             # max days to expiry

Output:
    data/options/AAPL_options.csv, data/options/SPY_options.csv, ...

CSV columns (sorted by date, expiration, strike, type):
    date,symbol,expiration,strike,type,bid,ask,mark,implied_volatility,
    delta,gamma,theta,vega,volume,open_interest
"""

import argparse
import os
import sys
import urllib.request
import tempfile

try:
    import pandas as pd
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pandas and pyarrow required. Run: pip install pandas pyarrow")
    sys.exit(1)


BASE_URL = "https://static.philippdubach.com/data/options"

DEFAULT_SYMBOLS = ["AAPL", "SPY", "TSLA", "MSFT", "NVDA", "AMZN", "GOOG", "META"]

# Columns to keep in the output CSV
OUTPUT_COLUMNS = [
    "date", "symbol", "expiration", "strike", "type",
    "bid", "ask", "mark", "implied_volatility",
    "delta", "gamma", "theta", "vega",
    "volume", "open_interest",
]


def _download_with_progress(url: str, dest: str):
    """Download a URL to a file with a progress indicator."""
    req = urllib.request.Request(url, headers={"User-Agent": "quantpricer/1.0"})
    resp = urllib.request.urlopen(req)
    total = int(resp.headers.get("Content-Length", 0))

    downloaded = 0
    chunk_size = 1024 * 1024  # 1 MB
    with open(dest, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded * 100 // total
                mb = downloaded / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                print(f"\r  downloading {mb:.0f}/{total_mb:.0f} MB ({pct}%)...", end="", flush=True)
    print(" done.", end="", flush=True)


def download_and_convert(symbol: str, output_dir: str,
                         start_date: str = None, end_date: str = None,
                         max_dte: int = None) -> bool:
    """Download parquet for one symbol, filter, and save as CSV."""
    url = f"{BASE_URL}/{symbol.lower()}/options.parquet"
    csv_path = os.path.join(output_dir, f"{symbol}_options.csv")

    print(f"\n  {symbol}:", flush=True)

    try:
        # Download to temp file
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = tmp.name

        _download_with_progress(url, tmp_path)

        print(f"\n  reading parquet...", end="", flush=True)
        df = pq.read_table(tmp_path).to_pandas()
        os.unlink(tmp_path)

        total_rows = len(df)

        # Normalize date columns
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["expiration"] = pd.to_datetime(df["expiration"]).dt.strftime("%Y-%m-%d")

        # Filter by date range
        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]

        # Filter by max days to expiry
        if max_dte is not None:
            df["_dte"] = (pd.to_datetime(df["expiration"]) - pd.to_datetime(df["date"])).dt.days
            df = df[df["_dte"] <= max_dte]
            df = df.drop(columns=["_dte"])

        # Filter out illiquid options (zero bid)
        df = df[df["bid"] > 0.01]

        # Keep only output columns
        available_cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
        df = df[available_cols]

        # Sort for deterministic replay: date -> expiration -> strike -> type
        df = df.sort_values(["date", "expiration", "strike", "type"]).reset_index(drop=True)

        # Round floats for cleaner CSV
        float_cols = ["bid", "ask", "mark", "implied_volatility",
                      "delta", "gamma", "theta", "vega"]
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].round(6)
        if "strike" in df.columns:
            df["strike"] = df["strike"].round(2)

        # Save
        df.to_csv(csv_path, index=False)

        n_dates = df["date"].nunique()
        n_rows = len(df)
        date_range = f"{df['date'].min()} to {df['date'].max()}" if n_rows > 0 else "empty"
        size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        print(f"\n  -> {csv_path} ({size_mb:.1f} MB)")
        print(f"     {n_rows:,} rows (from {total_rows:,} total), {n_dates} trading days, {date_range}")
        return True

    except Exception as e:
        print(f"\n  FAILED: {e}")
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download historical options data from philippdubach/options-data")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        help=f"Symbols to download (default: {' '.join(DEFAULT_SYMBOLS)})")
    parser.add_argument("--start", default=None,
                        help="Start date YYYY-MM-DD (default: all available)")
    parser.add_argument("--end", default=None,
                        help="End date YYYY-MM-DD (default: all available)")
    parser.add_argument("--max-dte", type=int, default=None,
                        help="Max days to expiry to keep (e.g., 90 for <=3 months)")
    parser.add_argument("--output", default="data/options",
                        help="Output directory (default: data/options)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=== Options Data Downloader (philippdubach/options-data) ===")
    print(f"  Source:  {BASE_URL}/{{symbol}}/options.parquet")
    print(f"  Symbols: {', '.join(args.symbols)}")
    if args.start:
        print(f"  Start:   {args.start}")
    if args.end:
        print(f"  End:     {args.end}")
    if args.max_dte:
        print(f"  Max DTE: {args.max_dte}")
    print(f"  Output:  {args.output}/")

    success = 0
    failed = 0
    for symbol in args.symbols:
        ok = download_and_convert(symbol, args.output, args.start, args.end, args.max_dte)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\nDone: {success} downloaded, {failed} failed.")

    if failed > 0:
        print("\nTip: Check your internet connection or verify symbol is in the dataset.")
        print("     Full symbol list: https://github.com/philippdubach/options-data")
        sys.exit(1)


if __name__ == "__main__":
    main()

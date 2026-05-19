# -*- coding: utf-8 -*-
"""
Fetch HOSE + HNX listed firms via vnstock.

Output: data/universe/universe_{date}.csv with columns:
    symbol, exchange, name, listed_date, industry
"""
import os, sys
from datetime import datetime
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from vnstock import Vnstock, Listing

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "universe")
os.makedirs(OUT_DIR, exist_ok=True)


def fetch_listings():
    """Try Listing API first, fallback to per-exchange query."""
    try:
        listing = Listing()
        df = listing.symbols_by_exchange()
        return df
    except Exception as e:
        print(f"  Listing.symbols_by_exchange failed: {e}")

    # Fallback: try alternative methods
    try:
        v = Vnstock()
        # vnstock 4.x sometimes exposes via stock listing
        df = v.stock(symbol="VCB").listing.symbols_by_exchange()
        return df
    except Exception as e:
        print(f"  alternative fetch failed: {e}")
        return pd.DataFrame()


def main():
    print("Fetching HOSE + HNX universe via vnstock...")
    df = fetch_listings()
    if df.empty:
        print("ERROR: empty universe. Check vnstock version.")
        return

    print(f"  total tickers (all exchanges): {len(df)}")
    print(f"  columns: {list(df.columns)}")

    # Filter HOSE + HNX (exclude UPCoM, OTC)
    if "exchange" in df.columns:
        df = df[df["exchange"].isin(["HOSE", "HNX"])].copy()
    elif "exchange_name" in df.columns:
        df = df[df["exchange_name"].isin(["HOSE", "HNX"])].copy()
    else:
        print(f"  WARNING: no 'exchange' column. Columns: {list(df.columns)}")
        print(df.head())

    print(f"  HOSE+HNX: {len(df)}")
    if len(df) > 0:
        print(df.head(5))
        out_path = os.path.join(OUT_DIR, f"universe_{datetime.now().strftime('%Y%m%d')}.csv")
        df.to_csv(out_path, index=False)
        print(f"\nSaved -> {out_path}")
        # Symlink "latest"
        latest = os.path.join(OUT_DIR, "universe_latest.csv")
        if os.path.lexists(latest):
            os.remove(latest)
        try:
            os.symlink(os.path.basename(out_path), latest)
            print(f"Symlink: universe_latest.csv -> {os.path.basename(out_path)}")
        except Exception:
            df.to_csv(latest, index=False)


if __name__ == "__main__":
    main()

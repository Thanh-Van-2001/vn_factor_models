# -*- coding: utf-8 -*-
"""
Fetch daily OHLCV for HOSE + HNX universe via vnstock.

Stores parquet per ticker:
    data/prices/{TICKER}_1d.parquet  with columns: time, open, high, low, close, volume

Resumable: skips tickers already fetched (delete files to re-fetch).
Throttled: 1.5s between requests to avoid vnstock rate limits.
"""
import os, sys, time, argparse
from datetime import datetime
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from vnstock import Vnstock

UNIVERSE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "universe")
PRICES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "prices")
os.makedirs(PRICES_DIR, exist_ok=True)


def fetch_one(symbol, start, end, source="VCI"):
    v = Vnstock().stock(symbol=symbol, source=source)
    df = v.quote.history(start=start, end=end, interval="1D")
    if df is None or len(df) == 0:
        return None
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--limit", type=int, default=None, help="Limit symbols (for testing)")
    ap.add_argument("--throttle", type=float, default=1.5, help="Seconds between requests")
    ap.add_argument("--exchange", default="HOSE,HNX", help="Comma-separated")
    ap.add_argument("--force", action="store_true", help="Re-fetch existing")
    args = ap.parse_args()

    # Load universe
    uni_path = os.path.join(UNIVERSE_DIR, "universe_latest.csv")
    if not os.path.exists(uni_path):
        print(f"ERROR: universe file missing: {uni_path}")
        print("Run: python src/fetch_universe.py")
        return
    uni = pd.read_csv(uni_path)
    excs = [e.strip() for e in args.exchange.split(",")]
    if "exchange" in uni.columns:
        uni = uni[uni["exchange"].isin(excs)]
    print(f"Universe: {len(uni)} tickers ({excs})")

    sym_col = "symbol" if "symbol" in uni.columns else uni.columns[0]
    symbols = uni[sym_col].dropna().unique().tolist()
    if args.limit:
        symbols = symbols[:args.limit]
        print(f"  Limited to first {args.limit}")

    n_done, n_skip, n_fail = 0, 0, 0
    t0 = time.time()
    for i, sym in enumerate(symbols):
        out = os.path.join(PRICES_DIR, f"{sym}_1d.parquet")
        if os.path.exists(out) and not args.force:
            n_skip += 1
            continue
        try:
            df = fetch_one(sym, args.start, args.end)
            if df is None or len(df) == 0:
                n_fail += 1
                print(f"  [{i+1}/{len(symbols)}] {sym}: empty")
                continue
            df.to_parquet(out)
            n_done += 1
            if (i+1) % 25 == 0 or i == 0:
                elapsed = time.time() - t0
                eta = elapsed / max(n_done, 1) * (len(symbols) - i - 1) / 60
                print(f"  [{i+1}/{len(symbols)}] {sym}: {len(df)} bars  done={n_done} skip={n_skip} fail={n_fail}  ETA={eta:.0f}min")
            time.sleep(args.throttle)
        except Exception as e:
            n_fail += 1
            err_short = str(e)[:80]
            print(f"  [{i+1}/{len(symbols)}] {sym}: FAIL {err_short}")
            time.sleep(args.throttle * 2)  # back off on error

    print(f"\nDONE: fetched={n_done}  skipped={n_skip}  failed={n_fail}  total={len(symbols)}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Fetch fundamental data per ticker for factor construction:
  - EPS (trailing 4Q) for EP factor
  - Book value (equity per share) for BM factor (control)
  - Operating profit / book equity for RMW
  - Total assets growth for CMA
  - Shares outstanding for market cap

Output: data/fundamentals/{TICKER}_fund.parquet
        with quarterly history of: eps_ttm, bvps, op_profit, total_assets, shares_out

Throttled and resumable.
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
FUND_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "fundamentals")
os.makedirs(FUND_DIR, exist_ok=True)


def fetch_one(symbol):
    """Fetch quarterly income statement + balance sheet for symbol."""
    v = Vnstock().stock(symbol=symbol, source="VCI")
    out = {}
    try:
        # Income statement (quarterly): for EPS, OP
        inc = v.finance.income_statement(period="quarter", lang="en")
        if inc is not None and len(inc) > 0:
            out["income"] = inc
    except Exception as e:
        print(f"    income_statement: {str(e)[:80]}")
    try:
        # Balance sheet (quarterly): for book value, total assets, shares outstanding
        bs = v.finance.balance_sheet(period="quarter", lang="en")
        if bs is not None and len(bs) > 0:
            out["balance_sheet"] = bs
    except Exception as e:
        print(f"    balance_sheet: {str(e)[:80]}")
    try:
        # Ratio (already includes derived ratios)
        rt = v.finance.ratio(period="quarter", lang="en")
        if rt is not None and len(rt) > 0:
            out["ratio"] = rt
    except Exception as e:
        print(f"    ratio: {str(e)[:80]}")
    return out


def normalise(symbol, raw):
    """Extract canonical columns from raw vnstock outputs."""
    rows = []
    inc = raw.get("income")
    bs = raw.get("balance_sheet")
    rt = raw.get("ratio")
    # vnstock 4.x has multi-index columns or specific col names; we keep the raw and let downstream pick
    # Save all three as separate parquets:
    saves = {}
    for k, df in raw.items():
        if df is not None and len(df) > 0:
            saves[k] = df
    return saves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--throttle", type=float, default=2.0, help="Seconds between requests")
    ap.add_argument("--exchange", default="HOSE,HNX")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    uni_path = os.path.join(UNIVERSE_DIR, "universe_latest.csv")
    uni = pd.read_csv(uni_path)
    excs = [e.strip() for e in args.exchange.split(",")]
    if "exchange" in uni.columns:
        uni = uni[uni["exchange"].isin(excs)]
    sym_col = "symbol" if "symbol" in uni.columns else uni.columns[0]
    symbols = uni[sym_col].dropna().unique().tolist()
    if args.limit:
        symbols = symbols[:args.limit]
    print(f"Fetching fundamentals for {len(symbols)} tickers")

    n_done, n_skip, n_fail = 0, 0, 0
    t0 = time.time()
    for i, sym in enumerate(symbols):
        marker = os.path.join(FUND_DIR, f"{sym}_fund_done.flag")
        if os.path.exists(marker) and not args.force:
            n_skip += 1
            continue
        try:
            raw = fetch_one(sym)
            saved = []
            for k, df in raw.items():
                out = os.path.join(FUND_DIR, f"{sym}_{k}.parquet")
                df.to_parquet(out)
                saved.append(k)
            if saved:
                with open(marker, "w") as f:
                    f.write(",".join(saved))
                n_done += 1
                if (i+1) % 10 == 0 or i == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / max(n_done, 1) * (len(symbols) - i - 1) / 60
                    print(f"  [{i+1}/{len(symbols)}] {sym}: saved {saved}  done={n_done} fail={n_fail}  ETA={eta:.0f}min")
            else:
                n_fail += 1
            time.sleep(args.throttle)
        except Exception as e:
            n_fail += 1
            print(f"  [{i+1}/{len(symbols)}] {sym}: FAIL {str(e)[:100]}")
            time.sleep(args.throttle * 2)

    print(f"\nDONE: fetched={n_done}  skipped={n_skip}  failed={n_fail}")


if __name__ == "__main__":
    main()

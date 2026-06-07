# -*- coding: utf-8 -*-
"""
Fetch ANNUAL income statement + balance sheet per name (VCI).

The vnstock community tier returns only the 4 most-recent periods of financials and the VCI
`ratio` endpoint is stuck on a 2018 snapshot, so we build the fundamental panel from the annual
income/balance statements, which reliably return the last 4 fiscal years (2022-2025). Saved as
pickle (mixed-type cells). Resumable + throttled + rate-limit cooldown.
"""
import os, sys, time, argparse
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from vnstock import Vnstock

ROOT = os.path.join(os.path.dirname(__file__), "..")
FUND = os.path.join(ROOT, "data", "fundamentals")
UNIV = os.path.join(ROOT, "data", "universe")
os.makedirs(FUND, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--throttle", type=float, default=3.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    uni = pd.read_csv(os.path.join(UNIV, "universe_latest.csv"))
    syms = uni["symbol"].tolist()
    n_inc, n_bs, t0 = 0, 0, time.time()

    def cooldown(e):
        if "rate limit" in str(e).lower() or "giới hạn" in str(e).lower() or isinstance(e, SystemExit):
            print("  rate limit -> 60s"); time.sleep(60); return True
        return False

    for i, sym in enumerate(syms):
        inc_out = os.path.join(FUND, f"{sym}_inc_year.pkl")
        bs_out = os.path.join(FUND, f"{sym}_bs_year.pkl")
        if not args.force and os.path.exists(inc_out) and os.path.exists(bs_out):
            continue
        try:
            v = Vnstock().stock(symbol=sym, source="VCI")
        except BaseException as e:
            cooldown(e); continue
        if args.force or not os.path.exists(inc_out):
            try:
                inc = v.finance.income_statement(period="year", lang="en")
                if inc is not None and len(inc):
                    inc.to_pickle(inc_out); n_inc += 1
            except BaseException as e:
                cooldown(e); print(f"  {sym} inc FAIL {str(e)[:50]}")
            time.sleep(args.throttle)
        if args.force or not os.path.exists(bs_out):
            try:
                bs = v.finance.balance_sheet(period="year", lang="en")
                if bs is not None and len(bs):
                    bs.to_pickle(bs_out); n_bs += 1
            except BaseException as e:
                cooldown(e); print(f"  {sym} bs FAIL {str(e)[:50]}")
            time.sleep(args.throttle)
        if (i + 1) % 10 == 0 or i == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (len(syms) - i - 1) / 60
            print(f"  [{i+1}/{len(syms)}] {sym} inc={n_inc} bs={n_bs} ETA={eta:.0f}min")
    print(f"\nDONE: inc={n_inc} bs={n_bs}")


if __name__ == "__main__":
    main()

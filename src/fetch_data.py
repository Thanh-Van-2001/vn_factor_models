# -*- coding: utf-8 -*-
"""
Unified data fetch for the VN factor-library experiments.

Universe = VN100 (HOSE liquid) + HNX30 (HNX liquid) -- the investable, float/liquidity
screened set the report (Section 3.1) says factors should be built on. For each name we
pull:
  - daily OHLCV (adjusted close from VCI)            -> data/prices/{SYM}_1d.parquet
  - quarterly financial ratios (PIT fundamentals)    -> data/ratios/{SYM}_ratio.parquet
and a universe/meta table with exchange + ICB-ish industry for sector neutralisation.

Resumable (skips existing files) and throttled.
"""
import os, sys, time, argparse
from datetime import datetime
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from vnstock import Listing, Vnstock

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
PRICES = os.path.join(DATA, "prices")
RATIOS = os.path.join(DATA, "ratios")
UNIV = os.path.join(DATA, "universe")
for d in (PRICES, RATIOS, UNIV):
    os.makedirs(d, exist_ok=True)


def build_universe():
    l = Listing()
    groups = {}
    for g in ("VN100", "HNX30"):
        try:
            groups[g] = list(l.symbols_by_group(g))
        except Exception as e:
            print(f"  group {g} failed: {str(e)[:80]}")
    syms = sorted(set(s for v in groups.values() for s in v))

    ex = l.symbols_by_exchange()[["symbol", "exchange", "en_organ_name"]]
    ind = l.symbols_by_industries()[["symbol", "industry_name"]]
    uni = pd.DataFrame({"symbol": syms})
    uni = uni.merge(ex, on="symbol", how="left").merge(ind, on="symbol", how="left")
    uni["in_vn100"] = uni["symbol"].isin(groups.get("VN100", []))
    uni["in_hnx30"] = uni["symbol"].isin(groups.get("HNX30", []))
    out = os.path.join(UNIV, "universe_latest.csv")
    uni.to_csv(out, index=False)
    print(f"Universe: {len(uni)} names ({uni['exchange'].value_counts().to_dict()}) -> {out}")
    return uni


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--throttle", type=float, default=2.6)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-universe", action="store_true")
    args = ap.parse_args()

    if args.skip_universe and os.path.exists(os.path.join(UNIV, "universe_latest.csv")):
        uni = pd.read_csv(os.path.join(UNIV, "universe_latest.csv"))
    else:
        uni = build_universe()
    syms = uni["symbol"].tolist()
    n_px, n_rt, n_fail = 0, 0, 0
    t0 = time.time()

    def cooldown(e):
        msg = str(e).lower()
        if "rate limit" in msg or "giới hạn" in msg or isinstance(e, SystemExit):
            print("  rate limit -> sleeping 60s"); time.sleep(60); return True
        return False

    for i, sym in enumerate(syms):
        px_out = os.path.join(PRICES, f"{sym}_1d.parquet")
        rt_out = os.path.join(RATIOS, f"{sym}_ratio.pkl")
        need_px = args.force or not os.path.exists(px_out)
        need_rt = args.force or not os.path.exists(rt_out)
        if not need_px and not need_rt:
            continue
        # one stock handle per symbol (avoids duplicate metadata requests)
        try:
            v = Vnstock().stock(symbol=sym, source="VCI")
        except BaseException as e:
            if cooldown(e):
                try:
                    v = Vnstock().stock(symbol=sym, source="VCI")
                except BaseException:
                    n_fail += 1; continue
            else:
                n_fail += 1; continue
        if need_px:
            try:
                df = v.quote.history(start=args.start, end=args.end, interval="1D")
                if df is not None and len(df):
                    df["time"] = pd.to_datetime(df["time"])
                    df.sort_values("time").reset_index(drop=True).to_parquet(px_out)
                    n_px += 1
            except BaseException as e:
                cooldown(e)
                print(f"  [{i+1}/{len(syms)}] {sym} px FAIL {str(e)[:60]}")
            time.sleep(args.throttle)
        if need_rt:
            try:
                rt = v.finance.ratio(period="quarter", lang="en")
                if rt is not None and len(rt):
                    rt.to_pickle(rt_out)   # pickle: ratio frame has mixed-type cells
                    n_rt += 1
            except BaseException as e:
                cooldown(e)
                print(f"  [{i+1}/{len(syms)}] {sym} rt FAIL {str(e)[:60]}")
            time.sleep(args.throttle)
        if (i + 1) % 10 == 0 or i == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (len(syms) - i - 1) / 60
            print(f"  [{i+1}/{len(syms)}] {sym}  px={n_px} rt={n_rt} fail={n_fail}  ETA={eta:.0f}min")

    print(f"\nDONE: prices={n_px} ratios={n_rt} fail={n_fail} total={len(syms)}")


if __name__ == "__main__":
    main()

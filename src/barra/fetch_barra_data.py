# -*- coding: utf-8 -*-
"""Fetch real data for a Barra-style VN risk model (VN100), rate-limit aware.

vnstock community tier = 60 req/min and HARD-terminates on breach, so we (a) throttle
~1.4s before every API call, (b) make only 2 calls/ticker (price + quarterly ratio;
shares come from the ratio table, sector/is_bank from the industry map), (c) are fully
resumable (skip tickers already saved). Wrap in a restart loop on bee for safety.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import pandas as pd
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from vnstock import Vnstock, Listing

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "..", "..", "data", "barra")
PX = os.path.join(OUT, "prices"); RT = os.path.join(OUT, "ratios")
for d in (OUT, PX, RT):
    os.makedirs(d, exist_ok=True)
START, END = "2021-01-01", pd.Timestamp.today().strftime("%Y-%m-%d")
GROUP = os.environ.get("BARRA_GROUP", "VN100")
THROTTLE = 1.4


def call(fn, *a, **k):
    """Throttle + survive rate-limit termination (sleep 65s and retry once)."""
    for attempt in range(2):
        time.sleep(THROTTLE)
        try:
            return fn(*a, **k)
        except BaseException as e:
            if "rate" in str(e).lower() or "limit" in str(e).lower() or isinstance(e, SystemExit):
                print("  rate-limited, sleeping 65s...", flush=True); time.sleep(65)
            else:
                raise
    return fn(*a, **k)


def main():
    syms = list(Listing().symbols_by_group(GROUP))
    ind = Listing().symbols_by_industries()[["symbol", "industry_name"]]
    ind_map = ind.set_index("symbol")["industry_name"].to_dict()
    print(f"universe {GROUP}: {len(syms)} symbols", flush=True)

    vix = os.path.join(OUT, "vnindex.parquet")
    if not os.path.exists(vix):
        h = call(Vnstock().stock(symbol="VNINDEX", source="VCI").quote.history,
                 start=START, end=END, interval="1D")
        h[["time", "close"]].assign(time=pd.to_datetime(h["time"])).to_parquet(vix)
        print("  saved VNINDEX", flush=True)

    meta = []
    for i, s in enumerate(syms):
        ppath = os.path.join(PX, f"{s}.parquet"); rpath = os.path.join(RT, f"{s}.parquet")
        shares = None
        try:
            st = Vnstock().stock(symbol=s, source="VCI")
            if not os.path.exists(ppath):
                h = call(st.quote.history, start=START, end=END, interval="1D")
                h[["time", "close", "volume"]].assign(time=pd.to_datetime(h["time"])).to_parquet(ppath)
            if not os.path.exists(rpath):
                r = call(st.finance.ratio, period="quarter", lang="en").set_index("item_en")
                qcols = [c for c in r.columns if "-Q" in str(c)]
                def row(n):
                    return pd.to_numeric(r.loc[n, qcols], errors="coerce") if n in r.index else pd.Series(index=qcols, dtype=float)
                rat = pd.DataFrame({"pe": row("P/E"), "pb": row("P/B"), "de": row("Debt/Equity"),
                                    "roe": row("ROE (%)"), "shares": row("Outstanding Shares (mil)")})
                rat.index.name = "period"; rat = rat.reset_index(); rat["symbol"] = s
                rat.to_parquet(rpath)
            # shares (latest) from ratio file
            rr = pd.read_parquet(rpath)
            sh = rr["shares"].dropna()
            shares = float(sh.iloc[-1]) * 1e6 if len(sh) else None
        except BaseException as e:
            print(f"  skip {s}: {str(e)[:60]}", flush=True)
        industry = ind_map.get(s, "Unknown")
        meta.append(dict(symbol=s, shares=shares, industry=industry,
                         is_bank=("Ngân hàng" in str(industry) or "Bank" in str(industry))))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(syms)}  (prices={len(os.listdir(PX))})", flush=True)
    pd.DataFrame(meta).to_csv(os.path.join(OUT, "meta.csv"), index=False)
    print(f"done. prices={len(os.listdir(PX))} ratios={len(os.listdir(RT))}", flush=True)


if __name__ == "__main__":
    main()

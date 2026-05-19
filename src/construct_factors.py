# -*- coding: utf-8 -*-
"""
Construct VN-3 / VN-4 factor returns from per-ticker prices and fundamentals.

Output: data/factors/factors_weekly.parquet  with columns:
    date, MKT, SMB, EMP, TMH, HML, RMW, CMA   (all factor returns)

Procedure (Huang, Liu, Shu 2023 + Fama-French 1993/2015):
  1. Each rebalance date (weekly Wed close):
     a. Build PIT universe (active tickers w/ >= 12mo price history)
     b. Compute characteristics: market cap, EP (trailing 4Q EPS / price),
        BM (book equity / market cap), 12-month avg daily turnover,
        operating profit / book equity, asset growth
     c. 2x3 sort (size median x EP top30/mid40/bot30): 6 portfolios SE,SN,SP,BE,BN,BP
     d. Long-short factors:
        SMB = (SE+SN+SP)/3 - (BE+BN+BP)/3
        EMP = (SE+BE)/2 - (SP+BP)/2
        HML similarly with BM
        TMH = sort on 12mo turnover, low minus high
  2. Concatenate weekly returns, save.
"""
import os, sys
import numpy as np
import pandas as pd
from glob import glob

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PRICES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "prices")
FUND_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "fundamentals")
FACTORS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "factors")
UNIVERSE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "universe")
os.makedirs(FACTORS_DIR, exist_ok=True)


def load_all_prices():
    """Load all available price parquets, returns dict {sym: DataFrame}."""
    files = glob(os.path.join(PRICES_DIR, "*_1d.parquet"))
    data = {}
    for f in files:
        sym = os.path.basename(f).replace("_1d.parquet", "")
        try:
            df = pd.read_parquet(f)
            if df.index.tz: df.index = df.index.tz_localize(None)
            data[sym] = df
        except Exception:
            pass
    print(f"  loaded {len(data)} ticker price series")
    return data


def build_panel(prices):
    """Build long-format panel: index=(date, symbol), columns=open/close/volume/return."""
    rows = []
    for sym, df in prices.items():
        d = df[["close", "volume"]].copy()
        d["symbol"] = sym
        d["return"] = d["close"].pct_change()
        rows.append(d.reset_index())
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.rename(columns={"time": "date"})
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


def market_cap_proxy(prices, fundamentals_dir=FUND_DIR):
    """Approximate market cap as close * shares_outstanding.
    For shares: read from balance_sheet parquet if available; else use volume-based proxy."""
    # Placeholder simplified: use close * total_volume as turnover proxy, market cap will need real shares
    # TODO: parse balance_sheet shares_outstanding properly (vnstock 4.x has nested column names)
    pass


def turnover_12m(prices, panel):
    """Compute 12-month rolling avg daily turnover ratio per (date, symbol).
    Turnover = volume / shares_outstanding. Without shares, use volume / (volume avg) as a proxy.
    """
    panel = panel.sort_values(["symbol", "date"])
    panel["vol_252d_avg"] = panel.groupby("symbol")["volume"].transform(lambda x: x.rolling(252).mean())
    # Self-relative turnover ratio: today_vol / 252d_avg_vol  (proxy for relative speculation)
    panel["turnover_proxy"] = panel["volume"] / (panel["vol_252d_avg"] + 1e-9)
    panel["turnover_12m"] = panel.groupby("symbol")["turnover_proxy"].transform(lambda x: x.rolling(252).mean())
    return panel


def ep_ratio(panel, fund_dir=FUND_DIR):
    """Earnings/Price ratio per ticker per date.
    Uses trailing 4Q EPS from income statement, divided by current close.
    For now, returns NaN as fundamentals require parsing — to be implemented when fundamentals fetched.
    """
    # TODO: parse income_statement parquets for net income, divide by shares outstanding to get EPS,
    # sum trailing 4Q, divide by price to get EP.
    panel["ep"] = np.nan
    return panel


def make_factors_weekly(panel):
    """Resample to weekly (Wed close) and form factor portfolios.
    Note: this is the VN-4 SCAFFOLD. Full implementation requires:
      - Real market cap (close * shares_out)
      - Real EP (trailing 4Q EPS / price)
      - Real turnover ratio (volume / shares_out)
      - Risk-free rate series
    Current stub uses TURNOVER PROXY only to demo the pipeline.
    """
    # For now: compute MKT (equal-weighted index return) + TMH (low-minus-high turnover)
    factors = []
    panel["week"] = panel["date"] - pd.to_timedelta(panel["date"].dt.dayofweek, unit="D") + pd.Timedelta(days=2)  # Wed-anchored
    weekly = panel.groupby(["week", "symbol"]).agg(
        close=("close", "last"),
        ret_w=("return", lambda x: (1 + x.fillna(0)).prod() - 1),
        turn=("turnover_12m", "last"),
    ).reset_index()

    grouped = []
    for week, df in weekly.groupby("week"):
        df = df.dropna(subset=["ret_w"])
        if len(df) < 30:
            continue
        # MKT proxy: equal-weighted (real should be value-weighted with real market cap)
        mkt = df["ret_w"].mean()
        # Turnover sort: top/bottom 30%
        df = df.dropna(subset=["turn"])
        if len(df) < 20:
            grouped.append({"week": week, "MKT": mkt, "TMH": np.nan, "n_tickers": len(df)})
            continue
        q_lo, q_hi = df["turn"].quantile(0.30), df["turn"].quantile(0.70)
        low_turn = df[df["turn"] <= q_lo]["ret_w"].mean()
        high_turn = df[df["turn"] >= q_hi]["ret_w"].mean()
        tmh = low_turn - high_turn
        grouped.append({"week": week, "MKT": mkt, "TMH": tmh, "n_tickers": len(df)})

    fac = pd.DataFrame(grouped).sort_values("week").reset_index(drop=True)
    return fac


def main():
    print("Loading prices...")
    prices = load_all_prices()
    if len(prices) == 0:
        print("ERROR: no price data. Run src/fetch_prices.py first.")
        return

    print("Building panel...")
    panel = build_panel(prices)
    print(f"  panel rows: {len(panel):,}  range {panel['date'].min()} -> {panel['date'].max()}")

    print("Computing turnover proxy...")
    panel = turnover_12m(prices, panel)

    print("Computing EP (TODO: real EPS)...")
    panel = ep_ratio(panel)

    print("Building weekly factors (TURNOVER-only stub, no real EP/SMB/HML yet)...")
    fac = make_factors_weekly(panel)
    print(f"  factor rows: {len(fac)}")
    print(fac.head(10))
    print(f"\n  MKT premium (annualized): {fac['MKT'].mean() * 52 * 100:.2f}%/yr")
    if "TMH" in fac.columns:
        valid = fac["TMH"].dropna()
        if len(valid):
            print(f"  TMH premium (annualized): {valid.mean() * 52 * 100:.2f}%/yr  | Sharpe: {valid.mean()/valid.std()*np.sqrt(52):.2f}")

    out = os.path.join(FACTORS_DIR, "factors_weekly_v0.parquet")
    fac.to_parquet(out)
    print(f"\nSaved -> {out}")
    print("\nNOTE: this is v0 scaffold. SMB/EMP/HML require fundamentals parsing (TODO).")


if __name__ == "__main__":
    main()

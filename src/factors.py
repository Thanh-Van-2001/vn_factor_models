# -*- coding: utf-8 -*-
"""
Factor library for the Vietnamese equity market.

Implements a faithful, computable subset of the BeeTrade "Cross-Sectional Factor
Library for the Vietnamese Equity Market" (June 2026) across its seven families:

  Value      : EP, BP, SP, EBITDAEV, DY                       (V1,V2,V3,V4,V6)
  Quality    : OP, GP, ROE, ROA, LEV(low), AG(low)            (Q1,Q2,Q3, Q5, I1)
  Momentum   : MOM(12-1), REV(1m), PH52                        (M1,M2,M4)
  Risk       : BETA(low), IVOL(low), MAX(low), SKEW(low)       (R1,R2,R3,R4)
  Liquidity  : SIZE(small), TURN(low), ILLIQ, ZERORET         (L1,L2,L3,L4)

Methodology follows the report:
  - Investable universe = VN100 + HNX30 (liquid / float screened, Section 3.1).
  - Point-in-time fundamentals with a conservative reporting lag (Section 3, 7.1):
    quarterly ratios become usable only `LAG_DAYS` after quarter end.
  - Value/yield factors are recomputed against the LIVE price (per-share fundamentals
    are backed out from the quarter-end snapshot), so they respond to price, not stale.
  - Robust standardisation: winsorise at 1/99 pct, then z = (x - median)/MAD  (Eq. 1).
  - Sector + size neutralisation: residual of z on industry dummies + log(mcap)  (Eq. 2).
  - Sign convention applied so a higher score => expected higher return (Section 3.3).

Output:
  data/factors/characteristics.parquet  -- raw monthly PIT characteristics + fwd returns
  data/factors/scores.parquet           -- neutralised, signed factor scores (the z-tilde)
"""
import os, glob, re, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data")
PRICES = os.path.join(DATA, "prices")
RATIOS = os.path.join(DATA, "ratios")
UNIV = os.path.join(DATA, "universe")
FACTORS = os.path.join(DATA, "factors")
os.makedirs(FACTORS, exist_ok=True)

LAG_DAYS = 60                      # conservative PIT reporting lag (report 7.1)
TD_Y = 252                         # trading days / year

# expected sign: +1 => long high values, -1 => long low values (Section 3.3)
SIGN = {
    "EP": +1, "BP": +1, "SP": +1,
    "OP": +1, "GP": +1, "ROE": +1, "ROA": +1, "LEV": -1, "AG": -1,
    "MOM": +1, "REV": +1, "PH52": +1,                       # REV stored as -ret already
    "BETA": -1, "IVOL": -1, "MAX": -1, "SKEW": -1,
    "SIZE": -1, "TURN": -1, "ILLIQ": +1, "ZERORET": -1,     # ILLIQ tested raw (report: muted/inverted)
}
FAMILY = {
    "Value": ["EP", "BP", "SP"],
    "Quality": ["OP", "GP", "ROE", "ROA", "LEV", "AG"],
    "Momentum": ["MOM", "REV", "PH52"],
    "Risk": ["BETA", "IVOL", "MAX", "SKEW"],
    "Liquidity": ["SIZE", "TURN", "ILLIQ", "ZERORET"],
}
ALL_FACTORS = [f for fam in FAMILY.values() for f in fam]
FUNDAMENTAL = {"EP", "BP", "SP", "OP", "GP", "ROE", "ROA", "LEV", "AG"}
FINANCIAL_INDUSTRIES = {"Ngân hàng", "Dịch vụ tài chính", "Bảo hiểm", "Chứng khoán"}
FUND_LAG_DAYS = 90        # annual report published ~Q1 of next year


# ---------------------------------------------------------------- data loading
def load_prices():
    closes, vols = {}, {}
    for f in glob.glob(os.path.join(PRICES, "*_1d.parquet")):
        sym = os.path.basename(f).replace("_1d.parquet", "")
        d = pd.read_parquet(f)
        d["time"] = pd.to_datetime(d["time"])
        d = d.set_index("time").sort_index()
        d = d[~d.index.duplicated(keep="last")]
        closes[sym] = d["close"]
        vols[sym] = d["volume"]
    close = pd.DataFrame(closes).sort_index()
    vol = pd.DataFrame(vols).sort_index()
    return close, vol


FUND_DIR = os.path.join(DATA, "fundamentals")
_YEAR_RE = re.compile(r"^\d{4}$")
_INC = {"Net sales": "revenue", "Sales": "sales", "Gross Profit": "gross_profit",
        "Operating profit/(loss)": "op_profit", "Net profit/(loss) after tax": "net_income",
        "Attributable to parent company": "net_parent", "EPS basic (VND)": "eps"}
_BS = {"Total Assets": "total_assets", "Owner's Equity": "equity",
       "Short-term borrowings": "st_debt", "Long-term borrowings": "lt_debt"}


def _statement_panel(path, mapping):
    df = pd.read_pickle(path)
    if "item_en" not in df.columns:
        return None
    ycols = [c for c in df.columns if _YEAR_RE.match(str(c))]
    if not ycols:
        return None
    sub = df[df["item_en"].isin(mapping)].copy()
    sub["field"] = sub["item_en"].map(mapping)
    sub = sub.drop_duplicates("field", keep="first").set_index("field")[ycols]
    tidy = sub.T.apply(pd.to_numeric, errors="coerce")
    tidy["year"] = [int(y) for y in tidy.index]
    return tidy.reset_index(drop=True)


def load_fundamentals():
    """Annual fundamental panel from income + balance statements (last ~4 fiscal years).
    avail = fiscal-year-end (Dec 31, Y) + FUND_LAG_DAYS. Carried forward only (no look-ahead)."""
    rows = []
    for f in glob.glob(os.path.join(FUND_DIR, "*_inc_year.pkl")):
        sym = os.path.basename(f).replace("_inc_year.pkl", "")
        bsf = os.path.join(FUND_DIR, f"{sym}_bs_year.pkl")
        inc = _statement_panel(f, _INC)
        bs = _statement_panel(bsf, _BS) if os.path.exists(bsf) else None
        if inc is None or bs is None:
            continue
        m = inc.merge(bs, on="year", how="outer")
        m["symbol"] = sym
        rows.append(m)
    fund = pd.concat(rows, ignore_index=True).sort_values(["symbol", "year"])
    fund["revenue"] = fund["revenue"].fillna(fund.get("sales"))
    fund["total_debt"] = fund[["st_debt", "lt_debt"]].sum(axis=1, min_count=1)
    # shares from EPS identity (handles dilution per year); fallback NaN
    np_ = fund["net_parent"].where(fund["net_parent"].notna(), fund["net_income"])
    fund["shares"] = (np_ / fund["eps"]).where(fund["eps"].abs() > 1e-9)
    fund["shares"] = fund["shares"].where(fund["shares"] > 0)
    fund["ag"] = fund.groupby("symbol")["total_assets"].pct_change(1)
    fund["fy_end"] = pd.to_datetime(fund["year"].astype(int).astype(str) + "-12-31")
    fund["avail"] = fund["fy_end"] + pd.Timedelta(days=FUND_LAG_DAYS)
    return fund.dropna(subset=["year"]).reset_index(drop=True)


def asof_fund(fund_sym, t):
    s = fund_sym[fund_sym["avail"] <= t]
    return s.iloc[-1] if len(s) else None


# ---------------------------------------------------------------- characteristics
def month_end_dates(index, min_history=260):
    s = pd.Series(index, index=index)
    me = s.resample("ME").last().dropna()
    me = pd.DatetimeIndex(me.values)
    return me[me >= index[min_history]]


def build_characteristics():
    close, vol = load_prices()
    fund = load_fundamentals()
    uni = pd.read_csv(os.path.join(UNIV, "universe_latest.csv")).set_index("symbol")
    industry = uni["industry_name"].to_dict()

    syms = [s for s in close.columns if s in set(fund["symbol"].unique())]
    close = close[syms]
    ret = close.pct_change()
    dollar_vol = close * vol                              # turnover value in (thousand VND)*shares
    mkt = ret.median(axis=1)                              # robust equal market proxy

    fund_by_sym = {s: g for s, g in fund.groupby("symbol")}
    # per-symbol shares fallback (median across years) for the SIZE proxy before fundamentals exist
    shares_fallback = fund.dropna(subset=["shares"]).groupby("symbol")["shares"].median().to_dict()
    dates = month_end_dates(close.index)
    recs = []
    for ti in range(len(dates) - 1):
        t = dates[ti]
        if t not in close.index:
            t = close.index[close.index.get_indexer([t], method="ffill")[0]]
        hist = close.index[close.index <= t]
        if len(hist) < 260:
            continue
        d252 = hist[-252:]
        d120 = hist[-120:]
        d21 = hist[-21:]
        rwin = ret.loc[d252]
        mwin = mkt.loc[d252]
        px_t = close.loc[t]

        # forward 1-month total return (to next rebalance)
        t_next = dates[ti + 1]
        if t_next not in close.index:
            idx = close.index[close.index.get_indexer([t_next], method="ffill")[0]]
            t_next = idx
        fwd = close.loc[t_next] / px_t - 1.0
        # vnstock close is in THOUSAND VND; ratio-derived per-share/levels are RAW VND.
        # Work in raw VND for all monetary quantities so value ratios & EV are consistent.
        px_raw = px_t * 1000.0

        for s in syms:
            if not np.isfinite(px_raw[s]) or px_raw[s] <= 0:
                continue
            r = asof_fund(fund_by_sym[s], t)         # carried-forward fundamentals (PIT)
            # shares for the size proxy: PIT shares if available, else per-symbol fallback
            shares = r["shares"] if (r is not None and np.isfinite(r.get("shares", np.nan))) \
                else shares_fallback.get(s, np.nan)
            if not np.isfinite(shares) or shares <= 0:
                continue
            mcap_live = px_raw[s] * shares
            if not np.isfinite(mcap_live) or mcap_live <= 0:
                continue
            ind = industry.get(s, "Other")
            is_fin = ind in FINANCIAL_INDUSTRIES

            # --- price-based ---
            cwin = close[s].loc[d252].dropna()
            rser = ret[s].loc[d252].dropna()
            p252, p21 = close[s].loc[hist[-252]], close[s].loc[hist[-21]]
            mom = (p21 / p252 - 1.0) if (np.isfinite(p252) and p252 > 0 and np.isfinite(p21)) else np.nan
            rev = -(px_t[s] / p21 - 1.0) if (np.isfinite(p21) and p21 > 0) else np.nan
            ph52 = px_t[s] / cwin.max() if len(cwin) else np.nan
            # beta & ivol via market regression over 252d
            rr = pd.concat([rser, mwin], axis=1).dropna()
            rr.columns = ["ri", "rm"]
            if len(rr) > 60:
                beta = rr.cov().iloc[0, 1] / rr["rm"].var()
                resid = rr["ri"] - (beta * rr["rm"])
                ivol = resid.iloc[-120:].std() * np.sqrt(TD_Y)
            else:
                beta, ivol = np.nan, np.nan
            r21 = ret[s].loc[d21].dropna()
            mx = r21.nlargest(5).mean() if len(r21) >= 10 else np.nan
            skew = ret[s].loc[d120].dropna().skew() if ret[s].loc[d120].dropna().size > 30 else np.nan
            turn = (vol[s].loc[d252] / shares).mean()
            amt = dollar_vol[s].loc[d252].replace(0, np.nan)
            illiq = (ret[s].loc[d252].abs() / amt).replace([np.inf, -np.inf], np.nan).mean() * 1e6
            zeroret = (ret[s].loc[hist[-120:]].abs() < 1e-8).mean()

            # --- fundamental (only when an annual report is actually available at t) ---
            ep = bp = sp = op = gp = roe = roa = lev = ag = np.nan
            if r is not None and np.isfinite(r.get("shares", np.nan)):
                eps = r.get("eps", np.nan)               # VND/share (raw)
                eq = r.get("equity", np.nan)             # VND
                revn = r.get("revenue", np.nan)
                ta = r.get("total_assets", np.nan)
                ni = r.get("net_income", np.nan)
                opp = r.get("op_profit", np.nan)
                gpf = r.get("gross_profit", np.nan)
                debt = r.get("total_debt", np.nan)
                ep = eps / px_raw[s] if np.isfinite(eps) else np.nan
                bp = eq / mcap_live if np.isfinite(eq) else np.nan
                sp = revn / mcap_live if np.isfinite(revn) else np.nan
                op = opp / eq if (np.isfinite(opp) and np.isfinite(eq) and eq != 0) else np.nan
                gp = gpf / ta if (np.isfinite(gpf) and np.isfinite(ta) and ta != 0 and not is_fin) else np.nan
                roe = ni / eq if (np.isfinite(ni) and np.isfinite(eq) and eq != 0) else np.nan
                roa = ni / ta if (np.isfinite(ni) and np.isfinite(ta) and ta != 0) else np.nan
                lev = debt / eq if (np.isfinite(debt) and np.isfinite(eq) and eq != 0 and not is_fin) else np.nan
                ag = r.get("ag", np.nan) if not is_fin else np.nan

            recs.append(dict(
                date=t, symbol=s, industry=ind, mcap=mcap_live, fwd_ret=fwd[s],
                EP=ep, BP=bp, SP=sp,
                OP=op, GP=gp, ROE=roe, ROA=roa, LEV=lev, AG=ag,
                MOM=mom, REV=rev, PH52=ph52,
                BETA=beta, IVOL=ivol, MAX=mx, SKEW=skew,
                SIZE=np.log(mcap_live), TURN=turn, ILLIQ=illiq, ZERORET=zeroret,
            ))
    panel = pd.DataFrame(recs)
    out = os.path.join(FACTORS, "characteristics.parquet")
    panel.to_parquet(out)
    print(f"characteristics: {panel['date'].nunique()} months x ~{len(panel)//max(panel['date'].nunique(),1)} "
          f"stocks = {len(panel):,} rows -> {out}")
    return panel


# ---------------------------------------------------------------- standardise / neutralise
def robust_z(x):
    x = x.astype(float).copy()
    lo, hi = x.quantile(0.01), x.quantile(0.99)
    x = x.clip(lo, hi)
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        sd = x.std()
        return (x - x.mean()) / sd if sd and np.isfinite(sd) else x * 0.0
    return (x - med) / (1.4826 * mad)


def neutralise(z, industry, logmcap):
    """Residual of z on industry dummies + log mcap (Eq. 2)."""
    df = pd.DataFrame({"z": z, "ind": industry.values, "lm": logmcap.values}).dropna()
    if len(df) < 10:
        return z
    D = pd.get_dummies(df["ind"], drop_first=True).astype(float)
    X = np.column_stack([np.ones(len(df)), df["lm"].values, D.values]) if len(D.columns) else \
        np.column_stack([np.ones(len(df)), df["lm"].values])
    y = df["z"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = pd.Series(y - X @ beta, index=df.index)
    return resid.reindex(z.index)


def build_scores(panel=None):
    if panel is None:
        panel = pd.read_parquet(os.path.join(FACTORS, "characteristics.parquet"))
    out = []
    for t, g in panel.groupby("date"):
        g = g.copy()
        if len(g) < 20:
            continue
        logmcap = np.log(g["mcap"])
        row = g[["date", "symbol", "industry", "mcap", "fwd_ret"]].copy()
        for fac in ALL_FACTORS:
            z = robust_z(g[fac])
            zt = neutralise(z, g["industry"], logmcap)
            row[fac] = SIGN[fac] * zt
        out.append(row)
    scores = pd.concat(out, ignore_index=True)
    sp = os.path.join(FACTORS, "scores.parquet")
    scores.to_parquet(sp)
    print(f"scores (signed, neutralised): {len(scores):,} rows, {scores['date'].nunique()} months -> {sp}")
    return scores


if __name__ == "__main__":
    p = build_characteristics()
    build_scores(p)

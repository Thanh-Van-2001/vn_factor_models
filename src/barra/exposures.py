# -*- coding: utf-8 -*-
"""Build a weekly, point-in-time Barra-style EXPOSURE panel from the fetched data.

Style factors (all standardised cross-sectionally each week -> cap-weighted mean 0, std 1,
winsorised at +/-3 MAD): SIZE, BETA, MOMENTUM, RESVOL, LIQUIDITY, VALUE, LEVERAGE, QUALITY.
Plus coarse INDUSTRY one-hot. Dependent variable = next-week total return. No look-ahead:
price descriptors use data up to t; fundamentals use the latest quarter reported >=45d before t.
"""
import os, glob, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
BARRA = os.path.join(ROOT, "..", "..", "data", "barra")
PX = os.path.join(BARRA, "prices"); RT = os.path.join(BARRA, "ratios")
STYLES = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "LIQUIDITY", "VALUE", "LEVERAGE", "QUALITY"]


def load_panels():
    closes, vols = {}, {}
    for f in glob.glob(os.path.join(PX, "*.parquet")):
        sym = os.path.basename(f)[:-8]
        d = pd.read_parquet(f).set_index("time")
        closes[sym] = d["close"]; vols[sym] = d["volume"]
    close = pd.DataFrame(closes).sort_index()
    vol = pd.DataFrame(vols).sort_index()
    close.index = pd.to_datetime(close.index); vol.index = pd.to_datetime(vol.index)
    return close, vol


def load_ratios():
    rows = []
    for f in glob.glob(os.path.join(RT, "*.parquet")):
        rows.append(pd.read_parquet(f))
    if not rows:
        return pd.DataFrame()
    r = pd.concat(rows, ignore_index=True)
    def qend(p):
        try:
            y, q = str(p).split("-Q"); m = {"1": 3, "2": 6, "3": 9, "4": 12}[q]
            return pd.Timestamp(int(y), m, 1) + pd.offsets.MonthEnd(0)
        except Exception:
            return pd.NaT
    r["qend"] = r["period"].map(qend)
    r["avail"] = r["qend"] + pd.Timedelta(days=45)   # reporting lag
    return r.dropna(subset=["avail"]).sort_values("avail")


def asof_fund(rat, sym, t):
    """Latest reported ratio for `sym` available by date t."""
    s = rat[(rat["symbol"] == sym) & (rat["avail"] <= t)]
    if s.empty:
        return dict(pe=np.nan, pb=np.nan, de=np.nan, roe=np.nan)
    last = s.iloc[-1]
    return dict(pe=last["pe"], pb=last["pb"], de=last["de"], roe=last["roe"])


def zscore(s, capw):
    """Winsorise (3 MAD) then standardise: cap-weighted mean 0, equal-weighted std 1."""
    x = s.copy().astype(float)
    med = x.median(); mad = (x - med).abs().median()
    if mad > 0:
        x = x.clip(med - 3 * 1.4826 * mad, med + 3 * 1.4826 * mad)
    w = capw.reindex(x.index).fillna(0)
    mu = np.nansum(x * w) / w[~x.isna()].sum() if w[~x.isna()].sum() > 0 else x.mean()
    sd = x.std()
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return ((x - mu) / sd).fillna(0.0)


def coarse_industry(meta):
    cnt = meta["industry"].value_counts()
    keep = set(cnt[cnt >= 4].index)
    return meta["industry"].where(meta["industry"].isin(keep), "Other").fillna("Other")


def main():
    close, vol = load_panels()
    rat = load_ratios()
    meta = pd.read_csv(os.path.join(BARRA, "meta.csv")).set_index("symbol")
    meta["ind"] = coarse_industry(meta)
    shares = meta["shares"].to_dict()
    syms = [s for s in close.columns if s in meta.index and pd.notna(shares.get(s))]
    close = close[syms]; vol = vol[syms]
    ret = close.pct_change()
    mkt = ret.median(axis=1)                 # robust market proxy for beta (cap-w done in model)

    # weekly rebalance = last ACTUAL trading day of each Wed-ending week (avoids holidays)
    wk_all = pd.Series(close.index, index=close.index).resample("W-WED").last().dropna()
    wk_all = pd.DatetimeIndex(pd.to_datetime(wk_all.values))
    start_i = int(wk_all.searchsorted(close.index[260]))
    rows = []
    for pos in range(start_i, len(wk_all) - 1):
        t = wk_all[pos]; t_next = wk_all[pos + 1]
        if t not in close.index or t_next not in close.index:
            continue
        hist = close.index[close.index <= t]
        if len(hist) < 260:
            continue
        win = hist[-252:]
        rd = ret.loc[win]
        mcap = (close.loc[t] * pd.Series(shares)).reindex(syms)
        capw = (mcap / mcap.sum())
        size = np.log(mcap)
        beta = rd.apply(lambda c: np.cov(c.dropna(), mkt.loc[c.dropna().index])[0, 1] /
                        np.var(mkt.loc[c.dropna().index]) if c.dropna().size > 60 else np.nan)
        logret = np.log(close.loc[win[0]:t]).diff()
        mom = logret.iloc[-252:-21].sum()
        resvol = rd.iloc[-120:].std()
        turn = (close.loc[win[-60:]] * vol.loc[win[-60:]]).mean() / mcap
        liq = np.log(turn.replace(0, np.nan))
        fund = pd.DataFrame({s: asof_fund(rat, s, t) for s in syms}).T
        ep = 1.0 / fund["pe"].where(fund["pe"] > 0)
        bm = 1.0 / fund["pb"].where(fund["pb"] > 0)
        value = zscore(ep, capw) + zscore(bm, capw)
        raw = pd.DataFrame({
            "SIZE": size, "BETA": beta, "MOMENTUM": mom, "RESVOL": resvol,
            "LIQUIDITY": liq, "VALUE": value, "LEVERAGE": fund["de"], "QUALITY": fund["roe"],
        })
        Z = pd.DataFrame({c: zscore(raw[c], capw) for c in STYLES})
        Z["mcap"] = mcap; Z["capw"] = capw
        Z["ind"] = meta.loc[syms, "ind"].values
        # next-week return (actual trading days)
        fwd = (close.loc[t_next] / close.loc[t] - 1.0).reindex(syms)
        Z["fwd_ret"] = fwd.values
        Z["date"] = t; Z["symbol"] = syms
        rows.append(Z.reset_index(drop=True))
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.dropna(subset=["fwd_ret"])
    out = os.path.join(BARRA, "exposures.parquet")
    panel.to_parquet(out)
    print(f"exposures: {panel['date'].nunique()} weeks x ~{int(len(panel)/panel['date'].nunique())} stocks "
          f"= {len(panel):,} rows -> {out}")
    print("industries:", sorted(panel['ind'].unique()))


if __name__ == "__main__":
    main()

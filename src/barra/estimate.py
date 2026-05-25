# -*- coding: utf-8 -*-
"""Barra cross-sectional estimation: weekly WLS of next-week returns on factor exposures.

Model per week:  r_i = f_country + Sum_m X_ind(i,m) f_ind,m + Sum_s X_style(i,s) f_style,s + u_i
with the standard identification constraint  Sum_m (capwt_m) f_ind,m = 0  (so f_country is the
cap-weighted market return and industries are pure relative tilts). Weights = sqrt(mcap)
(Barra convention). Solved exactly via the KKT system. Outputs the factor-return time series,
factor covariance, specific risk, and a significance table for the style factors.
"""
import os
import numpy as np
import pandas as pd

BARRA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "barra")
STYLES = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "LIQUIDITY", "VALUE", "LEVERAGE", "QUALITY"]


def solve_week(g):
    y = g["fwd_ret"].to_numpy(float)
    cap = g["mcap"].to_numpy(float)
    w = np.sqrt(cap); w = w / w.mean()
    capw = cap / cap.sum()
    inds = sorted(g["ind"].unique())
    Xi = np.column_stack([(g["ind"] == m).to_numpy(float) for m in inds])
    cwt = np.array([capw[(g["ind"] == m).to_numpy()].sum() for m in inds])
    Xc = np.ones((len(g), 1))
    Xs = g[STYLES].to_numpy(float)
    X = np.hstack([Xc, Xi, Xs])
    P = X.shape[1]
    R = np.zeros((1, P)); R[0, 1:1 + len(inds)] = cwt          # constraint on industries
    W = w
    A = (X * W[:, None]).T @ X
    b = (X * W[:, None]).T @ y
    KKT = np.block([[A, R.T], [R, np.zeros((1, 1))]])
    rhs = np.concatenate([b, [0.0]])
    try:
        sol = np.linalg.solve(KKT, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
    f = sol[:P]
    resid = y - X @ f
    ybar = np.average(y, weights=W)
    sstot = np.sum(W * (y - ybar) ** 2); ssres = np.sum(W * resid ** 2)
    r2 = 1 - ssres / sstot if sstot > 0 else np.nan
    names = ["country"] + [f"IND_{m}" for m in inds] + STYLES
    fr = dict(zip(names, f))
    spec = pd.Series(resid, index=g["symbol"].values)
    return fr, spec, r2


def main():
    panel = pd.read_parquet(os.path.join(BARRA, "exposures.parquet"))
    panel = panel.dropna(subset=["fwd_ret", "mcap"])
    frs, specs, r2s = {}, {}, {}
    for t, g in panel.groupby("date"):
        if len(g) < 30:
            continue
        fr, spec, r2 = solve_week(g)
        frs[t] = fr; specs[t] = spec; r2s[t] = r2
    F = pd.DataFrame(frs).T.sort_index()
    R2 = pd.Series(r2s).sort_index()
    spec_df = pd.DataFrame(specs).T.sort_index()
    F.to_parquet(os.path.join(BARRA, "factor_returns.parquet"))
    spec_df.to_parquet(os.path.join(BARRA, "specific_returns.parquet"))
    R2.to_frame("r2").to_parquet(os.path.join(BARRA, "cs_r2.parquet"))

    # style-factor significance (weekly -> annualised)
    print(f"\n=== Barra cross-sectional model: {len(F)} weekly fits, "
          f"avg cross-sectional R^2 = {R2.mean()*100:.1f}% ===")
    print(f"{'factor':12}{'ann.ret%':>10}{'ann.vol%':>10}{'t-stat':>9}{'IR':>7}")
    for s in STYLES:
        x = F[s].dropna()
        ann = x.mean() * 52 * 100
        vol = x.std() * np.sqrt(52) * 100
        t = x.mean() / x.std() * np.sqrt(len(x)) if x.std() else 0
        ir = (x.mean() * 52) / (x.std() * np.sqrt(52)) if x.std() else 0
        print(f"{s:12}{ann:>+10.2f}{vol:>10.2f}{t:>+9.2f}{ir:>+7.2f}")
    # annualised factor covariance (styles + country)
    cov = F[["country"] + STYLES].dropna().cov() * 52
    cov.to_parquet(os.path.join(BARRA, "factor_cov.parquet"))
    print("\nsaved factor_returns / factor_cov / specific_returns / cs_r2 to data/barra/")


if __name__ == "__main__":
    main()

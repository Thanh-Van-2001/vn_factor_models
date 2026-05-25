# -*- coding: utf-8 -*-
"""Barra model report: cumulative factor returns, factor vols, cross-sectional R^2, and a
portfolio RISK DECOMPOSITION (factor vs specific, and contribution by factor) -- the whole
point of a Barra-style risk model. Saves docs/barra.png."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
BARRA = os.path.join(ROOT, "..", "..", "data", "barra")
DOCS = os.path.join(ROOT, "..", "..", "docs")
STYLES = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "LIQUIDITY", "VALUE", "LEVERAGE", "QUALITY"]


def main():
    F = pd.read_parquet(os.path.join(BARRA, "factor_returns.parquet"))
    R2 = pd.read_parquet(os.path.join(BARRA, "cs_r2.parquet"))["r2"]
    spec = pd.read_parquet(os.path.join(BARRA, "specific_returns.parquet"))
    panel = pd.read_parquet(os.path.join(BARRA, "exposures.parquet"))

    # ---- risk decomposition of an equal-weight portfolio at the latest week ----
    last = panel[panel["date"] == panel["date"].max()].copy()
    inds = sorted(last["ind"].unique())
    cols = ["country"] + [f"IND_{m}" for m in inds] + STYLES
    N = len(last); w = np.full(N, 1.0 / N)
    Xc = np.ones((N, 1))
    Xi = np.column_stack([(last["ind"] == m).to_numpy(float) for m in inds])
    Xs = last[STYLES].to_numpy(float)
    X = pd.DataFrame(np.hstack([Xc, Xi, Xs]), columns=cols, index=last["symbol"].values)
    Fc = F.reindex(columns=cols).fillna(0.0)
    cov = Fc.cov() * 52.0
    b = X.T @ w                                   # portfolio factor exposures
    fac_var = float(b.values @ cov.values @ b.values)
    svar = (spec.var() * 52.0).reindex(last["symbol"].values).fillna(spec.var().mean() * 52)
    spec_var = float(np.sum(w ** 2 * svar.values))
    tot_vol = np.sqrt(fac_var + spec_var)
    mc = (cov.values @ b.values) * b.values       # variance contribution per factor
    grp = {"Market": mc[cols.index("country")],
           "Industry": mc[[cols.index(f"IND_{m}") for m in inds]].sum(),
           **{s: mc[cols.index(s)] for s in STYLES}}
    grp["Specific"] = spec_var

    print(f"=== Equal-weight portfolio risk decomposition (latest week, N={N}) ===")
    print(f"  total annualised vol = {tot_vol*100:.1f}%   "
          f"(factor {np.sqrt(max(fac_var,0))*100:.1f}% / specific {np.sqrt(spec_var)*100:.1f}%)")
    for k, v in sorted(grp.items(), key=lambda kv: -kv[1]):
        print(f"    {k:11} {v/(tot_vol**2)*100:+6.1f}% of variance")

    # ---- plots ----
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    cum = F[STYLES].cumsum()
    for s in STYLES:
        ax[0, 0].plot(cum.index, cum[s], label=s, lw=1.3)
    ax[0, 0].axhline(0, color="k", lw=.5); ax[0, 0].legend(fontsize=7, ncol=2)
    ax[0, 0].set_title("Cumulative style-factor returns (weekly)"); ax[0, 0].grid(alpha=.25)

    vols = (F[STYLES].std() * np.sqrt(52) * 100).sort_values()
    ax[0, 1].barh(vols.index, vols.values, color="#1f77b4")
    ax[0, 1].set_title("Annualised style-factor volatility (%)"); ax[0, 1].grid(alpha=.25)

    ax[1, 0].plot(R2.index, R2.values * 100, color="#2ca02c", lw=1)
    ax[1, 0].axhline(R2.mean() * 100, color="#d62728", ls="--", label=f"mean {R2.mean()*100:.1f}%")
    ax[1, 0].set_title("Weekly cross-sectional R^2 (%)"); ax[1, 0].legend(); ax[1, 0].grid(alpha=.25)

    items = sorted(grp.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in items]; vals = [v / (tot_vol ** 2) * 100 for _, v in items]
    colors = ["#d62728" if k == "Specific" else "#9467bd" if k in ("Market", "Industry") else "#1f77b4" for k in labels]
    ax[1, 1].bar(range(len(labels)), vals, color=colors)
    ax[1, 1].set_xticks(range(len(labels))); ax[1, 1].set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax[1, 1].axhline(0, color="k", lw=.5)
    ax[1, 1].set_title(f"Risk decomposition (% of variance) — total vol {tot_vol*100:.1f}%")
    ax[1, 1].grid(alpha=.25)

    plt.tight_layout()
    os.makedirs(DOCS, exist_ok=True)
    out = os.path.join(DOCS, "barra.png")
    plt.savefig(out, dpi=110); print("saved", out)


if __name__ == "__main__":
    main()

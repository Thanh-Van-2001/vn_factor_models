# -*- coding: utf-8 -*-
"""
Evaluation protocol for the VN factor library (report Section 3.4).

For every factor (signed, sector/size-neutralised score from factors.py) we report:
  - rank Information Coefficient: monthly Spearman(score_t, fwd_ret_{t->t+1}),
    its mean, t-statistic (mean/se), and information ratio (mean/std).
  - quintile spread: equal-weight Q5 minus Q1 forward return (annualised) and a
    monotonicity check across the five quintiles.
  - long-only tilt: Q5 minus universe mean (what survives the short-sale constraint).
  - signal decay: IC at horizons h = 1, 2, 3 months.
  - net-of-cost spread: gross Q5-Q1 minus a Vietnamese cost stack applied to realised
    portfolio turnover (broker + ad-valorem sell tax).

Also builds the composite (report Section 6.1): family = mean of member scores;
composite = conviction-weighted sum of families; evaluated the same way.

Outputs: results/factor_ic.csv, results/composite.csv, docs/factor_results.png
"""
import os, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
from factors import ALL_FACTORS, FAMILY, FACTORS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(ROOT, "results")
DOCS = os.path.join(ROOT, "docs")
os.makedirs(RESULTS, exist_ok=True)

COST_SIDE = 0.0025          # ~0.15% broker + ~0.10% ad-valorem sell tax, blended per side
N_QUANTILES = 5
MIN_NAMES = 25              # skip thin cross-sections (early sample) that give noisy IC


def spearman_ic(g, col):
    s = g[[col, "fwd_ret"]].dropna()
    if len(s) < MIN_NAMES:
        return np.nan
    return stats.spearmanr(s[col], s["fwd_ret"]).correlation


def ic_stats(panel, col):
    ics = panel.groupby("date").apply(lambda g: spearman_ic(g, col)).dropna()
    if len(ics) < 6:
        return None
    mean = ics.mean()
    t = mean / ics.std() * np.sqrt(len(ics)) if ics.std() else np.nan
    ir = mean / ics.std() if ics.std() else np.nan
    return dict(meanIC=mean, ICt=t, ICIR=ir, n_months=len(ics))


def quantile_spread(panel, col):
    """Equal-weight quintile forward returns; returns (q_means, spread_ann, monotonic,
    longonly_ann, net_spread_ann)."""
    qrets = {q: [] for q in range(1, N_QUANTILES + 1)}
    uni_rets, q5_names_prev, q1_names_prev = [], None, None
    q5_turn, q1_turn = [], []
    for t, g in panel.groupby("date"):
        s = g[[col, "fwd_ret", "symbol"]].dropna()
        if len(s) < N_QUANTILES * 4:
            continue
        s = s.copy()
        s["q"] = pd.qcut(s[col].rank(method="first"), N_QUANTILES, labels=range(1, N_QUANTILES + 1))
        uni_rets.append(s["fwd_ret"].mean())
        for q in range(1, N_QUANTILES + 1):
            qrets[q].append(s.loc[s["q"] == q, "fwd_ret"].mean())
        q5n = set(s.loc[s["q"] == N_QUANTILES, "symbol"])
        q1n = set(s.loc[s["q"] == 1, "symbol"])
        if q5_names_prev is not None:
            q5_turn.append(1 - len(q5n & q5_names_prev) / max(len(q5n), 1))
            q1_turn.append(1 - len(q1n & q1_names_prev) / max(len(q1n), 1))
        q5_names_prev, q1_names_prev = q5n, q1n
    if not uni_rets:
        return None
    qm = {q: np.nanmean(qrets[q]) for q in range(1, N_QUANTILES + 1)}
    spread_m = qm[N_QUANTILES] - qm[1]
    longonly_m = qm[N_QUANTILES] - np.nanmean(uni_rets)
    diffs = [qm[q + 1] - qm[q] for q in range(1, N_QUANTILES)]
    monotonic = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
    turn = np.nanmean(q5_turn + q1_turn) if (q5_turn or q1_turn) else 1.0
    net_spread_m = spread_m - turn * 2 * COST_SIDE        # both legs traded each month
    return dict(Q1=qm[1] * 12, Q2=qm[2] * 12, Q3=qm[3] * 12, Q4=qm[4] * 12, Q5=qm[5] * 12,
                spread_ann=spread_m * 12, longonly_ann=longonly_m * 12, monotonic=monotonic,
                turnover=turn, net_spread_ann=net_spread_m * 12)


def decay(panel, col, max_h=3):
    """IC at horizons h=1..max_h. score at date t vs return of the period starting h-1
    months later."""
    wide_score = panel.pivot_table(index="date", columns="symbol", values=col)
    wide_fwd = panel.pivot_table(index="date", columns="symbol", values="fwd_ret")
    # pivot_table drops all-NaN dates per column -> reindex both to a common date axis so
    # positional (iloc) alignment of score[t] vs fwd[t+h-1] is correct.
    dates = wide_score.index.union(wide_fwd.index)
    wide_score = wide_score.reindex(dates)
    wide_fwd = wide_fwd.reindex(dates)
    out = {}
    for h in range(1, max_h + 1):
        ics = []
        for i in range(len(dates) - (h - 1)):
            sc = wide_score.iloc[i]
            fr = wide_fwd.iloc[i + h - 1]
            df = pd.concat([sc, fr], axis=1).dropna()
            if len(df) >= MIN_NAMES:
                ics.append(stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1]).correlation)
        out[f"IC_h{h}"] = np.nanmean(ics) if ics else np.nan
    return out


def evaluate_one(panel, col):
    r = {"factor": col}
    ics = ic_stats(panel, col)
    if ics:
        r.update(ics)
    qs = quantile_spread(panel, col)
    if qs:
        r.update(qs)
    r.update(decay(panel, col))
    return r


def build_composite(scores):
    """Family score = mean of member factor scores; composite = conviction-weighted sum.
    Per report 6.1: overweight quality & value, treat momentum cautiously, and DEMOTE
    liquidity/size to a control (excluded from the directional composite)."""
    weights = {"Value": 0.35, "Quality": 0.30, "Momentum": 0.10, "Risk": 0.25}
    comp = scores[["date", "symbol", "industry", "mcap", "fwd_ret"]].copy()
    for fam, facs in FAMILY.items():
        comp[fam] = scores[facs].mean(axis=1)
    comp["COMPOSITE"] = sum(weights[f] * comp[f] for f in weights)
    return comp, weights


def main():
    scores = pd.read_parquet(os.path.join(FACTORS, "scores.parquet"))
    print(f"Loaded scores: {len(scores):,} rows, {scores['date'].nunique()} months, "
          f"{scores['date'].min().date()} -> {scores['date'].max().date()}")

    rows = [evaluate_one(scores, f) for f in ALL_FACTORS]
    res = pd.DataFrame(rows).set_index("factor")
    res["family"] = {f: fam for fam, facs in FAMILY.items() for f in facs}
    res = res[["family", "meanIC", "ICt", "ICIR", "n_months", "spread_ann", "net_spread_ann",
               "longonly_ann", "monotonic", "turnover", "IC_h1", "IC_h2", "IC_h3"]]
    res.to_csv(os.path.join(RESULTS, "factor_ic.csv"))

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n=== FACTOR EVALUATION (monthly, signed so +IC = works as report expects) ===")
    show = res.copy()
    for c in ["meanIC", "ICt", "ICIR"]:
        show[c] = show[c].round(3)
    for c in ["spread_ann", "net_spread_ann", "longonly_ann", "turnover", "IC_h1", "IC_h2", "IC_h3"]:
        show[c] = (show[c] * (100 if c.endswith("ann") else 1)).round(2 if not c.endswith("ann") else 1)
    print(show.to_string())

    # composite
    comp, weights = build_composite(scores)
    comp_rows = [evaluate_one(comp, f) for f in list(FAMILY) + ["COMPOSITE"]]
    cres = pd.DataFrame(comp_rows).set_index("factor")
    cres = cres[["meanIC", "ICt", "ICIR", "spread_ann", "net_spread_ann", "longonly_ann", "monotonic"]]
    cres.to_csv(os.path.join(RESULTS, "composite.csv"))
    print(f"\n=== COMPOSITE (weights {weights}) ===")
    cshow = cres.copy()
    for c in ["meanIC", "ICt", "ICIR"]:
        cshow[c] = cshow[c].round(3)
    for c in ["spread_ann", "net_spread_ann", "longonly_ann"]:
        cshow[c] = (cshow[c] * 100).round(1)
    print(cshow.to_string())

    _plot(res, cres)
    print(f"\nSaved results/factor_ic.csv, results/composite.csv, docs/factor_results.png")
    return res, cres


def _plot(res, cres):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    r = res.sort_values("meanIC")
    colors = ["#1b6ca8" if t else "#bbbbbb" for t in (r["ICt"].abs() > 1.96)]
    ax[0].barh(r.index, r["meanIC"], color=colors)
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].set_title("Mean rank IC by factor (blue: |t|>1.96)")
    ax[0].set_xlabel("mean monthly rank IC")
    fams = [c for c in cres.index if c != "COMPOSITE"] + ["COMPOSITE"]
    sp = cres.loc[fams, "spread_ann"] * 100
    ax[1].bar(fams, sp, color=["#1b6ca8"] * (len(fams) - 1) + ["#c0392b"])
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title("Annualised Q5-Q1 spread: families + composite")
    ax[1].set_ylabel("% / year")
    ax[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "factor_results.png"), dpi=110)


if __name__ == "__main__":
    main()

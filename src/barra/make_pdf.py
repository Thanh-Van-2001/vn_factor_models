# -*- coding: utf-8 -*-
"""Generate a PDF report for the VN Barra-style risk model (fpdf2). English text to avoid
unicode-font issues. Reads the saved model outputs so numbers stay in sync with the run."""
import os
import numpy as np
import pandas as pd
from fpdf import FPDF

BARRA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "barra")
DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs")
STYLES = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "LIQUIDITY", "VALUE", "LEVERAGE", "QUALITY"]

F = pd.read_parquet(os.path.join(BARRA, "factor_returns.parquet"))
R2 = pd.read_parquet(os.path.join(BARRA, "cs_r2.parquet"))["r2"]
spec = pd.read_parquet(os.path.join(BARRA, "specific_returns.parquet"))
panel = pd.read_parquet(os.path.join(BARRA, "exposures.parquet"))
nwk = len(F); nstk = panel["symbol"].nunique()

# factor significance
frows = []
for s in STYLES:
    x = F[s].dropna()
    frows.append((s, x.mean()*52*100, x.std()*np.sqrt(52)*100,
                  x.mean()/x.std()*np.sqrt(len(x)) if x.std() else 0,
                  (x.mean()*52)/(x.std()*np.sqrt(52)) if x.std() else 0))

# risk decomposition (equal-weight, latest week)
last = panel[panel["date"] == panel["date"].max()].copy()
inds = sorted(last["ind"].unique())
cols = ["country"] + [f"IND_{m}" for m in inds] + STYLES
N = len(last); w = np.full(N, 1.0/N)
X = pd.DataFrame(np.hstack([np.ones((N,1)),
                            np.column_stack([(last["ind"]==m).to_numpy(float) for m in inds]),
                            last[STYLES].to_numpy(float)]), columns=cols, index=last["symbol"].values)
Fc = F.reindex(columns=cols).fillna(0.0); cov = Fc.cov()*52
b = X.T @ w
fac_var = float(b.values @ cov.values @ b.values)
svar = (spec.var()*52).reindex(last["symbol"].values).fillna(spec.var().mean()*52)
spec_var = float(np.sum(w**2 * svar.values)); tot = np.sqrt(fac_var+spec_var)
mc = (cov.values @ b.values) * b.values
grp = {"Market": mc[cols.index("country")],
       "Industry (all)": mc[[cols.index(f"IND_{m}") for m in inds]].sum(),
       **{s: mc[cols.index(s)] for s in STYLES}, "Specific": spec_var}

pdf = FPDF(); pdf.set_auto_page_break(True, 15); pdf.add_page()
def H(t, s=14): pdf.set_font("Helvetica","B",s); pdf.cell(0,8,t,ln=1); pdf.ln(1)
def P(t): pdf.set_font("Helvetica","",10); pdf.multi_cell(0,5,t); pdf.ln(1)

pdf.set_font("Helvetica","B",18); pdf.cell(0,10,"VN Equities - Barra-style Risk Model",ln=1)
pdf.set_font("Helvetica","",10)
pdf.cell(0,6,f"Cross-sectional factor risk model | VN100 universe | 2021-2026 | {nwk} weekly fits",ln=1)
pdf.ln(3)
H("1. Overview")
P("A Barra-type cross-sectional risk model for the Vietnamese stock market, built alongside the "
  "existing Fama-French VN-3/VN-4 academic factors. Each week we regress stock returns on per-stock "
  "exposures (8 style factors + industry dummies) to estimate factor returns and a factor covariance "
  "matrix, then decompose portfolio risk. Goal: risk measurement/forecasting, not an alpha claim. "
  "Data is real (vnstock); no look-ahead (price descriptors use data up to t; fundamentals lag 45 days).")
H("2. Methodology")
P("Model per week:  r_i = f_country + SUM_m X_ind(i,m)*f_ind,m + SUM_s X_style(i,s)*f_style,s + u_i,\n"
  "with identification constraint SUM_m capwt_m * f_ind,m = 0 and regression weights sqrt(market cap), "
  "solved exactly via the KKT system. Styles standardised cross-sectionally each week (cap-weighted mean 0, "
  "std 1, winsorised +/-3 MAD): SIZE, BETA, MOMENTUM (12-1m), RESVOL (120d), LIQUIDITY (60d turnover), "
  "VALUE (EP+BM), LEVERAGE (Debt/Equity), QUALITY (ROE). Industries (>=4 names): Banks, Real estate, "
  "Securities, Utilities, Food & beverage, Construction, Building materials, Plastics-chemicals, Transport, Other.")
H("3. Factor returns (annualised) and significance")
pdf.set_font("Helvetica","B",9)
for h,wd in [("Style",34),("Ann.ret %",26),("Ann.vol %",26),("t-stat",22),("IR",18)]:
    pdf.cell(wd,6,h,border=1)
pdf.ln()
pdf.set_font("Helvetica","",9)
for s,ann,vol,t,ir in frows:
    for v,wd,f in [(s,34,"%s"),(ann,26,"%+.2f"),(vol,26,"%.2f"),(t,22,"%+.2f"),(ir,18,"%+.2f")]:
        pdf.cell(wd,6,(f % v),border=1)
    pdf.ln()
pdf.ln(1)
P(f"Average cross-sectional R^2 = {R2.mean()*100:.1f}%  (factors explain a large share of return dispersion). "
  "No style premium is statistically significant over this window (all |t| < 1.3) - expected for a thin "
  "VN100 universe over ~5 years. This is a risk model, not an alpha claim.")
H("4. Portfolio risk decomposition (equal-weight, latest week)")
P(f"Total annualised volatility = {tot*100:.1f}%   "
  f"(factor {np.sqrt(max(fac_var,0))*100:.1f}% / specific {np.sqrt(spec_var)*100:.1f}%). "
  "Idiosyncratic risk is almost fully diversified; risk is dominated by Market + common style factors.")
pdf.set_font("Helvetica","B",9); pdf.cell(60,6,"Source",border=1); pdf.cell(40,6,"% of variance",border=1); pdf.ln()
pdf.set_font("Helvetica","",9)
for k,v in sorted(grp.items(), key=lambda kv:-kv[1]):
    pdf.cell(60,6,k,border=1); pdf.cell(40,6,"%+.1f%%" % (v/tot**2*100),border=1); pdf.ln()

pdf.add_page(); H("5. Charts")
img = os.path.join(DOCS,"barra.png")
if os.path.exists(img): pdf.image(img, w=185)
pdf.ln(2)
H("6. Honest caveats / next steps")
P(f"- Only ~{nstk} names usable (vnstock community rate-limit capped the fetch at 79/100); effective "
  "~46-47 stocks/week after a 1-year history filter - thin for ~19 factors, so estimates are noisy.\n"
  "- BETA uses a median-market proxy; fundamentals are lagged quarterly.\n"
  "- Covariance is the sample estimate; production would add EWMA + Newey-West and bias-statistic "
  "backtests (does forecast vol match realised?).\n"
  "- Next: expand to HOSE (~400 names) for a firmer cross-section; add factor-risk attribution for live portfolios.")

out = os.path.join(DOCS, "Barra_VN_Report.pdf")
pdf.output(out); print("saved", out)

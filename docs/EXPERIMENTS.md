# Empirical test of the BeeTrade VN factor library

This note reports an **independent replication** of the cross-sectional factors in
[`Vietnam_Equity_Factor_Library_Bee.pdf`](Vietnam_Equity_Factor_Library_Bee.pdf)
(BeeTrade, *A Cross-Sectional Factor Library for the Vietnamese Equity Market*, June 2026)
on **live vnstock data**, following the report's own methodology (Section 3) and evaluation
protocol (Section 3.4). The goal is to check which of the report's qualitative "Vietnam evidence"
claims actually hold in the data, and to size each factor honestly.

Pipeline: [`src/fetch_data.py`](../src/fetch_data.py) → [`src/fetch_fund.py`](../src/fetch_fund.py)
→ [`src/factors.py`](../src/factors.py) → [`src/evaluate.py`](../src/evaluate.py)
→ [`src/zai_review.py`](../src/zai_review.py).

![results](factor_results.png)

## 1. Data and universe

| | |
|---|---|
| **Universe** | VN100 ∪ HNX30 = **130 liquid names** (HOSE 100, HNX 30) — the float/liquidity-screened investable set the report (§3.1) says factors should be built on |
| **Prices** | daily adjusted OHLCV from vnstock (VCI), 2016–2026 |
| **Fundamentals** | **annual** income statement + balance sheet, FY2022–2025 (see data caveat) |
| **Rebalance** | monthly (last trading day); forward 1-month total return as the target |
| **Sample** | price factors ~2018–2026 (≈119–129 months); fundamental factors ≈2023–2026 (39 months) |

**Data caveat (important).** The vnstock community tier returns only the **4 most recent periods**
of financial statements, and the VCI `ratio` endpoint is stuck on a frozen 2018 snapshot. We
therefore build fundamentals from the **annual** income/balance statements (which reliably return
FY2022–2025), derive shares outstanding from the EPS identity, and carry each fiscal year forward
with a 90-day reporting lag (point-in-time, no look-ahead). The consequence is that **Value/Quality
factors are tested on a short 39-month window** and should be read as indicative, not definitive.
Price-based factors (Momentum/Risk/Liquidity) use the full price history.

## 2. Methodology (as specified in the report §3)

- **Robust standardisation** (Eq. 1): winsorise at 1/99 pct, then `z = (x − median) / MAD`.
- **Sector + size neutralisation** (Eq. 2): residual of `z` on industry dummies + `log(mcap)`.
- **Sign convention** (§3.3): scores are signed so that **a positive mean rank IC means the factor
  works in the direction the report expects** (e.g. low-volatility, low-leverage are flipped).
- **Value/yield against live price**: per-share fundamentals are divided by the *current* price so
  value factors respond to price, not stale.
- Financials (banks/securities/insurance) are excluded from EV/asset-scaled factors (§ family notes).
- **Evaluation** (§3.4): monthly rank IC + t-stat + IR; quintile Q5−Q1 spread + monotonicity;
  long-only tilt (Q5 − universe, given the short-sale constraint); IC decay at 1–3 months;
  net-of-cost spread using a VN cost stack (~0.15% broker + ~0.10% ad-valorem sell tax per side).

## 3. Factor results (signed; +IC = works as report expects)

Monthly rank IC, its t-stat, annualised Q5−Q1 spread (gross and net of cost), and the long-only
tilt. `|t| > 1.96` ≈ significant. Full numbers in [`results/factor_ic.csv`](../results/factor_ic.csv).

| Factor | Family | mean IC | IC t | Q5−Q1 %/yr | net %/yr | n (mo) |
|---|---|---:|---:|---:|---:|---:|
| **EP** | Value | **+0.038** | **+2.27** | +11.9 | +11.4 | 39 |
| BP | Value | +0.031 | +1.40 | +20.6 | +20.1 | 39 |
| SP | Value | +0.016 | +1.15 | +12.6 | +12.3 | 39 |
| OP | Quality | +0.015 | +0.68 | −8.3 | −8.5 | 39 |
| GP | Quality | −0.017 | −0.73 | −5.7 | −5.9 | 39 |
| ROE | Quality | +0.015 | +0.68 | −7.0 | −7.2 | 39 |
| ROA | Quality | −0.005 | −0.25 | −7.9 | −8.0 | 39 |
| LEV (low) | Quality | −0.004 | −0.26 | −1.3 | −1.4 | 39 |
| AG (low) | Quality | −0.009 | −0.42 | −8.5 | −9.0 | 26 |
| MOM (12-1) | Momentum | −0.006 | −0.45 | −3.6 | −5.1 | 119 |
| REV (1m) | Momentum | −0.005 | −0.36 | −1.9 | −6.4 | 129 |
| PH52 | Momentum | +0.003 | +0.19 | −3.5 | −5.4 | 129 |
| BETA (low) | Risk | +0.017 | +0.84 | −0.7 | −1.3 | 128 |
| **IVOL (low)** | Risk | **+0.029** | **+2.40** | +0.7 | −0.4 | 128 |
| MAX (low) | Risk | +0.009 | +0.68 | −4.3 | −7.8 | 127 |
| SKEW (low) | Risk | +0.011 | +1.06 | +1.9 | +0.0 | 129 |
| SIZE (small) | Liquidity | +0.014 | +1.21 | +8.2 | +7.7 | 129 |
| TURN (low) | Liquidity | +0.025 | +1.70 | +6.7 | +6.4 | 129 |
| ILLIQ | Liquidity | −0.008 | −0.59 | −4.1 | −4.5 | 129 |
| ZERORET (low) | Liquidity | −0.015 | −1.17 | −7.4 | −8.3 | 129 |

**Composite** (family = mean of member scores; weights overweight Value/Quality, momentum cautious,
liquidity demoted to a control per §6.1 → Value 0.35, Quality 0.30, Risk 0.25, Momentum 0.10).
Full numbers in [`results/composite.csv`](../results/composite.csv).

| | mean IC | IC t | Q5−Q1 %/yr | net %/yr | monotonic |
|---|---:|---:|---:|---:|:---:|
| **Value** | **+0.047** | **+2.81** | +22.6 | +22.0 | **yes** |
| Quality | −0.002 | −0.07 | −9.0 | −9.3 | no |
| Momentum | −0.006 | −0.47 | −8.3 | −11.2 | no |
| Risk | +0.025 | +1.88 | −2.1 | −4.4 | no |
| Liquidity (control) | −0.008 | −0.56 | −3.9 | −4.2 | no |
| **COMPOSITE** | **+0.035** | +1.88 | +11.3 | +10.2 | no |

## 4. Verdict vs the report's "Vietnam evidence" claims

Generated by **GLM-4.6 (via z.ai)** from the numeric results above
([`results/zai_analysis.md`](../results/zai_analysis.md)), cross-checked against the tables.

| Report claim | Verdict | Evidence |
|---|---|---|
| OP is the **strongest** profitability proxy | **Contradicted** | OP insignificant (t=0.68), tied with ROE; Quality composite negative (t=−0.07) |
| **Value is real** and non-redundant | **Confirmed** ✅ | Value composite t=2.81, monotonic, +22.6%/yr; EP the driver (t=2.27) |
| Short-term **reversal is one of the most robust** signals | **Contradicted** (at monthly cadence) | REV t=−0.36; ~75%/mo turnover makes net spread −6.4%/yr |
| Cross-sectional **momentum is weak / regime-dependent** | **Confirmed** ✅ | MOM t=−0.45; momentum composite t=−0.47 |
| **Size weak; illiquidity muted/inverted** | **Confirmed** ✅ | SIZE t=1.21 (insig); ILLIQ IC=−0.008 (inverted/muted) |
| **Low-risk family generally works** | **Inconclusive** | only IVOL significant (t=2.40); BETA/MAX not; Risk composite t=1.88 |
| Low asset-growth / low leverage are defensive value-adds | **Contradicted** (this sample) | AG t=−0.42, LEV t=−0.26 |

## 5. Caveats

1. **Short fundamental sample.** Value/Quality rest on 39 months (FY2022–2025 annual data) — the
   strong Value t-stat is encouraging but window-sensitive. Quarterly PIT fundamentals (a paid data
   tier) would firm this up.
2. **Thin liquid universe (~130 names).** This deliberately excludes the small/illiquid tail, which
   biases *against* Size and Illiquidity — consistent with, but not a clean test of, those claims.
3. **No short side.** VN has no reliable borrow; the Q5−Q1 spreads are theoretical. The long-only
   tilt column (Q5 − universe) is what survives in practice.
4. **Price-limit censoring** (±7% HOSE) mechanically dampens reversal/MAX/skewness at the tails.
5. **Reversal cadence.** The report's reversal claim is a *1-month-horizon* effect that decays by
   ~2 months; our monthly rebalance with sector-neutralisation and realistic costs does not capture
   it. A weekly, cost-aware reversal sleeve is the right follow-up test.

## 6. Reproduce

```bash
pip install -r requirements.txt
python src/fetch_data.py          # universe + daily prices + (legacy) quarterly ratios
python src/fetch_fund.py          # annual income + balance statements
python src/factors.py             # PIT characteristics -> signed neutralised scores
python src/evaluate.py            # IC / spreads / decay / composite -> results/ + docs/factor_results.png
ZAI_TOKEN=*** python src/zai_review.py   # GLM-4.6 (z.ai) claim-by-claim analysis
```

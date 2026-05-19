# VN Factor Models — Empirical Research Repo

Replication of factor models for Vietnamese stock market based on:
- **Reference report**: `Factor_Models_Vietnam_Report_EN.docx` (May 2026, BeeTrade)
- **Primary source**: Huang, Liu, Shu (2023), "Factors and anomalies in the Vietnamese stock market", Pacific-Basin Finance Journal Vol. 82.

## VN-4 Model (recommended specification)

```
R_i,t - Rf_t = alpha + b_M*MKT + b_S*SMB + b_E*EMP + b_T*TMH + epsilon
```

| Factor | Construction | Expected sign |
|--------|--------------|---------------|
| MKT    | VW market excess return over VINBOR 1-week | + |
| SMB    | Small minus Big (size, market cap median split) | + |
| EMP    | High EP minus Low EP (earnings/price, 2x3 sort with size) | + |
| TMH    | Low minus High 12-month avg daily turnover ratio | + |

EMP construction (2x3 sort):
- Size: small/big at median market cap
- EP: top 30% / middle 40% / bottom 30%
- 6 portfolios: SE, SN, SP, BE, BN, BP
- EMP = 0.5*(SE+BE) - 0.5*(SP+BP)

TMH construction:
- Sort by 12-month average daily turnover (volume/shares outstanding)
- Low-turnover minus high-turnover long-short
- Why turnover and not bid-ask spread? Huang et al. show turnover proxies for retail SPECULATION (small-cap, low-institutional), not liquidity. Negative size-turnover correlation in VN (-0.09) confirms this.

## Universe

- HOSE + HNX listed (exclude UPCoM)
- Filter: at least 12 months of price+fundamental history
- Point-in-time universe per quarter (avoid look-ahead bias)

## Frequency

- Weekly returns (Wed close to Wed close, sufficient power)
- Monthly returns (robustness check)
- Risk-free: VINBOR 1-week interbank rate (proxy: SBV refinance rate or Treasury bond yield when VINBOR unavailable)

## Pipeline

```
src/
  fetch_universe.py         # HOSE + HNX tickers (vnstock)
  fetch_prices.py           # Daily OHLCV per ticker
  fetch_fundamentals.py     # Quarterly EPS, book value, total assets
  compute_characteristics.py # market cap, EP, BM, turnover, profitability
  construct_factors.py      # MKT, SMB, EMP, TMH (also FF-3 HML for benchmark)
  backtest_anomalies.py     # 21 anomalies x CAPM/FF3/VN3/VN4 alpha tests
  grs_test.py               # Gibbons-Ross-Shanken joint alpha F-test
  compare_models.py         # Adj R-sq, GRS F, max sharpe ratio comparison
```

## Validation against Huang et al. (2023) target results

Annualized factor premia 2007-2022 (target):
- MKT: +9.01%/y
- SMB: +8.64%/y
- EMP: positive significant (specific number ref to paper)
- TMH: positive significant (specific number ref to paper)

VN-4 explains 19/21 anomalies in the Huang catalog. The 2 residuals: short-term reversal + abnormal turnover (microstructure-driven).

If our replication gives premia within +/- 2% of these, factor construction is validated.

## 21 anomalies catalog (Huang 2023)

11 categories:
1. Beta (CAPM beta)
2. Size (market cap)
3. Volatility (total return vol)
4. Idiosyncratic volatility (residual vol after FF3)
5. Turnover (avg daily turnover, multiple windows)
6. Illiquidity (Amihud, bid-ask spread)
7. Reversal (short-term, 1-month)
8. Momentum (3-12 month)
9. 52-week high
10. Value (BM, EP, CP)
11. Profitability (OP, CP, ROE)

Long-short portfolios sorted on each characteristic. Test: long top quintile minus short bottom quintile, alpha vs CAPM/FF3/FF5/VN3/VN4.

## Trading frictions (Vietnam-specific)

- No short selling (KRX changed May 2025+ may enable)
- Daily price limits: 7% HOSE, 10% HNX, 40% first day HOSE
- Foreign ownership caps: 49% most sectors, lower in banking/telecom
- Implication: sort-based factor returns NEED bid-ask + price limit filters in test portfolios

## Quick start

```bash
cd /home/ubuntu/vn_factor_models
source ../holy_grail_fx_mining/venv/bin/activate
python src/fetch_universe.py
python src/fetch_prices.py --start 2017-01-01
python src/fetch_fundamentals.py
python src/construct_factors.py
python src/compare_models.py
```

## Status

- 2026-05-06: repo scaffolded
- TODO: implement scripts, run replication, validate against Huang 2023 target premia
- TODO: extend to 2025-2026 sample (post-KRX, post-FTSE upgrade if happens)

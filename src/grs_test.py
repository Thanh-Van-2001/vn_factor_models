# -*- coding: utf-8 -*-
"""
Gibbons-Ross-Shanken (1989) F-test for joint zero-alpha hypothesis.

Used to compare factor models: which one best prices a set of test portfolios?
The model with non-rejected H0 (alpha=0 for all test assets) wins.

Reference: Gibbons, Ross, Shanken (1989), Econometrica.
"""
import numpy as np
from scipy import stats


def grs_test(alphas, residuals, factor_returns):
    """
    Compute GRS F-statistic.

    Args:
        alphas: np.array shape (N,) — intercepts from time-series regressions of N test portfolios
        residuals: np.array shape (T, N) — residuals from those regressions
        factor_returns: np.array shape (T, K) — factor returns

    Returns:
        dict with F statistic, p-value, df1, df2
    """
    T, N = residuals.shape
    K = factor_returns.shape[1] if factor_returns.ndim > 1 else 1
    if factor_returns.ndim == 1:
        factor_returns = factor_returns.reshape(-1, 1)

    # Sample residual covariance
    sigma_hat = (residuals.T @ residuals) / (T - K - 1)
    # Factor sample mean and covariance
    f_mean = factor_returns.mean(axis=0)
    f_demeaned = factor_returns - f_mean
    omega = (f_demeaned.T @ f_demeaned) / (T - 1)

    # Sharpe ratio of factors squared
    sh_squared_f = f_mean @ np.linalg.inv(omega) @ f_mean

    # GRS F statistic
    sigma_inv = np.linalg.inv(sigma_hat)
    alpha_quad = alphas @ sigma_inv @ alphas
    grs_f = (T - N - K) / N * alpha_quad / (1 + sh_squared_f)

    df1 = N
    df2 = T - N - K
    p_value = 1 - stats.f.cdf(grs_f, df1, df2)

    return {
        "F": grs_f,
        "p_value": p_value,
        "df1": df1,
        "df2": df2,
        "alpha_quad": alpha_quad,
        "factor_sh_sq": sh_squared_f,
        "T": T, "N": N, "K": K,
    }


if __name__ == "__main__":
    # Quick demo with random data
    np.random.seed(42)
    T, N, K = 200, 10, 3
    factor_returns = np.random.randn(T, K) * 0.01
    # Generate test portfolios with NO alpha (correctly priced)
    betas = np.random.randn(N, K) * 0.5
    residuals = np.random.randn(T, N) * 0.02
    alphas = np.zeros(N)
    test_returns = factor_returns @ betas.T + residuals

    res = grs_test(alphas, residuals, factor_returns)
    print(f"GRS test (alpha=0 truth): F={res['F']:.3f}, p={res['p_value']:.3f}")
    print(f"  Should NOT reject H0 (p > 0.05)")

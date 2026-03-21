"""
QuantPricer Engine — Python analytics + C++ bridge for heavy computation.

Two modes:
  1. Pure Python: BS analytics (instant, for interactive sliders)
  2. C++ subprocess: Heston MC, Merton MC, barrier pricing, multi-asset
     (called via build/pricer_service binary)
"""

import numpy as np
from scipy.stats import norm
import subprocess
import os
import io
import pandas as pd

# ============================================================================
# C++ Pricer Service Bridge
# ============================================================================

PRICER_BINARY = os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'pricer_service')

def call_cpp(command: str, extra_stdin: str = "", **params) -> pd.DataFrame:
    """
    Call the C++ pricer service binary.
    Sends key-value params via stdin, reads CSV from stdout.
    Returns a pandas DataFrame.
    """
    input_lines = [f"COMMAND: {command}"]
    for k, v in params.items():
        input_lines.append(f"{k}: {v}")
    input_lines.append("---")
    if extra_stdin:
        input_lines.append(extra_stdin)

    try:
        result = subprocess.run(
            [PRICER_BINARY],
            input="\n".join(input_lines),
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(f"C++ pricer error: {result.stderr}")
        return pd.read_csv(io.StringIO(result.stdout))
    except FileNotFoundError:
        raise RuntimeError(
            f"C++ pricer binary not found at {PRICER_BINARY}. "
            "Run 'make release' first to build it."
        )

def cpp_available() -> bool:
    """Check if the C++ pricer binary exists."""
    return os.path.isfile(PRICER_BINARY)

# ============================================================================
# C++ Commands (thin wrappers)
# ============================================================================

def heston_smile(S0=100, K_min=70, K_max=130, K_step=2, r=0.05, T=1.0,
                 v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, num_paths=50000):
    return call_cpp("heston_smile", S0=S0, K_min=K_min, K_max=K_max, K_step=K_step,
                    r=r, T=T, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
                    num_paths=num_paths)

def merton_smile(S0=100, K_min=70, K_max=130, K_step=2, r=0.05, T=1.0,
                 sigma=0.2, lam=1.0, mu_j=-0.1, sigma_j=0.15, num_paths=50000):
    return call_cpp("merton_smile", S0=S0, K_min=K_min, K_max=K_max, K_step=K_step,
                    r=r, T=T, sigma=sigma, **{"lambda": lam}, mu_j=mu_j,
                    sigma_j=sigma_j, num_paths=num_paths)

def barrier_curve(S0=100, K=100, r=0.05, T=1.0, sigma=0.2,
                  H_min=50, H_max=99, H_step=1):
    return call_cpp("barrier_curve", S0=S0, K=K, r=r, T=T, sigma=sigma,
                    H_min=H_min, H_max=H_max, H_step=H_step)

def multi_asset_corr(S0=100, K=100, r=0.05, T=1.0, sigma=0.2, N=2,
                     rho_min=-0.4, rho_max=0.99, rho_step=0.05, num_paths=50000):
    return call_cpp("multi_asset", S0=S0, K=K, r=r, T=T, sigma=sigma, N=N,
                    rho_min=rho_min, rho_max=rho_max, rho_step=rho_step,
                    num_paths=num_paths)

def batch_implied_vol(S0, r, options_data):
    """
    Compute IV for a batch of options via C++.
    options_data: list of (price, strike, T, is_call_int)
    """
    lines = "\n".join(f"{p},{k},{t},{c}" for p, k, t, c in options_data)
    return call_cpp("batch_iv", extra_stdin=lines, S0=S0, r=r, count=len(options_data))

# ============================================================================
# Pure Python: Black-Scholes Analytics (for instant slider response)
# ============================================================================

def bs_d1(S, K, r, T, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def bs_call(S, K, r, T, sigma):
    d1 = bs_d1(S, K, r, T, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, r, T, sigma):
    d1 = bs_d1(S, K, r, T, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_delta_call(S, K, r, T, sigma):
    return norm.cdf(bs_d1(S, K, r, T, sigma))

def bs_delta_put(S, K, r, T, sigma):
    return bs_delta_call(S, K, r, T, sigma) - 1.0

def bs_gamma(S, K, r, T, sigma):
    return norm.pdf(bs_d1(S, K, r, T, sigma)) / (S * sigma * np.sqrt(T))

def bs_vega(S, K, r, T, sigma):
    return S * norm.pdf(bs_d1(S, K, r, T, sigma)) * np.sqrt(T)

def bs_theta_call(S, K, r, T, sigma):
    d1 = bs_d1(S, K, r, T, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)

def bs_theta_put(S, K, r, T, sigma):
    d1 = bs_d1(S, K, r, T, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)

def bs_rho_call(S, K, r, T, sigma):
    d2 = bs_d1(S, K, r, T, sigma) - sigma * np.sqrt(T)
    return K * T * np.exp(-r * T) * norm.cdf(d2)

def bs_rho_put(S, K, r, T, sigma):
    d2 = bs_d1(S, K, r, T, sigma) - sigma * np.sqrt(T)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)

def implied_vol(market_price, S, K, r, T, is_call=True, sigma_init=0.3):
    sigma = sigma_init
    for _ in range(100):
        price = bs_call(S, K, r, T, sigma) if is_call else bs_put(S, K, r, T, sigma)
        vega = bs_vega(S, K, r, T, sigma)
        diff = price - market_price
        if abs(diff) < 1e-8:
            return sigma
        if abs(vega) < 1e-14:
            return np.nan
        sigma -= diff / vega
        sigma = max(sigma, 1e-6)
    return sigma

def nelson_siegel(t, beta0, beta1, beta2, tau):
    t = np.asarray(t, dtype=float)
    x = np.maximum(t / tau, 1e-10)
    ex = np.exp(-x)
    g1 = (1 - ex) / x
    g2 = g1 - ex
    return beta0 + beta1 * g1 + beta2 * g2

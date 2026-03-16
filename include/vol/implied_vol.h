#ifndef QUANTPRICER_IMPLIED_VOL_H
#define QUANTPRICER_IMPLIED_VOL_H

// ============================================================================
// Implied Volatility Computation — Book Chapter 13
// Ch 13.1: Motivation — why we need implied vol from market prices
// Ch 13.2: Root-finding algorithms overview
// Ch 13.3: Interval Bisection method with function templates (Ch 5 callback)
// Ch 13.4: Newton-Raphson using Vega as the derivative, pointer-to-member
// Also connects to: Ch 7 (function objects used as pricing callbacks)
// ============================================================================

#include "greeks/black_scholes.h"
#include <cmath>
#include <stdexcept>
#include <functional>

/// Result of implied volatility computation
struct ImpliedVolResult {
    double sigma;          // Implied volatility
    int iterations;        // Number of iterations taken
    double residual;       // Final |f(sigma) - target|
    bool converged;
};

// ============================================================================
// Method 1: Interval Bisection — Ch 13.3
// ============================================================================

/// Generic bisection root-finder — Ch 13.3.1, 13.3.2
/// Finds x in [lo, hi] such that f(x) = target
/// Uses function templates to accept any callable (functor pattern from Ch 7)
template <typename F>
double bisection(F f, double target, double lo, double hi,
                 double tol = 1e-8, int max_iter = 200)
{
    double f_lo = f(lo) - target;
    double f_hi = f(hi) - target;

    if (f_lo * f_hi > 0.0)
        throw std::runtime_error("Bisection: f(lo) and f(hi) must bracket the root");

    for (int i = 0; i < max_iter; ++i) {
        double mid = 0.5 * (lo + hi);
        double f_mid = f(mid) - target;

        if (std::abs(f_mid) < tol || (hi - lo) < tol)
            return mid;

        if (f_lo * f_mid < 0.0) {
            hi = mid;
            f_hi = f_mid;
        } else {
            lo = mid;
            f_lo = f_mid;
        }
    }
    return 0.5 * (lo + hi);  // Best estimate
}

/// Implied volatility via bisection — Ch 13.3.2
/// Uses Black-Scholes call price as the pricing function
inline ImpliedVolResult implied_vol_bisection(
    double market_price, double S, double K, double r, double T,
    double sigma_lo = 0.001, double sigma_hi = 5.0,
    double tol = 1e-8, int max_iter = 200)
{
    auto bs_price = [&](double sigma) {
        return bs::call_price(S, K, r, T, sigma);
    };

    double f_lo = bs_price(sigma_lo) - market_price;
    double f_hi = bs_price(sigma_hi) - market_price;

    if (f_lo * f_hi > 0.0)
        return {0.0, 0, std::abs(f_lo), false};

    double lo = sigma_lo, hi = sigma_hi;
    int iter = 0;

    for (; iter < max_iter; ++iter) {
        double mid = 0.5 * (lo + hi);
        double f_mid = bs_price(mid) - market_price;

        if (std::abs(f_mid) < tol)
            return {mid, iter, std::abs(f_mid), true};

        if (f_lo * f_mid < 0.0) {
            hi = mid;
        } else {
            lo = mid;
            f_lo = f_mid;
        }
    }

    double best = 0.5 * (lo + hi);
    return {best, iter, std::abs(bs_price(best) - market_price), false};
}

// ============================================================================
// Method 2: Newton-Raphson — Ch 13.4
// ============================================================================

/// Implied volatility via Newton-Raphson — Ch 13.4.1, 13.4.3
/// Uses Vega as the derivative: σ_{n+1} = σ_n - (C(σ_n) - C_mkt) / Vega(σ_n)
/// Pointer-to-member-function concept from Ch 13.4.2
inline ImpliedVolResult implied_vol_newton(
    double market_price, double S, double K, double r, double T,
    double sigma_init = 0.3, double tol = 1e-8, int max_iter = 100)
{
    double sigma = sigma_init;

    for (int iter = 0; iter < max_iter; ++iter) {
        double price = bs::call_price(S, K, r, T, sigma);
        double v = bs::vega(S, K, r, T, sigma);

        double diff = price - market_price;

        if (std::abs(diff) < tol)
            return {sigma, iter, std::abs(diff), true};

        // Guard against zero Vega (deep ITM/OTM or near expiry)
        if (std::abs(v) < 1e-14) {
            // Fall back to bisection
            return implied_vol_bisection(market_price, S, K, r, T);
        }

        sigma -= diff / v;

        // Ensure sigma stays positive
        sigma = std::max(sigma, 1e-6);
    }

    return {sigma, max_iter, std::abs(bs::call_price(S, K, r, T, sigma) - market_price), false};
}

// ============================================================================
// Volatility Surface Builder — Extension beyond the book
// ============================================================================

/// Compute implied vol across a grid of strikes and maturities
/// Returns a 2D grid: rows = maturities, cols = strikes
struct VolSurface {
    std::vector<double> strikes;
    std::vector<double> maturities;
    std::vector<std::vector<double>> vols;  // vols[t_idx][k_idx]
};

inline VolSurface build_vol_surface(
    const std::vector<double>& strikes,
    const std::vector<double>& maturities,
    const std::vector<std::vector<double>>& market_prices,  // prices[t][k]
    double S, double r)
{
    VolSurface surface;
    surface.strikes = strikes;
    surface.maturities = maturities;
    surface.vols.resize(maturities.size());

    for (size_t t = 0; t < maturities.size(); ++t) {
        surface.vols[t].resize(strikes.size());
        for (size_t k = 0; k < strikes.size(); ++k) {
            auto result = implied_vol_newton(
                market_prices[t][k], S, strikes[k], r, maturities[t]);
            surface.vols[t][k] = result.converged ? result.sigma : 0.0;
        }
    }
    return surface;
}

#endif // QUANTPRICER_IMPLIED_VOL_H

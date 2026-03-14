#ifndef QUANTPRICER_BLACK_SCHOLES_H
#define QUANTPRICER_BLACK_SCHOLES_H

// ============================================================================
// Black-Scholes Analytic Pricing — Book Chapters 3, 10
// Ch 3.7:  calc_call_price / calc_put_price implementations
// Ch 10.1: Black-Scholes analytic pricing formula derivation context
// Ch 11.1: Analytic Greeks (Delta, Gamma, Vega, Theta, Rho)
// ============================================================================

#include "option/option.h"
#include "rng/rng.h"
#include <cmath>

namespace bs {

    // Standard normal CDF — shared utility
    inline double N(double x) {
        return 0.5 * std::erfc(-x * M_SQRT1_2);
    }

    // Standard normal PDF
    inline double n(double x) {
        return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
    }

    /// d1 and d2 from Black-Scholes formula — Ch 10.1
    inline double d1(double S, double K, double r, double T, double sigma) {
        return (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    }

    inline double d2(double S, double K, double r, double T, double sigma) {
        return d1(S, K, r, T, sigma) - sigma * std::sqrt(T);
    }

    /// European Call price: C = S*N(d1) - K*exp(-rT)*N(d2) — Ch 3.7, 10.1
    inline double call_price(double S, double K, double r, double T, double sigma) {
        double d_1 = d1(S, K, r, T, sigma);
        double d_2 = d_1 - sigma * std::sqrt(T);
        return S * N(d_1) - K * std::exp(-r * T) * N(d_2);
    }

    /// European Put price: P = K*exp(-rT)*N(-d2) - S*N(-d1) — Ch 3.7
    inline double put_price(double S, double K, double r, double T, double sigma) {
        double d_1 = d1(S, K, r, T, sigma);
        double d_2 = d_1 - sigma * std::sqrt(T);
        return K * std::exp(-r * T) * N(-d_2) - S * N(-d_1);
    }

    /// Convenience overload using VanillaOption struct — Ch 3.5
    inline double call_price(const VanillaOption& opt) {
        return call_price(opt.S, opt.K, opt.r, opt.T, opt.sigma);
    }

    inline double put_price(const VanillaOption& opt) {
        return put_price(opt.S, opt.K, opt.r, opt.T, opt.sigma);
    }

    // ============================================================================
    // Analytic Greeks — Chapter 11.1
    // ============================================================================

    /// Delta: ∂C/∂S = N(d1) for calls
    inline double delta_call(double S, double K, double r, double T, double sigma) {
        return N(d1(S, K, r, T, sigma));
    }

    inline double delta_put(double S, double K, double r, double T, double sigma) {
        return delta_call(S, K, r, T, sigma) - 1.0;
    }

    /// Gamma: ∂²C/∂S² = n(d1) / (S * sigma * sqrt(T)) — same for calls and puts
    inline double gamma(double S, double K, double r, double T, double sigma) {
        return n(d1(S, K, r, T, sigma)) / (S * sigma * std::sqrt(T));
    }

    /// Vega: ∂C/∂σ = S * n(d1) * sqrt(T)
    inline double vega(double S, double K, double r, double T, double sigma) {
        return S * n(d1(S, K, r, T, sigma)) * std::sqrt(T);
    }

    /// Theta (call): ∂C/∂t — Ch 11.1
    inline double theta_call(double S, double K, double r, double T, double sigma) {
        double d_1 = d1(S, K, r, T, sigma);
        double d_2 = d_1 - sigma * std::sqrt(T);
        return -(S * n(d_1) * sigma) / (2.0 * std::sqrt(T))
            - r * K * std::exp(-r * T) * N(d_2);
    }

    inline double theta_put(double S, double K, double r, double T, double sigma) {
        double d_1 = d1(S, K, r, T, sigma);
        double d_2 = d_1 - sigma * std::sqrt(T);
        return -(S * n(d_1) * sigma) / (2.0 * std::sqrt(T))
            + r * K * std::exp(-r * T) * N(-d_2);
    }

    /// Rho (call): ∂C/∂r = K * T * exp(-rT) * N(d2)
    inline double rho_call(double S, double K, double r, double T, double sigma) {
        return K * T * std::exp(-r * T) * N(d2(S, K, r, T, sigma));
    }

    inline double rho_put(double S, double K, double r, double T, double sigma) {
        return -K * T * std::exp(-r * T) * N(-d2(S, K, r, T, sigma));
    }

} // namespace bs

#endif // QUANTPRICER_BLACK_SCHOLES_H

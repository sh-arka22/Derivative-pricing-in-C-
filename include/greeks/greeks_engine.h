#ifndef QUANTPRICER_GREEKS_ENGINE_H
#define QUANTPRICER_GREEKS_ENGINE_H

// ============================================================================
// Greeks Calculation Engine — Book Chapter 11
// Ch 11.1: Analytic formulae for Delta, Gamma, Vega, Theta, Rho
// Ch 11.2: Finite Difference Method (bump-and-reprice) for Greeks
// Ch 11.3: Monte Carlo Greeks (pathwise / likelihood ratio)
// ============================================================================

#include "greeks/black_scholes.h"
#include "mc/monte_carlo.h"
#include <functional>

/// Holds all five first-order Greeks + Gamma
struct GreeksResult {
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
};

// ============================================================================
// Method 1: Analytic Greeks — Ch 11.1
// ============================================================================

/// Full analytic Greeks for a European call
inline GreeksResult analytic_greeks_call(double S, double K, double r,
                                          double T, double sigma) {
    return {
        bs::delta_call(S, K, r, T, sigma),
        bs::gamma(S, K, r, T, sigma),
        bs::vega(S, K, r, T, sigma),
        bs::theta_call(S, K, r, T, sigma),
        bs::rho_call(S, K, r, T, sigma)
    };
}

/// Full analytic Greeks for a European put
inline GreeksResult analytic_greeks_put(double S, double K, double r,
                                         double T, double sigma) {
    return {
        bs::delta_put(S, K, r, T, sigma),
        bs::gamma(S, K, r, T, sigma),
        bs::vega(S, K, r, T, sigma),
        bs::theta_put(S, K, r, T, sigma),
        bs::rho_put(S, K, r, T, sigma)
    };
}

// ============================================================================
// Method 2: Finite Difference Greeks (Bump-and-Reprice) — Ch 11.2
// ============================================================================

/// Generic FD Greek calculator — works with ANY pricing function
/// Uses central differences: ∂f/∂x ≈ (f(x+h) - f(x-h)) / 2h
/// Second derivative: ∂²f/∂x² ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
class FDGreeks {
public:
    using PriceFn = std::function<double(double S, double K, double r,
                                          double T, double sigma)>;

    explicit FDGreeks(PriceFn pricer) : pricer_(std::move(pricer)) {}

    /// Delta via central difference: (V(S+h) - V(S-h)) / 2h — Ch 11.2
    double delta(double S, double K, double r, double T, double sigma,
                 double h_pct = 0.01) const {
        double h = S * h_pct;
        double v_up = pricer_(S + h, K, r, T, sigma);
        double v_dn = pricer_(S - h, K, r, T, sigma);
        return (v_up - v_dn) / (2.0 * h);
    }

    /// Gamma via central second difference — Ch 11.2
    double gamma(double S, double K, double r, double T, double sigma,
                 double h_pct = 0.01) const {
        double h = S * h_pct;
        double v_up = pricer_(S + h, K, r, T, sigma);
        double v_mid = pricer_(S, K, r, T, sigma);
        double v_dn = pricer_(S - h, K, r, T, sigma);
        return (v_up - 2.0 * v_mid + v_dn) / (h * h);
    }

    /// Vega via central difference on sigma
    double vega(double S, double K, double r, double T, double sigma,
                double h = 0.001) const {
        double v_up = pricer_(S, K, r, T, sigma + h);
        double v_dn = pricer_(S, K, r, T, sigma - h);
        return (v_up - v_dn) / (2.0 * h);
    }

    /// Theta via forward difference on T (time decay is negative direction)
    double theta(double S, double K, double r, double T, double sigma,
                 double h = 1.0 / 365.0) const {
        double v_now = pricer_(S, K, r, T, sigma);
        double v_later = pricer_(S, K, r, T - h, sigma);
        return (v_later - v_now) / h;  // Note: sign convention
    }

    /// Rho via central difference on r
    double rho(double S, double K, double r, double T, double sigma,
               double h = 0.0001) const {
        double v_up = pricer_(S, K, r + h, T, sigma);
        double v_dn = pricer_(S, K, r - h, T, sigma);
        return (v_up - v_dn) / (2.0 * h);
    }

    /// Compute all Greeks at once
    GreeksResult all(double S, double K, double r, double T, double sigma) const {
        return {
            delta(S, K, r, T, sigma),
            gamma(S, K, r, T, sigma),
            vega(S, K, r, T, sigma),
            theta(S, K, r, T, sigma),
            rho(S, K, r, T, sigma)
        };
    }

private:
    PriceFn pricer_;
};

// ============================================================================
// Method 3: Monte Carlo Greeks — Ch 11.3
// ============================================================================

/// MC Delta via pathwise method (IPA) for European options — Ch 11.3
/// ∂/∂S E[e^{-rT} h(S_T)] = e^{-rT} E[h'(S_T) * S_T/S_0]
/// For calls: h'(x) = 1{x > K}, so Delta = e^{-rT} * E[1{S_T > K} * S_T / S_0]
inline double mc_delta_pathwise(
    double S0, double K, double r, double T, double sigma,
    size_t num_paths, unsigned long seed = 42)
{
    MersenneTwisterRNG rng(seed);
    double discount = std::exp(-r * T);
    double drift = (r - 0.5 * sigma * sigma) * T;
    double vol_sqrt_T = sigma * std::sqrt(T);

    double sum = 0.0;
    auto normals = generate_normals(rng, num_paths);

    for (size_t i = 0; i < num_paths; ++i) {
        double S_T = S0 * std::exp(drift + vol_sqrt_T * normals[i]);
        if (S_T > K) {
            sum += S_T / S0;  // Pathwise derivative for call
        }
    }
    return discount * sum / num_paths;
}

/// MC Vega via pathwise method
/// ∂C/∂σ = e^{-rT} E[1{S_T>K} * S_T * (Z*√T - σT)]
/// where Z is the standard normal used to generate S_T
inline double mc_vega_pathwise(
    double S0, double K, double r, double T, double sigma,
    size_t num_paths, unsigned long seed = 42)
{
    MersenneTwisterRNG rng(seed);
    double discount = std::exp(-r * T);
    double drift = (r - 0.5 * sigma * sigma) * T;
    double vol_sqrt_T = sigma * std::sqrt(T);
    double sqrt_T = std::sqrt(T);

    double sum = 0.0;
    auto normals = generate_normals(rng, num_paths);

    for (size_t i = 0; i < num_paths; ++i) {
        double z = normals[i];
        double S_T = S0 * std::exp(drift + vol_sqrt_T * z);
        if (S_T > K) {
            // ∂S_T/∂σ = S_T * (z*√T - σT)
            sum += S_T * (z * sqrt_T - sigma * T);
        }
    }
    return discount * sum / num_paths;
}

#endif // QUANTPRICER_GREEKS_ENGINE_H

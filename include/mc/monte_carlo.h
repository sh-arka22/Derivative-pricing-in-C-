#ifndef QUANTPRICER_MONTE_CARLO_H
#define QUANTPRICER_MONTE_CARLO_H

// ============================================================================
// Monte Carlo Pricing Engine — Book Chapters 10, 12, 15, 16
// Ch 10:  European vanilla MC using risk-neutral GBM, antithetic variates
// Ch 12:  Path-dependent (Asian) options — arithmetic/geometric averaging,
//         path generation classes, OOP design
// Ch 15:  Jump-diffusion MC (Merton model) with Poisson jumps
// Ch 16:  Heston stochastic volatility MC with Euler discretisation,
//         correlated Brownian motions via Cholesky decomposition
// Also uses: Ch 6 (STL containers/algorithms), Ch 7 (function objects),
//            Ch 14 (RNG hierarchy)
// ============================================================================

#include "payoff/payoff.h"
#include "option/option.h"
#include "rng/rng.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>

/// Result struct holding price + standard error
struct MCResult {
    double price;
    double std_error;
    double confidence_lo;    // 95% CI lower
    double confidence_hi;    // 95% CI upper
    size_t num_paths;
};

// ============================================================================
// European Option MC — Chapter 10.3, 10.4
// ============================================================================

/// Price European option via Monte Carlo under GBM — Ch 10.4
/// S(T) = S(0) * exp((r - σ²/2)T + σ√T * Z), Z ~ N(0,1)
/// With antithetic variates for variance reduction
inline MCResult mc_european(
    const PayOff& payoff,
    double S0, double r, double T, double sigma,
    size_t num_paths, unsigned long seed = 42)
{
    MersenneTwisterRNG rng(seed);
    double discount = std::exp(-r * T);
    double drift = (r - 0.5 * sigma * sigma) * T;
    double vol_sqrt_T = sigma * std::sqrt(T);

    double sum = 0.0, sum_sq = 0.0;

    // Use n_pairs consistently for both loop bound and denominator (Glasserman Ch 4.2)
    size_t n_pairs = std::max<size_t>(num_paths / 2, 1);

    for (size_t i = 0; i < n_pairs; ++i) {
        auto normals = generate_normals(rng, 1);
        double z = normals[0];

        // Regular path
        double S_T = S0 * std::exp(drift + vol_sqrt_T * z);
        double pv1 = discount * payoff(S_T);

        // Antithetic path — variance reduction (Ch 10 extension)
        double S_T_anti = S0 * std::exp(drift - vol_sqrt_T * z);
        double pv2 = discount * payoff(S_T_anti);

        double avg = 0.5 * (pv1 + pv2);
        sum += avg;
        sum_sq += avg * avg;
    }

    double mean = sum / n_pairs;
    double var = (sum_sq / n_pairs) - mean * mean;
    double se = std::sqrt(std::max(var, 0.0) / n_pairs);

    return {mean, se, mean - 1.96 * se, mean + 1.96 * se, 2 * n_pairs};
}

// ============================================================================
// Path Generation — Chapter 12.4
// ============================================================================

/// Generate a single GBM path with n_steps — Ch 12.4
inline std::vector<double> generate_gbm_path(
    double S0, double r, double sigma, double T,
    size_t n_steps, RandomNumberGenerator& rng)
{
    double dt = T / n_steps;
    double drift = (r - 0.5 * sigma * sigma) * dt;
    double vol_sqrt_dt = sigma * std::sqrt(dt);

    std::vector<double> path(n_steps + 1);
    path[0] = S0;

    auto normals = generate_normals(rng, n_steps);
    for (size_t i = 0; i < n_steps; ++i) {
        path[i + 1] = path[i] * std::exp(drift + vol_sqrt_dt * normals[i]);
    }
    return path;
}

// ============================================================================
// Asian Option MC — Chapter 12.5, 12.6
// ============================================================================

/// Arithmetic Asian Call/Put via Monte Carlo — Ch 12
/// Payoff depends on arithmetic average: max(A - K, 0) or max(K - A, 0)
/// where A = (1/n) * Σ S(t_i)
inline MCResult mc_asian_arithmetic(
    const PayOff& payoff,
    double S0, double r, double T, double sigma,
    size_t n_steps, size_t num_paths, unsigned long seed = 42)
{
    MersenneTwisterRNG rng(seed);
    double discount = std::exp(-r * T);

    double sum = 0.0, sum_sq = 0.0;

    for (size_t p = 0; p < num_paths; ++p) {
        auto path = generate_gbm_path(S0, r, sigma, T, n_steps, rng);

        // Arithmetic average (exclude S(0)) — Ch 12.5
        double avg = 0.0;
        for (size_t i = 1; i <= n_steps; ++i)
            avg += path[i];
        avg /= n_steps;

        double pv = discount * payoff(avg);
        sum += pv;
        sum_sq += pv * pv;
    }

    double mean = sum / num_paths;
    double var = (sum_sq / num_paths) - mean * mean;
    double se = std::sqrt(var / num_paths);

    return {mean, se, mean - 1.96 * se, mean + 1.96 * se, num_paths};
}

/// Geometric Asian (closed-form benchmark) — Ch 12 extension
/// Used as control variate for arithmetic Asian
inline MCResult mc_asian_geometric(
    const PayOff& payoff,
    double S0, double r, double T, double sigma,
    size_t n_steps, size_t num_paths, unsigned long seed = 42)
{
    MersenneTwisterRNG rng(seed);
    double discount = std::exp(-r * T);

    double sum = 0.0, sum_sq = 0.0;

    for (size_t p = 0; p < num_paths; ++p) {
        auto path = generate_gbm_path(S0, r, sigma, T, n_steps, rng);

        // Geometric average — exp(mean of log prices)
        double log_sum = 0.0;
        for (size_t i = 1; i <= n_steps; ++i)
            log_sum += std::log(path[i]);
        double geo_avg = std::exp(log_sum / n_steps);

        double pv = discount * payoff(geo_avg);
        sum += pv;
        sum_sq += pv * pv;
    }

    double mean = sum / num_paths;
    double var = (sum_sq / num_paths) - mean * mean;
    double se = std::sqrt(var / num_paths);

    return {mean, se, mean - 1.96 * se, mean + 1.96 * se, num_paths};
}

// ============================================================================
// Jump-Diffusion MC (Merton Model) — Chapter 15
// ============================================================================

/// Merton jump-diffusion Monte Carlo — Ch 15.2, 15.3
/// dS/S = (r - λk)dt + σ dW + J dN
/// where N is Poisson process with intensity λ, J ~ LN(μ_j, σ_j²)
inline MCResult mc_merton_jump(
    const PayOff& payoff,
    const MertonJumpParams& params,
    size_t num_paths, size_t n_steps = 252,
    unsigned long seed = 42)
{
    MersenneTwisterRNG rng(seed);
    double dt = params.T / n_steps;
    double discount = std::exp(-params.r * params.T);

    // Jump compensator: k = E[e^J - 1] = exp(μ_j + σ_j²/2) - 1
    double k = std::exp(params.mu_j + 0.5 * params.sigma_j * params.sigma_j) - 1.0;
    double drift = (params.r - params.lambda * k - 0.5 * params.sigma * params.sigma) * dt;
    double vol_sqrt_dt = params.sigma * std::sqrt(dt);

    // Poisson distribution for jump counts
    std::mt19937_64 poisson_engine(seed + 1);
    std::poisson_distribution<int> poisson(params.lambda * dt);
    std::normal_distribution<double> jump_normal(params.mu_j, params.sigma_j);

    double sum = 0.0, sum_sq = 0.0;

    for (size_t p = 0; p < num_paths; ++p) {
        double S = params.S0;
        auto normals = generate_normals(rng, n_steps);

        for (size_t i = 0; i < n_steps; ++i) {
            // Diffusion component
            double dW = vol_sqrt_dt * normals[i];

            // Jump component — Ch 15.1
            int n_jumps = poisson(poisson_engine);
            double jump_sum = 0.0;
            for (int j = 0; j < n_jumps; ++j)
                jump_sum += jump_normal(poisson_engine);

            S *= std::exp(drift + dW + jump_sum);
        }

        double pv = discount * payoff(S);
        sum += pv;
        sum_sq += pv * pv;
    }

    double mean = sum / num_paths;
    double var = (sum_sq / num_paths) - mean * mean;
    double se = std::sqrt(var / num_paths);

    return {mean, se, mean - 1.96 * se, mean + 1.96 * se, num_paths};
}

// ============================================================================
// Heston Stochastic Volatility MC — Chapter 16
//
// Heston SDE system (two coupled processes under risk-neutral measure):
//   dS = r*S*dt + sqrt(v)*S*dW_S          (stock price, GBM-like with stoch vol)
//   dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW_v   (variance, CIR-type mean-reverting)
//   Corr(dW_S, dW_v) = rho               (leverage effect: rho < 0 gives skew)
//
// Parameters:
//   S0    — initial spot price
//   v0    — initial variance (NOT volatility; sigma = sqrt(v))
//   kappa — mean reversion speed of variance
//   theta — long-run variance level (v mean-reverts to theta)
//   xi    — vol-of-vol (volatility of the variance process)
//   rho   — correlation between asset and variance Brownians
//
// Feller condition: 2*kappa*theta > xi^2  ensures v stays strictly positive
//
// Key interview insight:
//   When xi=0, variance is deterministic -> Heston collapses to Black-Scholes.
//   If xi=0 and v0=theta, then v(t)=theta forever -> sqrt(v)=sigma constant
//   -> mc_heston == mc_european
// ============================================================================

/// Heston model Monte Carlo with Euler discretisation — Ch 16.6, 16.7
inline MCResult mc_heston(
    const PayOff& payoff,
    const HestonParams& params,
    size_t num_paths, size_t n_steps = 252,
    unsigned long seed = 42)
{
    MersenneTwisterRNG rng(seed);

    // dt, sqrt(dt) — used every time step
    double dt = params.T / n_steps;
    double sqrt_dt = std::sqrt(dt);
    double discount = std::exp(-params.r * params.T);

    double sum = 0.0, sum_sq = 0.0;

    for (size_t p = 0; p < num_paths; ++p) {
        // Initialise both state variables
        double S = params.S0;   // stock starts at spot
        double v = params.v0;   // variance starts at v0

        for (size_t i = 0; i < n_steps; ++i) {
            // Correlated Brownian increments via Cholesky — Ch 16.3, 16.4
            // z1 ~ N(0,1) independent
            // z2 = rho*z1 + sqrt(1-rho^2)*eps2    where eps2 ~ N(0,1) independent
            // This gives Corr(z1, z2) = rho  (verified by construction)
            auto [z1, z2] = generate_correlated_normals(rng, params.rho);

            // FULL TRUNCATION scheme (Lord, Koekkoek & Van Dijk, 2010)
            // v+ = max(v, 0)  — prevents sqrt(negative)
            // All drift/diffusion coefficients use v+ but raw v is carried forward
            double v_pos = std::max(v, 0.0);
            double sqrt_v = std::sqrt(v_pos);

            // Stock update (log-Euler discretisation) — Ch 16.6
            // Math: S_{i+1} = S_i * exp((r - v+/2)*dt + sqrt(v+)*sqrt(dt)*Z_1)
            // Log-Euler preserves S > 0 by construction (unlike naive Euler)
            //   dS: drift = (r - v/2)*dt,  diffusion = sqrt(v)*sqrt(dt)*Z_1
            S *= std::exp((params.r - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * z1);

            // Variance update (standard Euler on CIR process) — Ch 16.6
            // Math: v_{i+1} = v_i + kappa*(theta - v+)*dt + xi*sqrt(v+)*sqrt(dt)*Z_2
            //   dv: drift = kappa*(theta - v+)*dt,  diffusion = xi*sqrt(v+)*sqrt(dt)*Z_2
            v += params.kappa * (params.theta - v_pos) * dt
                 + params.xi * sqrt_v * sqrt_dt * z2;
        }

        // Terminal payoff per path — same as GBM MC
        double pv = discount * payoff(S);
        sum += pv;
        sum_sq += pv * pv;
    }

    // Risk-neutral price = exp(-rT) * E[payoff(S_T)]
    double mean = sum / num_paths;
    double var = (sum_sq / num_paths) - mean * mean;
    double se = std::sqrt(var / num_paths);

    return {mean, se, mean - 1.96 * se, mean + 1.96 * se, num_paths};
}
// Demo result: rho=-0.7 -> Heston price ~ 10.41 vs BS = 10.45 (ATM close)
// but implied vol smile: ~23.2% at K=80 to ~17.3% at K=120

// ============================================================================
// Bates Model MC — Heston + Merton Jumps (Bates, 1996)
//
// The Bates model augments Heston's stochastic volatility with log-normal
// jumps in the spot process, giving it the ability to capture:
//   1. Stochastic vol  → smile curvature (wings controlled by xi, rho)
//   2. Jump risk       → short-maturity skew and excess kurtosis
//
// SDE system (risk-neutral):
//   dS = (r - λk)*S*dt + sqrt(v)*S*dW_S + S*(e^J - 1)*dN
//   dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW_v
//   Corr(dW_S, dW_v) = rho
//
// where:
//   N(t) = Poisson process with intensity λ
//   J ~ N(μ_j, σ_j²) — log-jump size
//   k = E[e^J - 1] = exp(μ_j + σ_j²/2) - 1  — jump compensator
//
// Key interview insight:
//   λ=0 → Bates collapses to Heston (no jumps)
//   xi=0,λ=0 → collapses to Black-Scholes
//   xi=0,λ>0 → collapses to Merton jump-diffusion
//   This nesting makes Bates the "union" of the two most important
//   extensions to Black-Scholes, and a popular calibration target.
// ============================================================================

/// Bates model Monte Carlo with Euler discretisation
inline MCResult mc_bates(
    const PayOff& payoff,
    const BatesParams& params,
    size_t num_paths, size_t n_steps = 252,
    unsigned long seed = 42)
{
    MersenneTwisterRNG rng(seed);

    double dt = params.T / n_steps;
    double sqrt_dt = std::sqrt(dt);
    double discount = std::exp(-params.r * params.T);

    // Jump compensator: k = E[e^J - 1] = exp(μ_j + σ_j²/2) - 1
    double k = std::exp(params.mu_j + 0.5 * params.sigma_j * params.sigma_j) - 1.0;

    // Poisson and jump-size distributions
    std::mt19937_64 jump_engine(seed + 1);
    std::poisson_distribution<int> poisson(params.lambda * dt);
    std::normal_distribution<double> jump_normal(params.mu_j, params.sigma_j);

    double sum = 0.0, sum_sq = 0.0;

    for (size_t p = 0; p < num_paths; ++p) {
        double S = params.S0;
        double v = params.v0;

        for (size_t i = 0; i < n_steps; ++i) {
            // Correlated Brownian increments for asset & variance
            auto [z1, z2] = generate_correlated_normals(rng, params.rho);

            // Full truncation for variance
            double v_pos = std::max(v, 0.0);
            double sqrt_v = std::sqrt(v_pos);

            // Jump component (identical to Merton)
            int n_jumps = poisson(jump_engine);
            double jump_sum = 0.0;
            for (int j = 0; j < n_jumps; ++j)
                jump_sum += jump_normal(jump_engine);

            // Stock: log-Euler with jump-compensated drift + jumps
            S *= std::exp((params.r - params.lambda * k - 0.5 * v_pos) * dt
                          + sqrt_v * sqrt_dt * z1
                          + jump_sum);

            // Variance: standard Euler on CIR process (same as Heston)
            v += params.kappa * (params.theta - v_pos) * dt
                 + params.xi * sqrt_v * sqrt_dt * z2;
        }

        double pv = discount * payoff(S);
        sum += pv;
        sum_sq += pv * pv;
    }

    double mean = sum / num_paths;
    double var = (sum_sq / num_paths) - mean * mean;
    double se = std::sqrt(var / num_paths);

    return {mean, se, mean - 1.96 * se, mean + 1.96 * se, num_paths};
}

#endif // QUANTPRICER_MONTE_CARLO_H

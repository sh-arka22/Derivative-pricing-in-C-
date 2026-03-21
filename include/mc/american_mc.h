#ifndef QUANTPRICER_AMERICAN_MC_H
#define QUANTPRICER_AMERICAN_MC_H

// ============================================================================
// Longstaff-Schwartz Least-Squares Monte Carlo (LSM) — Day 16
//
// Reference: Longstaff & Schwartz (2001), "Valuing American Options by
//            Simulation: A Simple Least-Squares Approach"
//
// Algorithm:
// 1. Generate N paths of the underlying under risk-neutral measure
// 2. At maturity: cashflow = payoff(S_T)
// 3. Working backwards from T-1 to 1:
//    a. Identify in-the-money paths (where early exercise has value)
//    b. Regress discounted future cashflows on polynomial basis of S
//       (Laguerre polynomials or simple monomials: 1, S, S^2, ...)
//    c. Continuation value = fitted regression value
//    d. If intrinsic > continuation: exercise now, update cashflow
// 4. Price = mean of discounted cashflows from optimal exercise
//
// The regression uses ordinary least squares solved via normal equations:
//   beta = (X'X)^{-1} X'Y
// ============================================================================

#include "payoff/payoff.h"
#include "option/option.h"
#include "rng/rng.h"
#include "mc/monte_carlo.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

/// Result from Longstaff-Schwartz pricing
struct LSMResult {
    double price;
    double std_error;
    double confidence_lo;
    double confidence_hi;
    size_t num_paths;
    double early_exercise_premium;  // vs European MC
};

// ============================================================================
// Internal: Least-squares regression via normal equations
// ============================================================================

namespace lsm_detail {

/// Solve least squares: Y = X * beta, returns beta
/// X is (n x p), Y is (n x 1)
/// Uses normal equations: beta = (X'X)^{-1} X'Y
/// For small p (3-5 basis functions), this is efficient and stable enough.
inline std::vector<double> least_squares(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& Y)
{
    size_t n = X.size();
    if (n == 0) return {};
    size_t p = X[0].size();

    // Compute X'X (p x p)
    std::vector<std::vector<double>> XtX(p, std::vector<double>(p, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < p; ++j) {
            for (size_t k = 0; k < p; ++k) {
                XtX[j][k] += X[i][j] * X[i][k];
            }
        }
    }

    // Compute X'Y (p x 1)
    std::vector<double> XtY(p, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < p; ++j) {
            XtY[j] += X[i][j] * Y[i];
        }
    }

    // Solve XtX * beta = XtY via Gaussian elimination with partial pivoting
    // (small system, p ~ 3-5, so this is fine)
    std::vector<double> beta(p, 0.0);

    // Augmented matrix [XtX | XtY]
    std::vector<std::vector<double>> aug(p, std::vector<double>(p + 1));
    for (size_t i = 0; i < p; ++i) {
        for (size_t j = 0; j < p; ++j) {
            aug[i][j] = XtX[i][j];
        }
        aug[i][p] = XtY[i];
    }

    // Forward elimination with partial pivoting
    for (size_t col = 0; col < p; ++col) {
        // Find pivot
        size_t max_row = col;
        double max_val = std::abs(aug[col][col]);
        for (size_t row = col + 1; row < p; ++row) {
            if (std::abs(aug[row][col]) > max_val) {
                max_val = std::abs(aug[row][col]);
                max_row = row;
            }
        }
        if (max_val < 1e-14) continue;  // Singular or near-singular

        std::swap(aug[col], aug[max_row]);

        // Eliminate below
        for (size_t row = col + 1; row < p; ++row) {
            double factor = aug[row][col] / aug[col][col];
            for (size_t j = col; j <= p; ++j) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    for (size_t i = p; i-- > 0; ) {
        beta[i] = aug[i][p];
        for (size_t j = i + 1; j < p; ++j) {
            beta[i] -= aug[i][j] * beta[j];
        }
        if (std::abs(aug[i][i]) > 1e-14) {
            beta[i] /= aug[i][i];
        } else {
            beta[i] = 0.0;
        }
    }

    return beta;
}

/// Evaluate polynomial basis at x: {1, x, x^2, ...}
/// Using simple monomials (weighted Laguerre polynomials are an alternative)
inline std::vector<double> polynomial_basis(double x, size_t degree) {
    std::vector<double> basis(degree + 1);
    basis[0] = 1.0;
    for (size_t i = 1; i <= degree; ++i) {
        basis[i] = basis[i - 1] * x;
    }
    return basis;
}

} // namespace lsm_detail

// ============================================================================
// Longstaff-Schwartz Monte Carlo
// ============================================================================

/// Longstaff-Schwartz least-squares Monte Carlo for American options
///
/// Parameters:
///   payoff    - PayOff functor (call, put, etc.)
///   S0        - Initial spot price
///   r         - Risk-free rate
///   T         - Time to maturity
///   sigma     - Volatility
///   num_paths - Number of MC paths (should be even for antithetic)
///   n_steps   - Number of time steps
///   poly_deg  - Degree of polynomial basis for regression (default 3)
///   seed      - RNG seed
///
/// Returns LSMResult with price, SE, confidence interval
inline LSMResult mc_american_lsm(
    const PayOff& payoff,
    double S0, double r, double T, double sigma,
    size_t num_paths, size_t n_steps = 50,
    size_t poly_deg = 3, unsigned long seed = 42)
{
    MersenneTwisterRNG rng(seed);
    double dt = T / n_steps;
    double discount = std::exp(-r * dt);

    // Step 1: Generate all paths
    // paths[i][j] = spot price of path i at time step j
    std::vector<std::vector<double>> paths(num_paths);
    for (size_t i = 0; i < num_paths; ++i) {
        paths[i] = generate_gbm_path(S0, r, sigma, T, n_steps, rng);
    }

    // Step 2: Initialize cashflows at maturity
    // cashflow[i] = discounted cashflow for path i (from optimal exercise)
    // exercise_time[i] = time step at which path i is exercised
    std::vector<double> cashflow(num_paths);
    std::vector<size_t> exercise_time(num_paths, n_steps);

    for (size_t i = 0; i < num_paths; ++i) {
        cashflow[i] = payoff(paths[i][n_steps]);
    }

    // Step 3: Backward induction from T-1 to 1
    for (size_t step = n_steps - 1; step >= 1; --step) {
        // Identify in-the-money paths
        std::vector<size_t> itm_indices;
        for (size_t i = 0; i < num_paths; ++i) {
            if (payoff(paths[i][step]) > 0.0) {
                itm_indices.push_back(i);
            }
        }

        if (itm_indices.size() < poly_deg + 1) continue;  // Not enough data for regression

        // Build regression: Y = discounted future cashflows, X = basis(S)
        std::vector<std::vector<double>> X(itm_indices.size());
        std::vector<double> Y(itm_indices.size());

        for (size_t k = 0; k < itm_indices.size(); ++k) {
            size_t idx = itm_indices[k];
            double S = paths[idx][step];

            // Normalize S by S0 for numerical stability of polynomial regression
            X[k] = lsm_detail::polynomial_basis(S / S0, poly_deg);

            // Discounted future cashflow
            size_t steps_to_exercise = exercise_time[idx] - step;
            Y[k] = cashflow[idx] * std::pow(discount, static_cast<double>(steps_to_exercise));
        }

        // Regression: estimate continuation value
        auto beta = lsm_detail::least_squares(X, Y);

        // Decision: exercise if intrinsic > continuation
        for (size_t k = 0; k < itm_indices.size(); ++k) {
            size_t idx = itm_indices[k];
            double S = paths[idx][step];
            double intrinsic = payoff(S);

            // Continuation value from regression
            auto basis = lsm_detail::polynomial_basis(S / S0, poly_deg);
            double continuation = 0.0;
            for (size_t j = 0; j < beta.size(); ++j) {
                continuation += beta[j] * basis[j];
            }

            if (intrinsic > continuation) {
                cashflow[idx] = intrinsic;
                exercise_time[idx] = step;
            }
        }
    }

    // Step 4: Compute price as mean of discounted cashflows
    double sum = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < num_paths; ++i) {
        double pv = cashflow[i] * std::pow(discount, static_cast<double>(exercise_time[i]));
        sum += pv;
        sum_sq += pv * pv;
    }

    double mean = sum / num_paths;
    double var = (sum_sq / num_paths) - mean * mean;
    double se = std::sqrt(var / num_paths);

    // Compute European price for comparison (early exercise premium)
    double eur_sum = 0.0;
    double full_discount = std::exp(-r * T);
    for (size_t i = 0; i < num_paths; ++i) {
        eur_sum += full_discount * payoff(paths[i][n_steps]);
    }
    double eur_price = eur_sum / num_paths;
    double eep = std::max(mean - eur_price, 0.0);

    return {mean, se, mean - 1.96 * se, mean + 1.96 * se, num_paths, eep};
}

#endif // QUANTPRICER_AMERICAN_MC_H

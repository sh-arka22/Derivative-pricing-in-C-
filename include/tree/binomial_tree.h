#ifndef QUANTPRICER_BINOMIAL_TREE_H
#define QUANTPRICER_BINOMIAL_TREE_H

// ============================================================================
// Binomial Tree Pricing — Day 16: American Options & Lattice Methods
//
// Cox-Ross-Rubinstein (CRR) parameterization:
//   u = exp(sigma * sqrt(dt))
//   d = 1/u
//   p = (exp(r * dt) - d) / (u - d)
//
// European: V(i,j) = exp(-r*dt) * [p * V(i+1,j+1) + (1-p) * V(i+1,j)]
// American: V(i,j) = max(intrinsic, continuation)
//
// Key optimization: O(N) space using single-array backward induction
// instead of O(N^2) full tree storage.
// ============================================================================

#include "payoff/payoff.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

/// Result from binomial tree pricing
struct TreeResult {
    double price;
    double delta;          // First-order finite difference at root
    double gamma;          // Second-order finite difference at root
    double theta;          // Time decay from first two time steps
    size_t n_steps;
};

/// CRR Binomial Tree pricer — supports European and American exercise
///
/// Mathematics:
///   At terminal nodes (step N): V(N, j) = payoff(S0 * u^j * d^(N-j))
///   Backward induction: V(i, j) = df * [p * V(i+1, j+1) + (1-p) * V(i+1, j)]
///   American early exercise: V(i, j) = max(V_cont, payoff(S(i,j)))
///
/// Implementation uses O(N) space: a single vector of N+1 values,
/// updated in-place from terminal to root.
inline TreeResult binomial_tree(
    const PayOff& payoff,
    double S0, double /*K*/, double r, double T, double sigma,
    size_t n_steps, bool american = false)
{
    if (n_steps == 0) throw std::invalid_argument("n_steps must be > 0");
    if (T <= 0.0) throw std::invalid_argument("T must be positive");
    if (sigma <= 0.0) throw std::invalid_argument("sigma must be positive");

    double dt = T / n_steps;
    double u = std::exp(sigma * std::sqrt(dt));     // Up factor
    double d = 1.0 / u;                              // Down factor (recombining)
    double df = std::exp(-r * dt);                    // Discount factor per step
    double p = (std::exp(r * dt) - d) / (u - d);     // Risk-neutral up probability
    double q = 1.0 - p;                               // Risk-neutral down probability

    // Precompute asset prices at terminal nodes: S(N,j) = S0 * u^(2j - N)
    // Using the fact that S(N,j) = S0 * u^j * d^(N-j) = S0 * u^(2j-N)
    size_t N = n_steps;
    std::vector<double> V(N + 1);

    // Terminal payoffs
    for (size_t j = 0; j <= N; ++j) {
        double S_T = S0 * std::pow(u, static_cast<double>(2 * j) - static_cast<double>(N));
        V[j] = payoff(S_T);
    }

    // Store values at step 1 and 2 for Greeks computation
    // We need V at steps 0, 1, 2 to compute delta, gamma, theta
    double V10 = 0.0, V11 = 0.0;           // Step 1: node 0, node 1
    double V20 = 0.0, V21 = 0.0, V22 = 0.0; // Step 2: nodes 0, 1, 2

    // Backward induction: O(N) space
    for (size_t i = N; i-- > 0; ) {
        for (size_t j = 0; j <= i; ++j) {
            // Continuation value
            double continuation = df * (p * V[j + 1] + q * V[j]);

            if (american) {
                // Asset price at node (i, j)
                double S_ij = S0 * std::pow(u, static_cast<double>(2 * j) - static_cast<double>(i));
                double intrinsic = payoff(S_ij);
                V[j] = std::max(continuation, intrinsic);
            } else {
                V[j] = continuation;
            }
        }

        // Capture values for Greeks
        if (i == 2 && N >= 2) {
            V20 = V[0]; V21 = V[1]; V22 = V[2];
        }
        if (i == 1 && N >= 1) {
            V10 = V[0]; V11 = V[1];
        }
    }

    // Greeks via finite differences on the tree
    double price = V[0];
    double delta = 0.0, gamma = 0.0, theta = 0.0;

    if (N >= 1) {
        double S_up = S0 * u;
        double S_down = S0 * d;
        delta = (V11 - V10) / (S_up - S_down);
    }

    if (N >= 2) {
        double S_uu = S0 * u * u;
        double S_dd = S0 * d * d;
        double delta_up = (V22 - V21) / (S_uu - S0);
        double delta_down = (V21 - V20) / (S0 - S_dd);
        gamma = (delta_up - delta_down) / (0.5 * (S_uu - S_dd));
        theta = (V21 - price) / (2.0 * dt);
    }

    return {price, delta, gamma, theta, n_steps};
}

/// Convenience: price European option via binomial tree
inline TreeResult binomial_european(
    const PayOff& payoff,
    double S0, double K, double r, double T, double sigma,
    size_t n_steps)
{
    return binomial_tree(payoff, S0, K, r, T, sigma, n_steps, false);
}

/// Convenience: price American option via binomial tree
inline TreeResult binomial_american(
    const PayOff& payoff,
    double S0, double K, double r, double T, double sigma,
    size_t n_steps)
{
    return binomial_tree(payoff, S0, K, r, T, sigma, n_steps, true);
}

/// Early exercise premium: American price - European price
/// This should be >= 0 (American option is always worth at least as much)
inline double early_exercise_premium(
    const PayOff& payoff,
    double S0, double K, double r, double T, double sigma,
    size_t n_steps)
{
    auto eur = binomial_european(payoff, S0, K, r, T, sigma, n_steps);
    auto ame = binomial_american(payoff, S0, K, r, T, sigma, n_steps);
    return ame.price - eur.price;
}

#endif // QUANTPRICER_BINOMIAL_TREE_H

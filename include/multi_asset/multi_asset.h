#ifndef QUANTPRICER_MULTI_ASSET_H
#define QUANTPRICER_MULTI_ASSET_H

// ============================================================================
// Multi-Asset & Basket Options — Day 11
//
// ============================================================================
// MATHEMATICS — Correlated Multi-Dimensional GBM
// ============================================================================
//
// N assets each follow GBM under risk-neutral measure:
//
//   dS_i / S_i = r * dt + sigma_i * dW_i,    i = 1, ..., N
//
// where the Brownian motions are correlated:
//
//   Corr(dW_i, dW_j) = rho_ij      (instantaneous correlation)
//   E[dW_i * dW_j] = rho_ij * dt
//
// The correlation matrix R = [rho_ij] must be:
//   - Symmetric: rho_ij = rho_ji
//   - Positive semi-definite: x' R x >= 0 for all x
//   - Unit diagonal: rho_ii = 1
//
// To simulate correlated paths, we use the CHOLESKY DECOMPOSITION:
//
//   R = L * L'    where L is lower triangular
//
//   Given N independent standard normals Z = (Z_1, ..., Z_N)',
//   the correlated normals are: W = L * Z
//   Then: Corr(W) = L * I * L' = L * L' = R    ✓
//
// Each asset's terminal value (log-Euler, one step):
//
//   S_i(T) = S_i(0) * exp((r - sigma_i^2/2)*T + sigma_i*sqrt(T)*W_i)
//
// For path-dependent multi-asset options, we step through time:
//
//   S_i(t+dt) = S_i(t) * exp((r - sigma_i^2/2)*dt + sigma_i*sqrt(dt)*W_i(t))
//
// ============================================================================
// PAYOFF TYPES
// ============================================================================
//
// Basket Call/Put:     payoff on weighted average
//   Basket = sum(w_i * S_i(T))
//   Call payoff = max(Basket - K, 0)
//   Put payoff  = max(K - Basket, 0)
//
// Best-of Call:        payoff on the maximum
//   payoff = max(max_i(S_i(T)) - K, 0)
//
// Worst-of Put:        payoff on the minimum
//   payoff = max(K - min_i(S_i(T)), 0)
//
// Rainbow (Best-of):   max(S_1(T), S_2(T), ..., S_N(T))
//   No strike — pure rainbow option
//
// Spread Option:       payoff on difference (2 assets)
//   payoff = max(S_1(T) - S_2(T) - K, 0)
//
// Exchange Option:     special case of spread with K=0 (Margrabe's formula)
//   payoff = max(S_1(T) - S_2(T), 0)
//
// ============================================================================
// KEY INTERVIEW INSIGHTS
// ============================================================================
//
// 1. CORRELATION AND BASKET PRICE:
//    Higher correlation → basket vol is higher → basket call is MORE expensive.
//    Intuition: if assets move together, the average behaves like a single
//    volatile asset. If uncorrelated, averaging smooths out → lower vol.
//    Basket vol ≈ sqrt(w' * Sigma * w) where Sigma_ij = sigma_i*sigma_j*rho_ij.
//
// 2. CORRELATION AND BEST-OF/WORST-OF:
//    Higher correlation → best-of call is CHEAPER (assets move together,
//    so max(S_i) ≈ any S_i). Lower correlation → more dispersion → higher max.
//    Opposite effect from basket! This is the "correlation trade".
//
// 3. MARGRABE'S FORMULA (1978):
//    Exchange option V = S_1*N(d1) - S_2*N(d2) where
//    sigma_spread = sqrt(sigma_1^2 + sigma_2^2 - 2*rho*sigma_1*sigma_2)
//    This is Black-Scholes with S_2 as numeraire. No discounting needed!
//    Key insight: the exchange option price does NOT depend on r.
//
// 4. DIMENSION CURSE:
//    MC scales as O(N*M) where N=assets, M=paths — linear in dimension.
//    PDE/FDM scales as O(grid^N) — exponential! This is why MC dominates
//    for multi-asset pricing. "MC is the only game in town for N > 3."
// ============================================================================

#include "greeks/black_scholes.h"
#include "matrix/matrix.h"
#include "rng/rng.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <stdexcept>

namespace multi_asset {

// ============================================================================
// Multi-Asset Parameters
// ============================================================================

struct MultiAssetParams {
    std::vector<double> S0;         // Initial spot prices (N assets)
    std::vector<double> sigma;      // Volatilities (N assets)
    std::vector<double> weights;    // Portfolio weights (for basket)
    QMatrix<double> corr;           // Correlation matrix (N x N)
    double r;                       // Risk-free rate
    double T;                       // Time to maturity
    double K;                       // Strike price

    size_t n_assets() const { return S0.size(); }

    /// Validate that all dimensions are consistent and correlation is valid
    void validate() const {
        size_t N = n_assets();
        if (sigma.size() != N)
            throw std::invalid_argument("sigma size mismatch");
        if (!weights.empty() && weights.size() != N)
            throw std::invalid_argument("weights size mismatch");
        if (corr.rows() != N || corr.cols() != N)
            throw std::invalid_argument("correlation matrix size mismatch");
        // Check diagonal = 1
        for (size_t i = 0; i < N; ++i) {
            if (std::abs(corr(i, i) - 1.0) > 1e-10)
                throw std::invalid_argument("correlation diagonal must be 1");
        }
    }
};

/// Result struct for multi-asset MC pricing
struct MultiAssetMCResult {
    double price;
    double std_error;
    double confidence_lo;
    double confidence_hi;
    size_t num_paths;
};

// ============================================================================
// Multi-Asset Payoff Types
// ============================================================================

enum class PayoffType {
    BasketCall,     // max(sum(w_i * S_i) - K, 0)
    BasketPut,      // max(K - sum(w_i * S_i), 0)
    BestOfCall,     // max(max(S_i) - K, 0)
    WorstOfCall,    // max(min(S_i) - K, 0)
    BestOfPut,      // max(K - max(S_i), 0)
    WorstOfPut,     // max(K - min(S_i), 0)
    Rainbow,        // max(S_1, S_2, ..., S_N) — no strike
    SpreadCall,     // max(S_1 - S_2 - K, 0) — 2 assets only
    Exchange        // max(S_1 - S_2, 0) — Margrabe, 2 assets only
};

/// Evaluate multi-asset payoff given terminal spot prices
inline double evaluate_payoff(
    PayoffType type,
    const std::vector<double>& S_T,
    const std::vector<double>& weights,
    double K)
{
    switch (type) {
        case PayoffType::BasketCall: {
            double basket = 0.0;
            for (size_t i = 0; i < S_T.size(); ++i)
                basket += weights[i] * S_T[i];
            return std::max(basket - K, 0.0);
        }
        case PayoffType::BasketPut: {
            double basket = 0.0;
            for (size_t i = 0; i < S_T.size(); ++i)
                basket += weights[i] * S_T[i];
            return std::max(K - basket, 0.0);
        }
        case PayoffType::BestOfCall:
            return std::max(*std::max_element(S_T.begin(), S_T.end()) - K, 0.0);
        case PayoffType::WorstOfCall:
            return std::max(*std::min_element(S_T.begin(), S_T.end()) - K, 0.0);
        case PayoffType::BestOfPut:
            return std::max(K - *std::max_element(S_T.begin(), S_T.end()), 0.0);
        case PayoffType::WorstOfPut:
            return std::max(K - *std::min_element(S_T.begin(), S_T.end()), 0.0);
        case PayoffType::Rainbow:
            return *std::max_element(S_T.begin(), S_T.end());
        case PayoffType::SpreadCall:
            return std::max(S_T[0] - S_T[1] - K, 0.0);
        case PayoffType::Exchange:
            return std::max(S_T[0] - S_T[1], 0.0);
    }
    return 0.0;
}

// ============================================================================
// Correlated Normal Generation via Cholesky — N-dimensional
// ============================================================================
//
// Given correlation matrix R (N x N, SPD):
//   1. Compute L = cholesky(R)     →  R = L * L'
//   2. Generate Z = (Z_1,...,Z_N)  ~  iid N(0,1)
//   3. Correlated normals: W = L * Z
//
// Math: Cov(W) = L * Cov(Z) * L' = L * I * L' = L * L' = R  ✓
//
// This generalises the 2D Cholesky from Heston (Day 9):
//   Z_2 = rho*Z_1 + sqrt(1-rho^2)*Z_2
// to arbitrary N dimensions.
// ============================================================================

/// Generate N correlated standard normals from the Cholesky factor L
inline std::vector<double> generate_correlated_normals_nd(
    const QMatrix<double>& L, RandomNumberGenerator& rng)
{
    size_t N = L.rows();
    // Step 1: generate N independent standard normals
    auto Z = generate_normals(rng, N);

    // Step 2: multiply by Cholesky factor → W = L * Z
    // W_i = sum_j L(i,j) * Z_j  (only j <= i since L is lower triangular)
    std::vector<double> W(N, 0.0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            W[i] += L(i, j) * Z[j];
        }
    }
    return W;
}

// ============================================================================
// Monte Carlo Multi-Asset Pricing Engine
// ============================================================================
//
// Algorithm:
//   1. Cholesky-decompose the correlation matrix: R = L * L'
//   2. For each MC path:
//      a. Generate N independent standard normals Z
//      b. Correlate: W = L * Z
//      c. Evolve each asset: S_i(T) = S_i(0) * exp((r - σ_i²/2)T + σ_i√T W_i)
//      d. Evaluate the payoff at terminal spots
//   3. Price = exp(-rT) * mean(payoffs)
//
// Complexity: O(num_paths * N^2) per simulation (N^2 from Cholesky multiply)
//             Cholesky decomposition itself is O(N^3) but done once.
//
// For path-dependent multi-asset options, replace single-step evolution
// with n_steps time steps (not implemented here — terminal payoffs only).
// ============================================================================

inline MultiAssetMCResult mc_multi_asset(
    const MultiAssetParams& params,
    PayoffType payoff_type,
    size_t num_paths,
    unsigned long seed = 42)
{
    params.validate();
    size_t N = params.n_assets();

    // Step 1: Cholesky decomposition of correlation matrix
    // R = L * L'  →  L is lower triangular
    // This is done ONCE, not per path — O(N^3)
    QMatrix<double> L = cholesky(params.corr);

    MersenneTwisterRNG rng(seed);
    double discount = std::exp(-params.r * params.T);
    double sqrt_T = std::sqrt(params.T);

    // Precompute drift for each asset: (r - sigma_i^2/2) * T
    std::vector<double> drift(N);
    for (size_t i = 0; i < N; ++i) {
        drift[i] = (params.r - 0.5 * params.sigma[i] * params.sigma[i]) * params.T;
    }

    double sum = 0.0, sum_sq = 0.0;

    for (size_t p = 0; p < num_paths; ++p) {
        // Step 2a-b: generate N correlated normals via Cholesky
        auto W = generate_correlated_normals_nd(L, rng);

        // Step 2c: evolve each asset to terminal time
        // S_i(T) = S_i(0) * exp((r - sigma_i^2/2)*T + sigma_i*sqrt(T)*W_i)
        //        = S_i(0) * exp(drift_i + sigma_i * sqrt_T * W_i)
        std::vector<double> S_T(N);
        for (size_t i = 0; i < N; ++i) {
            S_T[i] = params.S0[i] * std::exp(drift[i] + params.sigma[i] * sqrt_T * W[i]);
        }

        // Step 2d: evaluate payoff
        double pv = discount * evaluate_payoff(payoff_type, S_T, params.weights, params.K);
        sum += pv;
        sum_sq += pv * pv;
    }

    // Step 3: price = mean of discounted payoffs
    double mean = sum / num_paths;
    double var = (sum_sq / num_paths) - mean * mean;
    double se = std::sqrt(std::max(var, 0.0) / num_paths);

    return {mean, se, mean - 1.96 * se, mean + 1.96 * se, num_paths};
}

// ============================================================================
// Margrabe's Exchange Option Formula (1978) — Analytic Benchmark
// ============================================================================
//
// The exchange option gives the holder the right to exchange asset 2 for asset 1:
//   payoff = max(S_1(T) - S_2(T), 0)
//
// This is equivalent to a call on S_1 with "strike" S_2, using S_2 as numeraire.
//
// CLOSED-FORM (Margrabe 1978):
//
//   V = S_1 * N(d1) - S_2 * N(d2)
//
//   d1 = [ln(S_1/S_2) + sigma_spread^2 * T / 2] / (sigma_spread * sqrt(T))
//   d2 = d1 - sigma_spread * sqrt(T)
//
//   sigma_spread = sqrt(sigma_1^2 + sigma_2^2 - 2*rho*sigma_1*sigma_2)
//
// KEY PROPERTIES:
//   - NO discounting! (S_2 is the numeraire, not cash)
//   - Does NOT depend on the risk-free rate r
//   - Reduces to BS call when sigma_2 = 0 (asset 2 is riskless bond)
//   - sigma_spread is the vol of the ratio S_1/S_2
//
// Math derivation:
//   Under the S_2-forward measure, S_1/S_2 is a martingale with vol sigma_spread.
//   By Ito's lemma on log(S_1/S_2):
//     d(ln(S_1/S_2)) = -sigma_spread^2/2 * dt + sigma_1*dW_1 - sigma_2*dW_2
//   The vol of the log-ratio is sigma_spread (by variance of the diffusion terms,
//   using Var(sigma_1*dW_1 - sigma_2*dW_2) = (sigma_1^2 + sigma_2^2 - 2*rho*sigma_1*sigma_2)*dt).
// ============================================================================

struct MargrabeResult {
    double price;
    double sigma_spread;  // Effective volatility of S1/S2
    double d1;
    double d2;
};

inline MargrabeResult margrabe_price(
    double S1, double S2,
    double sigma1, double sigma2, double rho,
    double T)
{
    // Spread volatility: vol of log(S1/S2)
    // sigma_spread = sqrt(sigma_1^2 + sigma_2^2 - 2*rho*sigma_1*sigma_2)
    double sigma_sq = sigma1 * sigma1 + sigma2 * sigma2 - 2.0 * rho * sigma1 * sigma2;
    if (sigma_sq < 0.0) sigma_sq = 0.0;  // Guard against numerical issues
    double sigma_spread = std::sqrt(sigma_sq);

    double sigma_sqrtT = sigma_spread * std::sqrt(T);

    // d1 = [ln(S1/S2) + sigma_spread^2 * T / 2] / (sigma_spread * sqrt(T))
    double d1 = (std::log(S1 / S2) + 0.5 * sigma_sq * T) / sigma_sqrtT;
    double d2 = d1 - sigma_sqrtT;

    // V = S1 * N(d1) - S2 * N(d2)   — no discounting!
    double price = S1 * bs::N(d1) - S2 * bs::N(d2);

    return {price, sigma_spread, d1, d2};
}

// ============================================================================
// Basket Volatility — Analytic Approximation
// ============================================================================
//
// For a basket B = sum(w_i * S_i), the basket variance under GBM is:
//
//   sigma_basket^2 ≈ sum_i sum_j w_i * w_j * sigma_i * sigma_j * rho_ij
//                       * (S_i(0) / B(0)) * (S_j(0) / B(0))
//
// This is the "moment-matching" approximation: treat the basket as a single
// lognormal asset with matched first two moments.
//
// Basket forward: F_B = sum(w_i * S_i(0) * exp(r*T))
// Basket vol:     sigma_B as above
// Then use BS with F_B and sigma_B for an approximate basket option price.
//
// This is a quick-and-dirty approximation — MC is the gold standard.
// ============================================================================

inline double basket_vol(const MultiAssetParams& params) {
    size_t N = params.n_assets();
    const auto& w = params.weights;

    // Basket spot value
    double B0 = 0.0;
    for (size_t i = 0; i < N; ++i)
        B0 += w[i] * params.S0[i];

    // Basket variance ≈ sum_i sum_j w_i*w_j * sigma_i*sigma_j * rho_ij * (S_i/B) * (S_j/B)
    double var = 0.0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            var += w[i] * w[j]
                 * params.sigma[i] * params.sigma[j]
                 * params.corr(i, j)
                 * (params.S0[i] / B0) * (params.S0[j] / B0);
        }
    }
    return std::sqrt(std::max(var, 0.0));
}

/// Approximate basket call price using lognormal moment-matching + BS formula
inline double basket_call_approx(const MultiAssetParams& params) {
    double B0 = 0.0;
    for (size_t i = 0; i < params.n_assets(); ++i)
        B0 += params.weights[i] * params.S0[i];

    double sigma_B = basket_vol(params);
    return bs::call_price(B0, params.K, params.r, params.T, sigma_B);
}

/// Approximate basket put price using lognormal moment-matching + BS formula
inline double basket_put_approx(const MultiAssetParams& params) {
    double B0 = 0.0;
    for (size_t i = 0; i < params.n_assets(); ++i)
        B0 += params.weights[i] * params.S0[i];

    double sigma_B = basket_vol(params);
    return bs::put_price(B0, params.K, params.r, params.T, sigma_B);
}

// ============================================================================
// Helper: Build uniform correlation matrix (all off-diag = rho)
// ============================================================================

inline QMatrix<double> uniform_corr_matrix(size_t N, double rho) {
    if (rho < -1.0 / (static_cast<double>(N) - 1.0))
        throw std::invalid_argument("rho too negative for PSD matrix");
    QMatrix<double> R(N, N, rho);
    for (size_t i = 0; i < N; ++i)
        R(i, i) = 1.0;
    return R;
}

// ============================================================================
// Helper: Build equal-weighted params for common use cases
// ============================================================================

inline MultiAssetParams make_basket_params(
    size_t N, double S0, double sigma, double rho,
    double r, double T, double K)
{
    MultiAssetParams p;
    p.S0.assign(N, S0);
    p.sigma.assign(N, sigma);
    p.weights.assign(N, 1.0 / static_cast<double>(N));
    p.corr = uniform_corr_matrix(N, rho);
    p.r = r;
    p.T = T;
    p.K = K;
    return p;
}

} // namespace multi_asset

#endif // QUANTPRICER_MULTI_ASSET_H

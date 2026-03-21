#ifndef QUANTPRICER_RISK_H
#define QUANTPRICER_RISK_H

// ============================================================================
// Risk Management — Day 12
//
// ============================================================================
// MATHEMATICS — Value at Risk & Coherent Risk Measures
// ============================================================================
//
// VALUE AT RISK (VaR):
//   "The maximum loss over a given time horizon at a given confidence level."
//
//   VaR_alpha = -inf{ x : P(L <= x) >= alpha }
//             = -quantile(P&L distribution, 1 - alpha)
//
//   Example: 1-day 99% VaR = $10M means:
//     "There is a 1% chance of losing more than $10M tomorrow."
//
//   Three approaches:
//     1. Parametric (Normal):  VaR = -mu + z_alpha * sigma
//        where z_alpha = N^{-1}(alpha), e.g. z_0.99 = 2.326
//     2. Historical:           sort historical P&L, take (1-alpha) percentile
//     3. Monte Carlo:          simulate P&L paths, take percentile
//
// ============================================================================
// CONDITIONAL VAR (CVaR) / EXPECTED SHORTFALL (ES):
//
//   CVaR_alpha = E[L | L > VaR_alpha]
//              = expected loss given that we're in the worst (1-alpha) tail
//
//   For Normal distribution:  CVaR = mu + sigma * phi(z_alpha) / (1 - alpha)
//   where phi = standard normal PDF
//
//   Key interview insight: WHY CVaR > VaR?
//     CVaR is the AVERAGE of all losses beyond VaR. Since VaR is just the
//     threshold, CVaR captures the severity of tail losses, not just frequency.
//
// ============================================================================
// COHERENT RISK MEASURES (Artzner et al., 1999):
//
//   A risk measure rho is COHERENT if it satisfies:
//     1. Monotonicity:     X <= Y  =>  rho(X) >= rho(Y)
//     2. Subadditivity:    rho(X+Y) <= rho(X) + rho(Y)     ← diversification!
//     3. Positive homogeneity:  rho(lambda*X) = lambda*rho(X)  for lambda > 0
//     4. Translation invariance: rho(X + c) = rho(X) - c
//
//   VaR is NOT coherent — it violates SUBADDITIVITY:
//     There exist portfolios A, B where VaR(A+B) > VaR(A) + VaR(B)
//     This means VaR can penalise diversification — absurd!
//
//   CVaR IS coherent — it satisfies all four axioms.
//   This is why Basel III (2013) switched from VaR to Expected Shortfall.
//
// ============================================================================
// PORTFOLIO RISK DECOMPOSITION:
//
//   For a portfolio with weights w and covariance matrix Sigma:
//     Portfolio variance:   sigma_p^2 = w' * Sigma * w
//     Portfolio vol:        sigma_p = sqrt(w' * Sigma * w)
//     Parametric VaR:       VaR = z_alpha * sigma_p * sqrt(dt)
//
//   Marginal VaR:    partial VaR / partial w_i
//     = z_alpha * (Sigma * w)_i / sigma_p
//     Interpretation: how much does VaR change if I add $1 to asset i?
//
//   Component VaR:   CVaR_i = w_i * MVaR_i
//     Property: sum(CVaR_i) = Total VaR  (Euler decomposition)
//     Interpretation: asset i's contribution to total portfolio risk.
//
// ============================================================================
// PERFORMANCE RATIOS:
//
//   Sharpe Ratio:    SR = (mu_p - r_f) / sigma_p
//     Reward per unit of TOTAL risk. Industry standard benchmark.
//
//   Sortino Ratio:   Sort = (mu_p - r_f) / sigma_downside
//     Only penalises DOWNSIDE volatility. Better for asymmetric returns.
//     sigma_downside = sqrt(E[min(R - r_f, 0)^2])
//
//   Calmar Ratio:    Calmar = annualised_return / max_drawdown
//     Reward per unit of worst peak-to-trough decline.
//
//   Max Drawdown:    MDD = max_t (peak_t - value_t) / peak_t
//     The largest percentage drop from peak. Computed with running maximum.
//
// ============================================================================
// KEY INTERVIEW INSIGHTS:
//
// 1. "VaR tells you the minimum loss in the worst 1% of cases.
//     CVaR tells you the AVERAGE loss in the worst 1% of cases."
//
// 2. VaR subadditivity counterexample: two binary options that individually
//    have low VaR but together can blow up (concentrated tail risk).
//
// 3. "Why did Basel III switch from VaR to ES?"
//    Because VaR can encourage risk concentration in the tails — a bank
//    could restructure positions so individual VaRs look small but the
//    combined tail risk is catastrophic. ES prevents this.
//
// 4. Sharpe ratio is scale-invariant (doubling leverage doesn't change it)
//    but Sortino better captures strategies with positive skew (e.g. trend-following).
//
// 5. Max drawdown is path-dependent — two strategies with identical return/vol
//    can have very different drawdowns depending on the ORDER of returns.
// ============================================================================

#include "matrix/matrix.h"
#include "rng/rng.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace risk {

// ============================================================================
// Risk Measure Results
// ============================================================================

struct VaRResult {
    double var;             // Value at Risk (positive = loss)
    double cvar;            // Conditional VaR / Expected Shortfall
    double confidence;      // Confidence level (e.g. 0.99)
    size_t num_scenarios;   // Number of scenarios used
};

struct PerformanceMetrics {
    double mean_return;
    double volatility;
    double sharpe;
    double sortino;
    double calmar;
    double max_drawdown;
    double skewness;
    double kurtosis;
};

struct RiskDecomposition {
    std::vector<double> marginal_var;   // dVaR/dw_i
    std::vector<double> component_var;  // w_i * marginal_var_i
    double total_var;                   // sum(component_var) = portfolio VaR
};

// ============================================================================
// Parametric (Normal) VaR & CVaR
// ============================================================================
//
// Assumes P&L ~ N(mu, sigma^2) over the horizon.
//
// VaR_alpha  = -(mu - z_alpha * sigma)  = -mu + z_alpha * sigma
//   where z_alpha = Phi^{-1}(alpha), e.g. z_0.99 = 2.3263
//
// CVaR_alpha = -mu + sigma * phi(z_alpha) / (1 - alpha)
//   where phi = standard normal PDF
//
// For portfolio: sigma = sqrt(w' * Sigma * w)
// ============================================================================

/// Standard normal quantile (inverse CDF) via rational approximation
inline double normal_quantile(double p) {
    // Abramowitz & Stegun approximation (same as in rng.h)
    if (p <= 0.0) return -1e10;
    if (p >= 1.0) return 1e10;
    if (p > 0.5) return -normal_quantile(1.0 - p);

    double t = std::sqrt(-2.0 * std::log(p));
    double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
    double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
    return -(t - (c0 + c1 * t + c2 * t * t) /
                 (1.0 + d1 * t + d2 * t * t + d3 * t * t * t));
}

/// Standard normal PDF
inline double normal_pdf(double x) {
    return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
}

/// Parametric VaR for a single asset or portfolio
/// mu = expected P&L, sigma = P&L volatility, alpha = confidence (e.g. 0.99)
inline VaRResult parametric_var(double mu, double sigma, double alpha) {
    // z_alpha = Phi^{-1}(alpha) — the critical value
    double z = normal_quantile(alpha);

    // VaR = -(mu - z * sigma) = -mu + z * sigma
    // Convention: VaR is reported as a POSITIVE number representing loss
    double var = -mu + z * sigma;

    // CVaR = -mu + sigma * phi(z) / (1 - alpha)
    // phi(z) = standard normal PDF evaluated at z
    double cvar = -mu + sigma * normal_pdf(z) / (1.0 - alpha);

    return {var, cvar, alpha, 0};
}

/// Portfolio parametric VaR given weights and covariance matrix
/// w = portfolio weights (dollar amounts or fractional)
/// Sigma = covariance matrix of returns
/// mu_vec = expected returns per asset
/// alpha = confidence level
inline VaRResult portfolio_parametric_var(
    const std::vector<double>& w,
    const QMatrix<double>& Sigma,
    const std::vector<double>& mu_vec,
    double alpha)
{
    size_t N = w.size();

    // Portfolio expected return: mu_p = w' * mu
    double mu_p = 0.0;
    for (size_t i = 0; i < N; ++i)
        mu_p += w[i] * mu_vec[i];

    // Portfolio variance: sigma_p^2 = w' * Sigma * w
    double var_p = 0.0;
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            var_p += w[i] * w[j] * Sigma(i, j);
    double sigma_p = std::sqrt(std::max(var_p, 0.0));

    return parametric_var(mu_p, sigma_p, alpha);
}

// ============================================================================
// Historical / Monte Carlo VaR & CVaR
// ============================================================================
//
// Given a vector of P&L scenarios (simulated or historical):
//   1. Sort in ascending order (worst to best)
//   2. VaR  = -P&L at the (1-alpha) percentile
//   3. CVaR = -mean of all P&L below the VaR threshold
//
// No distributional assumptions — fully nonparametric.
// ============================================================================

/// Compute VaR and CVaR from a vector of P&L scenarios
/// pnl = vector of profit/loss values (positive = profit, negative = loss)
/// alpha = confidence level (e.g. 0.99)
inline VaRResult scenario_var(std::vector<double> pnl, double alpha) {
    if (pnl.empty()) throw std::invalid_argument("P&L vector is empty");

    size_t n = pnl.size();
    std::sort(pnl.begin(), pnl.end());  // ascending: worst first

    // VaR: the (1-alpha) quantile
    // Index: floor((1-alpha) * n)
    size_t var_idx = static_cast<size_t>(std::floor((1.0 - alpha) * n));
    if (var_idx >= n) var_idx = n - 1;
    double var = -pnl[var_idx];  // Positive = loss

    // CVaR: mean of all scenarios at or below VaR threshold
    // These are the worst (1-alpha) fraction of outcomes
    double tail_sum = 0.0;
    size_t tail_count = var_idx + 1;  // Include the VaR point
    for (size_t i = 0; i <= var_idx; ++i)
        tail_sum += pnl[i];
    double cvar = -tail_sum / tail_count;  // Positive = loss

    return {var, cvar, alpha, n};
}

// ============================================================================
// Monte Carlo Portfolio VaR
// ============================================================================
//
// Simulate correlated asset returns using Cholesky decomposition:
//   1. Cholesky: Sigma = L * L'
//   2. For each scenario:
//      a. Generate N independent normals Z
//      b. Correlated returns: R = mu + L * Z
//      c. Portfolio P&L = w' * R
//   3. Compute VaR/CVaR from the P&L distribution
//
// This naturally captures non-linear effects, fat tails, and correlations.
// ============================================================================

inline VaRResult mc_portfolio_var(
    const std::vector<double>& w,
    const std::vector<double>& mu_vec,
    const QMatrix<double>& cov,
    double alpha,
    size_t num_scenarios = 100000,
    unsigned long seed = 42)
{
    size_t N = w.size();

    // Cholesky decomposition of covariance matrix
    QMatrix<double> L = cholesky(cov);

    MersenneTwisterRNG rng(seed);
    std::vector<double> pnl(num_scenarios);

    for (size_t s = 0; s < num_scenarios; ++s) {
        // Generate N independent standard normals
        auto Z = generate_normals(rng, N);

        // Correlated returns: R_i = mu_i + sum_j L(i,j) * Z_j
        double portfolio_pnl = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double R_i = mu_vec[i];
            for (size_t j = 0; j <= i; ++j)
                R_i += L(i, j) * Z[j];
            portfolio_pnl += w[i] * R_i;
        }
        pnl[s] = portfolio_pnl;
    }

    return scenario_var(pnl, alpha);
}

// ============================================================================
// Risk Decomposition — Marginal & Component VaR
// ============================================================================
//
// Marginal VaR: how much VaR changes per unit increase in position i
//   MVaR_i = z_alpha * (Sigma * w)_i / sigma_p
//
// Component VaR: each asset's contribution to total VaR
//   CVaR_i = w_i * MVaR_i
//
// KEY PROPERTY (Euler's theorem for homogeneous functions):
//   sum(CVaR_i) = Total VaR
//
// This decomposition tells you WHERE the risk is coming from.
// ============================================================================

inline RiskDecomposition decompose_var(
    const std::vector<double>& w,
    const QMatrix<double>& Sigma,
    double alpha)
{
    size_t N = w.size();
    double z = normal_quantile(alpha);

    // Portfolio variance: sigma_p^2 = w' * Sigma * w
    double var_p = 0.0;
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            var_p += w[i] * w[j] * Sigma(i, j);
    double sigma_p = std::sqrt(std::max(var_p, 0.0));

    // Sigma * w vector
    std::vector<double> Sigma_w(N, 0.0);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            Sigma_w[i] += Sigma(i, j) * w[j];

    // Marginal VaR: z * (Sigma * w)_i / sigma_p
    std::vector<double> mvar(N);
    for (size_t i = 0; i < N; ++i)
        mvar[i] = z * Sigma_w[i] / sigma_p;

    // Component VaR: w_i * MVaR_i
    std::vector<double> cvar(N);
    double total = 0.0;
    for (size_t i = 0; i < N; ++i) {
        cvar[i] = w[i] * mvar[i];
        total += cvar[i];
    }

    return {mvar, cvar, total};
}

// ============================================================================
// Performance Metrics — Sharpe, Sortino, Calmar, Max Drawdown
// ============================================================================

/// Compute all performance metrics from a return series
/// returns = vector of periodic returns (e.g. daily)
/// rf = risk-free rate per period (e.g. daily rf)
/// periods_per_year = annualisation factor (252 for daily, 12 for monthly)
inline PerformanceMetrics compute_metrics(
    const std::vector<double>& returns,
    double rf = 0.0,
    double periods_per_year = 252.0)
{
    size_t n = returns.size();
    if (n < 2) throw std::invalid_argument("Need at least 2 returns");

    // Mean return
    double sum = 0.0;
    for (double r : returns) sum += r;
    double mu = sum / n;

    // Volatility (standard deviation)
    double sq_sum = 0.0;
    for (double r : returns) sq_sum += (r - mu) * (r - mu);
    double vol = std::sqrt(sq_sum / (n - 1));

    // Downside deviation (for Sortino)
    // sigma_down = sqrt(E[min(R - rf, 0)^2])
    double down_sum = 0.0;
    for (double r : returns) {
        double diff = r - rf;
        if (diff < 0.0) down_sum += diff * diff;
    }
    double downside_vol = std::sqrt(down_sum / n);

    // Skewness: E[(R - mu)^3] / sigma^3
    double skew_sum = 0.0;
    for (double r : returns) {
        double d = (r - mu) / vol;
        skew_sum += d * d * d;
    }
    double skewness = skew_sum / n;

    // Excess Kurtosis: E[(R - mu)^4] / sigma^4 - 3
    double kurt_sum = 0.0;
    for (double r : returns) {
        double d = (r - mu) / vol;
        kurt_sum += d * d * d * d;
    }
    double kurtosis = kurt_sum / n - 3.0;

    // Max Drawdown — track running peak
    // Compute equity curve: cumulative returns
    double equity = 1.0;
    double peak = 1.0;
    double max_dd = 0.0;
    for (double r : returns) {
        equity *= (1.0 + r);
        if (equity > peak) peak = equity;
        double dd = (peak - equity) / peak;
        if (dd > max_dd) max_dd = dd;
    }

    // Annualise
    double ann_return = mu * periods_per_year;
    double ann_vol = vol * std::sqrt(periods_per_year);
    double ann_rf = rf * periods_per_year;
    double ann_down_vol = downside_vol * std::sqrt(periods_per_year);

    // Sharpe: (annualised_return - rf) / annualised_vol
    double sharpe = (ann_vol > 1e-15) ? (ann_return - ann_rf) / ann_vol : 0.0;

    // Sortino: (annualised_return - rf) / annualised_downside_vol
    double sortino = (ann_down_vol > 1e-15) ? (ann_return - ann_rf) / ann_down_vol : 0.0;

    // Calmar: annualised_return / max_drawdown
    double calmar = (max_dd > 1e-15) ? ann_return / max_dd : 0.0;

    return {ann_return, ann_vol, sharpe, sortino, calmar, max_dd, skewness, kurtosis};
}

// ============================================================================
// Stress Testing — Scenario-based P&L
// ============================================================================
//
// Given a portfolio and a set of stress scenarios (e.g. "2008 crisis",
// "rates +200bp", "equity -20%"), compute the P&L under each scenario.
//
// Scenario: vector of returns for each asset
// P&L = w' * scenario_returns
// ============================================================================

struct StressResult {
    std::string name;
    std::vector<double> asset_returns;
    double portfolio_pnl;
};

inline std::vector<StressResult> stress_test(
    const std::vector<double>& w,
    const std::vector<std::pair<std::string, std::vector<double>>>& scenarios)
{
    std::vector<StressResult> results;
    for (auto& [name, returns] : scenarios) {
        double pnl = 0.0;
        for (size_t i = 0; i < w.size() && i < returns.size(); ++i)
            pnl += w[i] * returns[i];
        results.push_back({name, returns, pnl});
    }
    return results;
}

// ============================================================================
// VaR Subadditivity Counterexample — Interview Classic
// ============================================================================
//
// Demonstrates that VaR violates subadditivity:
//   VaR(A + B) > VaR(A) + VaR(B)
//
// Construction: two digital options, each pays $100 with probability 4%.
// At 95% confidence:
//   - VaR(A) = 0 (95% of time, no loss)
//   - VaR(B) = 0 (95% of time, no loss)
//   - VaR(A+B) > 0 (because P(at least one pays) = 1-(0.96)^2 = 7.84% > 5%)
//
// So VaR(A+B) > VaR(A) + VaR(B) = 0. Diversification is "penalised"!
// ============================================================================

struct SubadditivityDemo {
    double var_A;
    double var_B;
    double var_AB;
    bool violated;  // true if VaR(A+B) > VaR(A) + VaR(B)
};

inline SubadditivityDemo demonstrate_subadditivity(
    size_t num_scenarios = 1000000,
    unsigned long seed = 42)
{
    MersenneTwisterRNG rng(seed);

    // Two independent binary bets:
    // Each loses $100 with prob 4%, gains $0 with prob 96%
    // Premium paid = $4 (fair price), so P&L = {-96 with prob 4%, +4 with prob 96%}
    double loss = -96.0;   // lose $100 minus $4 premium
    double gain = 4.0;     // keep $4 premium
    double prob_loss = 0.04;

    std::vector<double> pnl_A(num_scenarios), pnl_B(num_scenarios), pnl_AB(num_scenarios);

    for (size_t i = 0; i < num_scenarios; ++i) {
        double u1 = rng.generate_uniform();
        double u2 = rng.generate_uniform();
        pnl_A[i]  = (u1 < prob_loss) ? loss : gain;
        pnl_B[i]  = (u2 < prob_loss) ? loss : gain;
        pnl_AB[i] = pnl_A[i] + pnl_B[i];
    }

    double alpha = 0.95;
    auto var_a  = scenario_var(pnl_A, alpha);
    auto var_b  = scenario_var(pnl_B, alpha);
    auto var_ab = scenario_var(pnl_AB, alpha);

    bool violated = var_ab.var > var_a.var + var_b.var + 1e-10;

    return {var_a.var, var_b.var, var_ab.var, violated};
}

// ============================================================================
// Simulate Correlated Return Series — for testing
// ============================================================================

inline std::vector<std::vector<double>> simulate_returns(
    const std::vector<double>& mu,
    const QMatrix<double>& cov,
    size_t n_days,
    unsigned long seed = 42)
{
    size_t N = mu.size();
    QMatrix<double> L = cholesky(cov);
    MersenneTwisterRNG rng(seed);

    // returns[asset][day]
    std::vector<std::vector<double>> returns(N, std::vector<double>(n_days));

    for (size_t d = 0; d < n_days; ++d) {
        auto Z = generate_normals(rng, N);
        for (size_t i = 0; i < N; ++i) {
            double R_i = mu[i];
            for (size_t j = 0; j <= i; ++j)
                R_i += L(i, j) * Z[j];
            returns[i][d] = R_i;
        }
    }
    return returns;
}

// ============================================================================
// Build covariance matrix from volatilities and correlation matrix
// ============================================================================

inline QMatrix<double> build_cov_matrix(
    const std::vector<double>& sigma,
    const QMatrix<double>& corr)
{
    size_t N = sigma.size();
    QMatrix<double> cov(N, N);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            cov(i, j) = sigma[i] * sigma[j] * corr(i, j);
    return cov;
}

} // namespace risk

#endif // QUANTPRICER_RISK_H

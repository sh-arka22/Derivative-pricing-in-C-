#ifndef QUANTPRICER_RATE_MODELS_H
#define QUANTPRICER_RATE_MODELS_H

// ============================================================================
// Interest Rate Models — Day 14
//
// ============================================================================
// THE SHORT RATE FRAMEWORK
// ============================================================================
//
// The short rate r(t) is the instantaneous risk-free rate at time t.
// Under the risk-neutral measure Q, the price of a zero-coupon bond is:
//
//   P(t,T) = E^Q[ exp(-integral_t^T r(s) ds) | F_t ]
//
// Different models specify different SDEs for r(t). The challenge:
//   - Must capture mean reversion (rates don't wander to infinity)
//   - Should produce realistic yield curve shapes
//   - Ideally have analytic bond prices (for calibration speed)
//
// ============================================================================
// MODEL 1: VASICEK (1977)
// ============================================================================
//
//   dr = kappa * (theta - r) * dt + sigma * dW
//
//   - Ornstein-Uhlenbeck process (Gaussian, mean-reverting)
//   - kappa = mean reversion speed (higher = faster pull to theta)
//   - theta = long-run mean level
//   - sigma = volatility of the short rate
//
//   Distribution: r(t) | r(0) ~ N(mu(t), V(t)) where
//     mu(t) = theta + (r0 - theta) * exp(-kappa*t)
//     V(t)  = sigma^2 / (2*kappa) * (1 - exp(-2*kappa*t))
//
//   ANALYTIC BOND PRICE:
//     P(t,T) = A(t,T) * exp(-B(t,T) * r(t))
//
//     B(tau) = (1 - exp(-kappa*tau)) / kappa        where tau = T - t
//     A(tau) = exp([(B(tau) - tau) * (kappa^2*theta - sigma^2/2) / kappa^2]
//                  - sigma^2 * B(tau)^2 / (4*kappa))
//
//   Key interview insight: VASICEK ALLOWS NEGATIVE RATES.
//     r(t) is Gaussian → can go below zero. Pre-2008 this was seen as a
//     flaw. Post-2008 (EUR, JPY, CHF negative rates): actually realistic!
//     "The 'bug' became a feature."
//
// ============================================================================
// MODEL 2: COX-INGERSOLL-ROSS (CIR, 1985)
// ============================================================================
//
//   dr = kappa * (theta - r) * dt + sigma * sqrt(r) * dW
//
//   - Square-root diffusion (non-central chi-squared distribution)
//   - The sqrt(r) term ensures volatility → 0 as r → 0
//
//   FELLER CONDITION: 2*kappa*theta > sigma^2
//     If satisfied: r(t) > 0 almost surely (never touches zero)
//     If violated: r can reach zero but is reflected (stays non-negative)
//
//   ANALYTIC BOND PRICE:
//     P(t,T) = A(tau) * exp(-B(tau) * r(t))
//
//     gamma = sqrt(kappa^2 + 2*sigma^2)
//     B(tau) = 2*(exp(gamma*tau) - 1) / [(gamma+kappa)*(exp(gamma*tau)-1) + 2*gamma]
//     A(tau) = [2*gamma*exp((kappa+gamma)*tau/2) /
//               ((gamma+kappa)*(exp(gamma*tau)-1) + 2*gamma)]^(2*kappa*theta/sigma^2)
//
//   Key interview insight: CIR vs Vasicek
//     "CIR guarantees positive rates but has a more complex distribution
//     (non-central chi-squared vs Gaussian). Vasicek is simpler but allows
//     negative rates. Choose based on the market you're modelling."
//
// ============================================================================
// MODEL 3: HULL-WHITE (1990) — Extended Vasicek
// ============================================================================
//
//   dr = [theta(t) - a*r] * dt + sigma * dW
//
//   - theta(t) is a TIME-DEPENDENT drift chosen to FIT THE INITIAL CURVE
//   - This is the key advantage: it EXACTLY reprices today's bond prices
//
//   Calibration: theta(t) = dF(0,t)/dt + a*F(0,t) + sigma^2/(2*a) * (1-exp(-2*a*t))
//   where F(0,t) = instantaneous forward rate from the market curve
//
//   ANALYTIC BOND PRICE (given the initial curve):
//     P(t,T) = P(0,T)/P(0,t) * exp(-B(tau)*[r(t) - f(0,t)] - sigma^2/(4*a)*B(tau)^2*(1-exp(-2*a*t)))
//     where B(tau) = (1 - exp(-a*tau))/a, tau = T-t
//
//   Key interview insight: WHY HULL-WHITE?
//     "Vasicek/CIR have constant parameters — they can't match today's
//     yield curve exactly. Hull-White adds theta(t) to guarantee the model
//     prices all traded bonds correctly. This is essential for relative-value
//     trading: you need the model to agree with the market before you can
//     use it to find mispricings."
//
// ============================================================================
// SIMULATION SCHEMES:
//
// Vasicek (additive noise): Euler-Maruyama is exact in distribution
//   r(t+dt) = r(t) + kappa*(theta - r(t))*dt + sigma*sqrt(dt)*Z
//   Actually EXACT: r(t+dt) = mu(dt) + sqrt(V(dt)) * Z  (Gaussian transition)
//
// CIR (multiplicative noise): Milstein improves convergence
//   Euler:    r(t+dt) = r(t) + kappa*(theta-r(t))*dt + sigma*sqrt(r(t)*dt)*Z
//   Milstein: + 0.25*sigma^2*(Z^2-1)*dt   (correction term)
//   Full truncation: r_pos = max(r, 0) to prevent negative values
//
// Hull-White: same as Vasicek but with time-dependent theta(t)
// ============================================================================

#include "rng/rng.h"
#include "fixed_income/fixed_income.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace rates {

// ============================================================================
// Model Parameters
// ============================================================================

struct VasicekParams {
    double r0;      // Initial short rate
    double kappa;   // Mean reversion speed
    double theta;   // Long-run mean
    double sigma;   // Volatility

    VasicekParams() : r0(0.05), kappa(0.5), theta(0.05), sigma(0.01) {}
    VasicekParams(double r, double k, double th, double s)
        : r0(r), kappa(k), theta(th), sigma(s) {}
};

struct CIRParams {
    double r0;
    double kappa;
    double theta;
    double sigma;

    CIRParams() : r0(0.05), kappa(0.5), theta(0.05), sigma(0.05) {}
    CIRParams(double r, double k, double th, double s)
        : r0(r), kappa(k), theta(th), sigma(s) {}

    /// Feller condition: 2*kappa*theta > sigma^2 ensures r > 0 a.s.
    bool feller_satisfied() const {
        return 2.0 * kappa * theta > sigma * sigma;
    }
};

struct HullWhiteParams {
    double a;       // Mean reversion speed
    double sigma;   // Volatility
    // theta(t) is calibrated to the initial curve — not stored as a constant

    HullWhiteParams() : a(0.1), sigma(0.01) {}
    HullWhiteParams(double a_val, double s) : a(a_val), sigma(s) {}
};

// ============================================================================
// Vasicek Analytic Bond Pricing
// ============================================================================
//
// P(t,T) = A(tau) * exp(-B(tau) * r(t))     where tau = T - t
//
// B(tau) = (1 - exp(-kappa*tau)) / kappa
// A(tau) = exp([(B - tau)*(kappa^2*theta - sigma^2/2)/kappa^2]
//              - sigma^2*B^2/(4*kappa))
// ============================================================================

/// Vasicek B(tau) function
inline double vasicek_B(double tau, double kappa) {
    if (std::abs(kappa) < 1e-10) return tau;  // limit kappa→0
    return (1.0 - std::exp(-kappa * tau)) / kappa;
}

/// Vasicek A(tau) function (returns ln(A) for numerical stability)
inline double vasicek_lnA(double tau, const VasicekParams& p) {
    double B = vasicek_B(tau, p.kappa);
    double kappa_sq = p.kappa * p.kappa;
    double sigma_sq = p.sigma * p.sigma;

    // ln(A) = (B - tau) * (kappa^2*theta - sigma^2/2) / kappa^2
    //       - sigma^2 * B^2 / (4*kappa)
    double term1 = (B - tau) * (kappa_sq * p.theta - sigma_sq / 2.0) / kappa_sq;
    double term2 = -sigma_sq * B * B / (4.0 * p.kappa);
    return term1 + term2;
}

/// Vasicek zero-coupon bond price P(0,T) given r(0)
inline double vasicek_bond_price(double T, const VasicekParams& p) {
    double B = vasicek_B(T, p.kappa);
    double lnA = vasicek_lnA(T, p);
    return std::exp(lnA - B * p.r0);
}

/// Vasicek zero rate: r(T) = -ln(P(0,T)) / T
inline double vasicek_zero_rate(double T, const VasicekParams& p) {
    if (T <= 1e-10) return p.r0;
    return -std::log(vasicek_bond_price(T, p)) / T;
}

/// Build yield curve from Vasicek model
inline fixed_income::YieldCurve vasicek_curve(
    const VasicekParams& p,
    const std::vector<double>& tenors = {0.25,0.5,1,2,3,5,7,10,15,20,30})
{
    std::vector<double> zeros(tenors.size());
    for (size_t i = 0; i < tenors.size(); ++i)
        zeros[i] = vasicek_zero_rate(tenors[i], p);
    return fixed_income::YieldCurve(tenors, zeros);
}

/// Vasicek conditional mean and variance
/// E[r(t) | r(0)] = theta + (r0-theta)*exp(-kappa*t)
/// Var[r(t) | r(0)] = sigma^2/(2*kappa) * (1-exp(-2*kappa*t))
inline double vasicek_mean(double t, const VasicekParams& p) {
    return p.theta + (p.r0 - p.theta) * std::exp(-p.kappa * t);
}

inline double vasicek_var(double t, const VasicekParams& p) {
    return p.sigma * p.sigma / (2.0 * p.kappa) * (1.0 - std::exp(-2.0 * p.kappa * t));
}

// ============================================================================
// CIR Analytic Bond Pricing
// ============================================================================
//
// P(t,T) = A(tau) * exp(-B(tau) * r(t))     where tau = T - t
//
// gamma = sqrt(kappa^2 + 2*sigma^2)
// B(tau) = 2*(exp(gamma*tau)-1) / [(gamma+kappa)*(exp(gamma*tau)-1) + 2*gamma]
// A(tau) = [2*gamma*exp((kappa+gamma)*tau/2) /
//           ((gamma+kappa)*(exp(gamma*tau)-1) + 2*gamma)]^(2*kappa*theta/sigma^2)
// ============================================================================

/// CIR B(tau) function
inline double cir_B(double tau, const CIRParams& p) {
    double gamma = std::sqrt(p.kappa * p.kappa + 2.0 * p.sigma * p.sigma);
    double eg = std::exp(gamma * tau);
    return 2.0 * (eg - 1.0) / ((gamma + p.kappa) * (eg - 1.0) + 2.0 * gamma);
}

/// CIR ln(A(tau)) function
inline double cir_lnA(double tau, const CIRParams& p) {
    double gamma = std::sqrt(p.kappa * p.kappa + 2.0 * p.sigma * p.sigma);
    double eg = std::exp(gamma * tau);
    double denom = (gamma + p.kappa) * (eg - 1.0) + 2.0 * gamma;
    double exponent = 2.0 * p.kappa * p.theta / (p.sigma * p.sigma);
    return exponent * std::log(2.0 * gamma * std::exp((p.kappa + gamma) * tau / 2.0) / denom);
}

/// CIR zero-coupon bond price P(0,T)
inline double cir_bond_price(double T, const CIRParams& p) {
    double B = cir_B(T, p);
    double lnA = cir_lnA(T, p);
    return std::exp(lnA - B * p.r0);
}

/// CIR zero rate
inline double cir_zero_rate(double T, const CIRParams& p) {
    if (T <= 1e-10) return p.r0;
    return -std::log(cir_bond_price(T, p)) / T;
}

/// Build yield curve from CIR model
inline fixed_income::YieldCurve cir_curve(
    const CIRParams& p,
    const std::vector<double>& tenors = {0.25,0.5,1,2,3,5,7,10,15,20,30})
{
    std::vector<double> zeros(tenors.size());
    for (size_t i = 0; i < tenors.size(); ++i)
        zeros[i] = cir_zero_rate(tenors[i], p);
    return fixed_income::YieldCurve(tenors, zeros);
}

// ============================================================================
// Hull-White — Analytic Bond Price (relative to initial curve)
// ============================================================================
//
// Given the initial market curve P_mkt(0,T), Hull-White prices bonds as:
//
//   P_HW(t,T) = P_mkt(0,T)/P_mkt(0,t)
//               * exp(-B(tau)*[r(t) - f(0,t)]
//                     - sigma^2/(4*a) * B(tau)^2 * (1 - exp(-2*a*t)))
//
// where B(tau) = (1-exp(-a*tau))/a, f(0,t) = inst. forward from market curve
//
// At t=0: P_HW(0,T) = P_mkt(0,T) * exp(-B(T)*[r0 - f(0,0)] - ...)
// With r0 = f(0,0) (short rate equals the instantaneous forward at t=0),
// P_HW(0,T) = P_mkt(0,T) exactly. This is the curve-fitting guarantee.
// ============================================================================

/// Hull-White B(tau) = (1 - exp(-a*tau)) / a — same structure as Vasicek
inline double hw_B(double tau, double a) {
    if (std::abs(a) < 1e-10) return tau;
    return (1.0 - std::exp(-a * tau)) / a;
}

/// Hull-White theta(t) — the time-dependent drift function
/// theta(t) = dF/dt(0,t) + a*F(0,t) + sigma^2/(2*a)*(1-exp(-2*a*t))
/// F(0,t) = instantaneous forward rate from the initial curve
inline double hw_theta(double t, const HullWhiteParams& p,
                        const fixed_income::YieldCurve& mkt_curve) {
    double f = mkt_curve.inst_forward(t);
    // Numerical derivative of forward rate: dF/dt
    double dt = 0.001;
    double df_dt = (mkt_curve.inst_forward(t + dt) - mkt_curve.inst_forward(t - dt)) / (2.0 * dt);

    return df_dt + p.a * f + p.sigma * p.sigma / (2.0 * p.a)
           * (1.0 - std::exp(-2.0 * p.a * t));
}

// ============================================================================
// Monte Carlo Simulation of Short Rate Paths
// ============================================================================

/// Simulate Vasicek paths — exact transition (Gaussian)
/// Returns paths[path_idx][step_idx], where step 0 = r(0)
inline std::vector<std::vector<double>> simulate_vasicek(
    const VasicekParams& p,
    double T, size_t n_steps, size_t n_paths,
    unsigned long seed = 42)
{
    MersenneTwisterRNG rng(seed);
    double dt = T / n_steps;

    // Exact transition: r(t+dt) ~ N(mu, var) where
    //   mu  = theta + (r_t - theta)*exp(-kappa*dt)
    //   var = sigma^2/(2*kappa) * (1 - exp(-2*kappa*dt))
    double exp_kdt = std::exp(-p.kappa * dt);
    double std_dev = std::sqrt(p.sigma * p.sigma / (2.0 * p.kappa)
                               * (1.0 - std::exp(-2.0 * p.kappa * dt)));

    std::vector<std::vector<double>> paths(n_paths, std::vector<double>(n_steps + 1));

    for (size_t i = 0; i < n_paths; ++i) {
        paths[i][0] = p.r0;
        auto Z = generate_normals(rng, n_steps);

        for (size_t j = 0; j < n_steps; ++j) {
            // Exact Gaussian transition — not Euler, but exact in distribution
            double mu = p.theta + (paths[i][j] - p.theta) * exp_kdt;
            paths[i][j + 1] = mu + std_dev * Z[j];
        }
    }
    return paths;
}

/// Simulate CIR paths — Euler with full truncation + Milstein correction
inline std::vector<std::vector<double>> simulate_cir(
    const CIRParams& p,
    double T, size_t n_steps, size_t n_paths,
    unsigned long seed = 42)
{
    MersenneTwisterRNG rng(seed);
    double dt = T / n_steps;
    double sqrt_dt = std::sqrt(dt);

    std::vector<std::vector<double>> paths(n_paths, std::vector<double>(n_steps + 1));

    for (size_t i = 0; i < n_paths; ++i) {
        paths[i][0] = p.r0;
        auto Z = generate_normals(rng, n_steps);

        for (size_t j = 0; j < n_steps; ++j) {
            double r = paths[i][j];
            // Full truncation: r_pos = max(r, 0)
            double r_pos = std::max(r, 0.0);
            double sqrt_r = std::sqrt(r_pos);

            // Milstein scheme for CIR:
            // dr = kappa*(theta-r_pos)*dt + sigma*sqrt(r_pos)*dW + 0.25*sigma^2*(dW^2-dt)
            // where dW = sqrt(dt)*Z
            double dW = sqrt_dt * Z[j];
            paths[i][j + 1] = r
                + p.kappa * (p.theta - r_pos) * dt
                + p.sigma * sqrt_r * dW
                + 0.25 * p.sigma * p.sigma * (dW * dW - dt);  // Milstein correction
        }
    }
    return paths;
}

/// Simulate Hull-White paths — Euler with time-dependent theta
inline std::vector<std::vector<double>> simulate_hull_white(
    const HullWhiteParams& p,
    const fixed_income::YieldCurve& mkt_curve,
    double T, size_t n_steps, size_t n_paths,
    unsigned long seed = 42)
{
    MersenneTwisterRNG rng(seed);
    double dt = T / n_steps;
    double sqrt_dt = std::sqrt(dt);

    // Initial short rate = instantaneous forward at t=0
    double r0 = mkt_curve.inst_forward(0.001);

    std::vector<std::vector<double>> paths(n_paths, std::vector<double>(n_steps + 1));

    for (size_t i = 0; i < n_paths; ++i) {
        paths[i][0] = r0;
        auto Z = generate_normals(rng, n_steps);

        for (size_t j = 0; j < n_steps; ++j) {
            double t = j * dt;
            double th = hw_theta(t, p, mkt_curve);
            // dr = [theta(t) - a*r]*dt + sigma*dW
            paths[i][j + 1] = paths[i][j]
                + (th - p.a * paths[i][j]) * dt
                + p.sigma * sqrt_dt * Z[j];
        }
    }
    return paths;
}

// ============================================================================
// MC Bond Pricing — price ZCB via simulated short rate paths
// ============================================================================
//
// P(0,T) = E[exp(-integral_0^T r(s) ds)]
//        ≈ mean over paths of exp(-sum r(t_i)*dt)
// ============================================================================

struct MCBondResult {
    double price;
    double std_error;
    double zero_rate;   // Implied zero rate = -ln(P)/T
};

/// Price zero-coupon bond from simulated short rate paths
inline MCBondResult mc_bond_price(
    const std::vector<std::vector<double>>& paths,
    double T)
{
    size_t n_paths = paths.size();
    size_t n_steps = paths[0].size() - 1;
    double dt = T / n_steps;

    double sum = 0.0, sum_sq = 0.0;

    for (size_t i = 0; i < n_paths; ++i) {
        // Trapezoidal rule for integral of r(t)
        double integral = 0.5 * paths[i][0];
        for (size_t j = 1; j < n_steps; ++j)
            integral += paths[i][j];
        integral += 0.5 * paths[i][n_steps];
        integral *= dt;

        double pv = std::exp(-integral);
        sum += pv;
        sum_sq += pv * pv;
    }

    double mean = sum / n_paths;
    double var = (sum_sq / n_paths) - mean * mean;
    double se = std::sqrt(std::max(var, 0.0) / n_paths);
    double zr = (T > 1e-10) ? -std::log(mean) / T : 0.0;

    return {mean, se, zr};
}

// ============================================================================
// Vasicek Bond Option — Jamshidian (1989)
// ============================================================================
//
// European call on a zero-coupon bond P(T,S) with strike X, exercised at T:
//   C = P(0,S)*N(d1) - X*P(0,T)*N(d2)
//
// where:
//   sigma_P = sigma/kappa * (1-exp(-kappa*(S-T))) * sqrt((1-exp(-2*kappa*T))/(2*kappa))
//   d1 = ln(P(0,S)/(X*P(0,T))) / sigma_P + sigma_P/2
//   d2 = d1 - sigma_P
//
// This is Black's formula with the bond forward price as the underlying.
// ============================================================================

inline double vasicek_bond_call(
    double T,       // Option expiry
    double S,       // Bond maturity (S > T)
    double X,       // Strike price
    const VasicekParams& p)
{
    double P_T = vasicek_bond_price(T, p);
    double P_S = vasicek_bond_price(S, p);

    // Bond price volatility
    double B_ST = vasicek_B(S - T, p.kappa);
    double sigma_P = p.sigma * B_ST
                   * std::sqrt((1.0 - std::exp(-2.0 * p.kappa * T)) / (2.0 * p.kappa));

    if (sigma_P < 1e-15) return std::max(P_S - X * P_T, 0.0);

    // Black's formula on bond forward
    double d1 = std::log(P_S / (X * P_T)) / sigma_P + sigma_P / 2.0;
    double d2 = d1 - sigma_P;

    // N(x) = standard normal CDF
    auto N = [](double x) { return 0.5 * std::erfc(-x * M_SQRT1_2); };

    return P_S * N(d1) - X * P_T * N(d2);
}

/// Put via put-call parity: Put = Call - P(0,S) + X*P(0,T)
inline double vasicek_bond_put(
    double T, double S, double X,
    const VasicekParams& p)
{
    double call = vasicek_bond_call(T, S, X, p);
    double P_T = vasicek_bond_price(T, p);
    double P_S = vasicek_bond_price(S, p);
    return call - P_S + X * P_T;
}

} // namespace rates

#endif // QUANTPRICER_RATE_MODELS_H

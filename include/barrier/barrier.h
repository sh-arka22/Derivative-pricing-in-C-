#ifndef QUANTPRICER_BARRIER_H
#define QUANTPRICER_BARRIER_H

// ============================================================================
// Barrier Options — Day 10
//
// A barrier option's payoff depends on whether the underlying price S(t)
// crosses a predetermined barrier level H during the option's life.
//
// ============================================================================
// MATHEMATICS
// ============================================================================
//
// Barrier option types (4 barriers x 2 option types = 8 combinations):
//
//   Knock-Out: option dies worthless if barrier is breached
//     Down-and-Out (DO): extinguished if S falls to or below H
//     Up-and-Out   (UO): extinguished if S rises to or above H
//
//   Knock-In: option only activates if barrier is breached
//     Down-and-In  (DI): activated when S falls to or below H
//     Up-and-In    (UI): activated when S rises to or above H
//
// ============================================================================
// IN-OUT PARITY (key identity for validation and interview):
//
//   V_in(S,K,H) + V_out(S,K,H) = V_vanilla(S,K)    [when rebate = 0]
//
//   Every path either hits H or doesn't. If it hits: "in" pays, "out" doesn't.
//   If it doesn't hit: "out" pays, "in" doesn't. Together they cover all paths.
//   This means you only need to price one; the other is free.
//
// ============================================================================
// ANALYTIC FORMULAS — Reiner & Rubinstein (1991), Haug (2007)
//
// Under Black-Scholes (continuous monitoring), the price decomposes into
// building blocks A, B, C, D, E, F using the reflection principle of
// Brownian motion. The key insight: reflecting a BM path at the barrier
// produces another valid BM path with a known Radon-Nikodym derivative,
// which is (H/S)^(2*mu) where mu = (r - sigma^2/2) / sigma^2.
//
// ============================================================================
// CONTINUOUS vs DISCRETE MONITORING
//
// Analytic formulas assume the barrier is checked at every instant (continuous).
// MC naturally checks the barrier at each time step (discrete).
//
// Broadie-Glasserman-Kou (1997) correction bridges this gap:
//   H_adjusted = H * exp(+/- beta * sigma * sqrt(dt))
//   where beta = -zeta(1/2) / sqrt(2*pi) ~ 0.5826
//
// Intuition: with discrete monitoring, the asset can "sneak past" the barrier
// between observation dates. The correction tightens the barrier to compensate.
//
// Key interview insight:
//   Barrier options are CHEAPER than vanilla (knock-out removes payoff scenarios).
//   A knock-out option's value approaches zero as S approaches the barrier.
//   Delta of a knock-out can be discontinuous at the barrier — hedging nightmare.
//   This is why barrier options are popular for structured products (cheaper)
//   but challenging for risk management (discontinuous Greeks).
// ============================================================================

#include "greeks/black_scholes.h"
#include "mc/monte_carlo.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace barrier {

// ============================================================================
// Barrier Option Types & Parameters
// ============================================================================

enum class BarrierType {
    DownAndOut,   // Knocked out when S <= H (cheaper than vanilla)
    DownAndIn,    // Activated when S <= H
    UpAndOut,     // Knocked out when S >= H (cheaper than vanilla)
    UpAndIn       // Activated when S >= H
};

enum class OptionType {
    Call,   // payoff = max(S - K, 0)
    Put     // payoff = max(K - S, 0)
};

struct BarrierParams {
    double S0;          // Spot price
    double K;           // Strike price
    double H;           // Barrier level
    double r;           // Risk-free rate
    double T;           // Time to maturity
    double sigma;       // Volatility
    double rebate;      // Cash rebate paid on knock-out (default 0)
    BarrierType type;
    OptionType opt_type;

    BarrierParams()
        : S0(100), K(100), H(90), r(0.05), T(1.0), sigma(0.2),
          rebate(0.0), type(BarrierType::DownAndOut), opt_type(OptionType::Call) {}

    BarrierParams(double s0, double k, double h, double rate, double mat,
                  double vol, BarrierType bt, OptionType ot, double reb = 0.0)
        : S0(s0), K(k), H(h), r(rate), T(mat), sigma(vol),
          rebate(reb), type(bt), opt_type(ot) {}

    bool is_down() const {
        return type == BarrierType::DownAndOut || type == BarrierType::DownAndIn;
    }
    bool is_out() const {
        return type == BarrierType::DownAndOut || type == BarrierType::UpAndOut;
    }
    bool is_call() const { return opt_type == OptionType::Call; }
};

/// Result struct for barrier MC pricing
struct BarrierMCResult {
    double price;
    double std_error;
    double confidence_lo;
    double confidence_hi;
    size_t num_paths;
    double knock_pct;       // Percentage of paths that hit the barrier
};

// ============================================================================
// Analytic Pricing — Haug (2007) Building Blocks
// ============================================================================
//
// Derived quantities from (S, K, H, r, T, sigma):
//
//   mu     = (r - sigma^2/2) / sigma^2       — drift adjustment for reflection
//   lambda = sqrt(mu^2 + 2*r/sigma^2)        — used in rebate term F
//   sigma_sqrtT = sigma * sqrt(T)
//
//   x1 = ln(S/K) / sigma_sqrtT + (1 + mu) * sigma_sqrtT
//   x2 = ln(S/H) / sigma_sqrtT + (1 + mu) * sigma_sqrtT
//   y1 = ln(H^2/(S*K)) / sigma_sqrtT + (1 + mu) * sigma_sqrtT
//   y2 = ln(H/S) / sigma_sqrtT + (1 + mu) * sigma_sqrtT
//   z  = ln(H/S) / sigma_sqrtT + lambda * sigma_sqrtT
//
// Building blocks (phi = +1 call, -1 put; eta = +1 down, -1 up):
//
//   A = phi*S*N(phi*x1) - phi*K*e^{-rT}*N(phi*(x1 - sigma_sqrtT))
//       → this is just the vanilla BS price
//
//   B = phi*S*N(phi*x2) - phi*K*e^{-rT}*N(phi*(x2 - sigma_sqrtT))
//       → BS price with H replacing K in the log term
//
//   C = phi*S*(H/S)^{2(mu+1)}*N(eta*y1) - phi*K*e^{-rT}*(H/S)^{2mu}*N(eta*(y1 - sigma_sqrtT))
//       → "reflected" vanilla term (reflection principle)
//
//   D = phi*S*(H/S)^{2(mu+1)}*N(eta*y2) - phi*K*e^{-rT}*(H/S)^{2mu}*N(eta*(y2 - sigma_sqrtT))
//       → "reflected" B term
//
//   E = rebate*e^{-rT}*[N(eta*(x2-sigma_sqrtT)) - (H/S)^{2mu}*N(eta*(y2-sigma_sqrtT))]
//       → knock-in rebate (paid at expiry)
//
//   F = rebate*[(H/S)^{mu+lambda}*N(eta*z) + (H/S)^{mu-lambda}*N(eta*(z-2*lambda*sigma_sqrtT))]
//       → knock-out rebate (paid at hit time)
//
// The 8 barrier prices are assembled from these blocks.
// ============================================================================

namespace detail {

/// Compute all Haug building blocks A through F
struct HaugTerms {
    double A, B, C, D, E, F;
};

inline HaugTerms compute_terms(
    double S, double K, double H, double r, double T, double sigma,
    double rebate, double phi, double eta)
{
    double sigma_sqrtT = sigma * std::sqrt(T);
    double disc = std::exp(-r * T);

    // mu = (r - sigma^2/2) / sigma^2 — drift param for reflection principle
    double mu = (r - 0.5 * sigma * sigma) / (sigma * sigma);

    // lambda = sqrt(mu^2 + 2r/sigma^2) — appears in rebate term F
    double lam = std::sqrt(mu * mu + 2.0 * r / (sigma * sigma));

    // Log-moneyness and barrier-distance terms
    double x1 = std::log(S / K) / sigma_sqrtT + (1.0 + mu) * sigma_sqrtT;
    double x2 = std::log(S / H) / sigma_sqrtT + (1.0 + mu) * sigma_sqrtT;
    double y1 = std::log(H * H / (S * K)) / sigma_sqrtT + (1.0 + mu) * sigma_sqrtT;
    double y2 = std::log(H / S) / sigma_sqrtT + (1.0 + mu) * sigma_sqrtT;
    double z  = std::log(H / S) / sigma_sqrtT + lam * sigma_sqrtT;

    // (H/S) ratio powers — from the reflection principle Radon-Nikodym derivative
    double hs = H / S;
    double hs_2mu   = std::pow(hs, 2.0 * mu);
    double hs_2mu2  = std::pow(hs, 2.0 * (mu + 1.0));
    double hs_ml    = std::pow(hs, mu + lam);
    double hs_ml_n  = std::pow(hs, mu - lam);

    // A = vanilla BS price (phi selects call/put)
    double A_val = phi * S * bs::N(phi * x1)
                 - phi * K * disc * bs::N(phi * (x1 - sigma_sqrtT));

    // B = BS-like term with barrier replacing strike in log
    double B_val = phi * S * bs::N(phi * x2)
                 - phi * K * disc * bs::N(phi * (x2 - sigma_sqrtT));

    // C = reflected vanilla (reflection principle: paths that hit H)
    double C_val = phi * S * hs_2mu2 * bs::N(eta * y1)
                 - phi * K * disc * hs_2mu * bs::N(eta * (y1 - sigma_sqrtT));

    // D = reflected B term
    double D_val = phi * S * hs_2mu2 * bs::N(eta * y2)
                 - phi * K * disc * hs_2mu * bs::N(eta * (y2 - sigma_sqrtT));

    // E = knock-in rebate (paid at expiry if barrier hit)
    double E_val = rebate * disc
                 * (bs::N(eta * (x2 - sigma_sqrtT))
                    - hs_2mu * bs::N(eta * (y2 - sigma_sqrtT)));

    // F = knock-out rebate (paid at first barrier hit time)
    double F_val = rebate
                 * (hs_ml * bs::N(eta * z)
                    + hs_ml_n * bs::N(eta * (z - 2.0 * lam * sigma_sqrtT)));

    return {A_val, B_val, C_val, D_val, E_val, F_val};
}

} // namespace detail

// ============================================================================
// Analytic Barrier Price — Haug (2007) Table 8-1
// ============================================================================
//
// The 8 barrier types are assembled from building blocks A–F.
// The formula depends on whether K >= H or K < H.
//
// Dispatch table (phi = +1 call, -1 put; eta = +1 down, -1 up):
//
//  Type               | K >= H                | K < H
//  -------------------|-----------------------|----------------------
//  Down-and-In  Call  | C + E                 | A - B + D + E
//  Down-and-Out Call  | A - C + F             | B - D + F
//  Up-and-In    Call  | A + E                 | B - C + D + E
//  Up-and-Out   Call  | F                     | A - B + C - D + F
//  Down-and-In  Put   | B - C + D + E         | A + E
//  Down-and-Out Put   | A - B + C - D + F     | F
//  Up-and-In    Put   | A - B + D + E         | C + E
//  Up-and-Out   Put   | B - D + F             | A - C + F
// ============================================================================

inline double barrier_price(const BarrierParams& p) {
    if (p.T <= 0.0) throw std::invalid_argument("T must be positive");
    if (p.sigma <= 0.0) throw std::invalid_argument("sigma must be positive");

    double S = p.S0, K = p.K, H = p.H;
    bool is_down = p.is_down();
    bool is_out  = p.is_out();
    bool is_call = p.is_call();

    // Check if barrier already breached
    if (is_down && S <= H) {
        // Down barrier already breached
        if (is_out) return p.rebate * std::exp(-p.r * p.T);  // Knocked out
        // Knocked in → vanilla
        return is_call ? bs::call_price(S, K, p.r, p.T, p.sigma)
                       : bs::put_price(S, K, p.r, p.T, p.sigma);
    }
    if (!is_down && S >= H) {
        // Up barrier already breached
        if (is_out) return p.rebate * std::exp(-p.r * p.T);  // Knocked out
        // Knocked in → vanilla
        return is_call ? bs::call_price(S, K, p.r, p.T, p.sigma)
                       : bs::put_price(S, K, p.r, p.T, p.sigma);
    }

    // phi = +1 for call, -1 for put
    // eta = +1 for down barrier, -1 for up barrier
    double phi = is_call ? 1.0 : -1.0;
    double eta = is_down ? 1.0 : -1.0;

    auto t = detail::compute_terms(S, K, H, p.r, p.T, p.sigma, p.rebate, phi, eta);

    // Dispatch based on barrier type, option type, and K vs H
    // Reference: Haug (2007) "The Complete Guide to Option Pricing Formulas" Table 8-1
    switch (p.type) {
        case BarrierType::DownAndIn:
            if (is_call)
                return (K >= H) ? t.C + t.E : t.A - t.B + t.D + t.E;
            else
                return (K >= H) ? t.B - t.C + t.D + t.E : t.A + t.E;

        case BarrierType::DownAndOut:
            if (is_call)
                return (K >= H) ? t.A - t.C + t.F : t.B - t.D + t.F;
            else
                return (K >= H) ? t.A - t.B + t.C - t.D + t.F : t.F;

        case BarrierType::UpAndIn:
            if (is_call)
                return (K >= H) ? t.A + t.E : t.B - t.C + t.D + t.E;
            else
                return (K >= H) ? t.A - t.B + t.D + t.E : t.C + t.E;

        case BarrierType::UpAndOut:
            if (is_call)
                return (K >= H) ? t.F : t.A - t.B + t.C - t.D + t.F;
            else
                return (K >= H) ? t.B - t.D + t.F : t.A - t.C + t.F;
    }
    return 0.0;  // unreachable
}

// ============================================================================
// In-Out Parity Check — V_in + V_out = V_vanilla (when rebate = 0)
// ============================================================================

inline double in_out_parity_error(const BarrierParams& p) {
    BarrierParams p_in = p, p_out = p;
    p_in.rebate = 0.0;
    p_out.rebate = 0.0;

    if (p.is_out()) {
        p_in.type = p.is_down() ? BarrierType::DownAndIn : BarrierType::UpAndIn;
        p_out.type = p.type;
    } else {
        p_out.type = p.is_down() ? BarrierType::DownAndOut : BarrierType::UpAndOut;
        p_in.type = p.type;
    }

    double v_in  = barrier_price(p_in);
    double v_out = barrier_price(p_out);
    double vanilla = p.is_call()
        ? bs::call_price(p.S0, p.K, p.r, p.T, p.sigma)
        : bs::put_price(p.S0, p.K, p.r, p.T, p.sigma);

    return std::abs(v_in + v_out - vanilla);
}

// ============================================================================
// Monte Carlo Barrier Pricing — Discrete Path Monitoring
// ============================================================================
//
// Algorithm:
//   1. Generate GBM path with n_steps discrete time points
//   2. At each point, check if barrier has been breached
//   3. Knock-out: if breached, payoff = 0 (or rebate)
//      Knock-in:  if NOT breached, payoff = 0
//   4. Average discounted payoffs across all paths
//
// Note: discrete monitoring OVERPRICES knock-out options (misses some
// barrier hits between steps) and UNDERPRICES knock-in options.
// The Broadie-Glasserman-Kou correction fixes this bias.
// ============================================================================

inline BarrierMCResult mc_barrier(
    const BarrierParams& p,
    size_t num_paths, size_t n_steps = 252,
    unsigned long seed = 42)
{
    MersenneTwisterRNG rng(seed);
    double discount = std::exp(-p.r * p.T);

    bool is_down = p.is_down();
    bool is_out  = p.is_out();
    bool is_call = p.is_call();

    double sum = 0.0, sum_sq = 0.0;
    size_t knock_count = 0;

    for (size_t i = 0; i < num_paths; ++i) {
        // Generate GBM path: path[0] = S0, path[n_steps] = S_T
        auto path = generate_gbm_path(p.S0, p.r, p.sigma, p.T, n_steps, rng);

        // Check if barrier was breached at any monitoring point
        // For down: breached if any S(t_i) <= H
        // For up:   breached if any S(t_i) >= H
        bool breached = false;
        for (size_t j = 1; j <= n_steps; ++j) {
            if ((is_down && path[j] <= p.H) || (!is_down && path[j] >= p.H)) {
                breached = true;
                break;
            }
        }

        if (breached) ++knock_count;

        // Determine payoff based on barrier type
        double terminal = path[n_steps];
        double intrinsic = is_call ? std::max(terminal - p.K, 0.0)
                                   : std::max(p.K - terminal, 0.0);
        double pv;

        if (is_out) {
            // Knock-out: pay intrinsic only if barrier NOT breached
            pv = breached ? discount * p.rebate : discount * intrinsic;
        } else {
            // Knock-in: pay intrinsic only if barrier WAS breached
            pv = breached ? discount * intrinsic : 0.0;
        }

        sum += pv;
        sum_sq += pv * pv;
    }

    double mean = sum / num_paths;
    double var = (sum_sq / num_paths) - mean * mean;
    double se = std::sqrt(std::max(var, 0.0) / num_paths);
    double knock_pct = 100.0 * knock_count / num_paths;

    return {mean, se, mean - 1.96 * se, mean + 1.96 * se, num_paths, knock_pct};
}

// ============================================================================
// Broadie-Glasserman-Kou (1997) Continuity Correction
// ============================================================================
//
// Problem: MC with m monitoring dates overprices knock-out options because
// the asset can "sneak past" the barrier between observation dates.
//
// Solution: shift the barrier to account for unobserved crossings.
//
// Math: For m equally-spaced monitoring dates with dt = T/m:
//
//   Up barrier:   H_adj = H * exp( beta * sigma * sqrt(dt) )
//   Down barrier: H_adj = H * exp(-beta * sigma * sqrt(dt) )
//
// where beta = -zeta(1/2) / sqrt(2*pi) ~ 0.5826
//       zeta(1/2) ~ -1.4603545... (Riemann zeta at 1/2)
//
// Intuition: the correction TIGHTENS the barrier (moves it closer to S0),
// making knock-out MORE likely, compensating for missed crossings.
//
// Usage: replace H with H_adj in the ANALYTIC formula to approximate
// the discrete-monitoring price without running MC.
//
// Key interview insight:
//   The BGK correction is O(1/sqrt(m)) accurate — the discrete barrier
//   price converges to the continuous price at rate 1/sqrt(m), and the
//   correction captures the leading term exactly.
// ============================================================================

/// Broadie-Glasserman-Kou beta constant
/// beta = -zeta(1/2) / sqrt(2*pi) where zeta(1/2) ~ -1.4603545
inline constexpr double BGK_BETA = 0.5826;

/// Adjust barrier for discrete monitoring via BGK correction
inline double bgk_adjusted_barrier(double H, double sigma, double T,
                                    size_t n_steps, bool is_down) {
    double dt = T / n_steps;
    if (is_down) {
        // Down barrier: shift DOWN (tighten = make knock-out more likely)
        return H * std::exp(-BGK_BETA * sigma * std::sqrt(dt));
    } else {
        // Up barrier: shift UP (tighten = make knock-out more likely)
        return H * std::exp(BGK_BETA * sigma * std::sqrt(dt));
    }
}

/// Price with BGK correction: use analytic formula with adjusted barrier
/// This approximates the DISCRETE monitoring price without running MC
inline double barrier_price_bgk(const BarrierParams& p, size_t n_steps) {
    double H_adj = bgk_adjusted_barrier(p.H, p.sigma, p.T, n_steps, p.is_down());
    BarrierParams p_adj = p;
    p_adj.H = H_adj;
    return barrier_price(p_adj);
}

} // namespace barrier

#endif // QUANTPRICER_BARRIER_H

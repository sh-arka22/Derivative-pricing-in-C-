#ifndef QUANTPRICER_OPTION_H
#define QUANTPRICER_OPTION_H

// ============================================================================
// Option Classes — Book Chapters 3, 10, 12, 16, 17
// Ch 3:  VanillaOption with K, r, T, S, sigma — OOP design, const correctness
// Ch 10: European option specification for Monte Carlo
// Ch 12: Asian option with path-dependent averaging
// Ch 16: Heston option parameters (vol-of-vol, correlation, etc.)
// Ch 17: FDM option parameters (mesh boundaries)
// ============================================================================

#include "payoff/payoff.h"
#include <memory>
#include <string>

/// Core European option parameters — Ch 3.5
struct VanillaOption {
    double K;       // Strike price
    double r;       // Risk-free rate
    double T;       // Time to maturity (years)
    double S;       // Spot price
    double sigma;   // Volatility

    VanillaOption() : K(100.0), r(0.05), T(1.0), S(100.0), sigma(0.2) {}  // Ch 3.7 defaults

    VanillaOption(double strike, double rate, double maturity,
                  double spot, double vol)
        : K(strike), r(rate), T(maturity), S(spot), sigma(vol) {}
};

/// Option wrapper with polymorphic payoff — connects Ch 3 and Ch 4
class Option {
public:
    Option(std::unique_ptr<PayOff> payoff, double r, double T, double S, double sigma)
        : payoff_(std::move(payoff)), r_(r), T_(T), S_(S), sigma_(sigma) {}

    Option(const Option& other)
        : payoff_(other.payoff_->clone()), r_(other.r_), T_(other.T_),
          S_(other.S_), sigma_(other.sigma_) {}

    Option& operator=(const Option& other) {
        if (this != &other) {
            payoff_ = other.payoff_->clone();
            r_ = other.r_;  T_ = other.T_;
            S_ = other.S_;  sigma_ = other.sigma_;
        }
        return *this;
    }

    double payoff(double spot) const { return (*payoff_)(spot); }
    const PayOff& get_payoff() const { return *payoff_; }

    double r() const     { return r_; }
    double T() const     { return T_; }
    double S() const     { return S_; }
    double sigma() const { return sigma_; }

private:
    std::unique_ptr<PayOff> payoff_;
    double r_;
    double T_;
    double S_;
    double sigma_;
};

/// Heston model parameters — Ch 16.5
struct HestonParams {
    double S0;          // Initial spot
    double K;           // Strike
    double r;           // Risk-free rate
    double T;           // Maturity
    double v0;          // Initial variance
    double kappa;       // Mean reversion speed of variance
    double theta;       // Long-run variance
    double xi;          // Vol-of-vol (σ_v)
    double rho;         // Correlation between asset and variance Brownians

    HestonParams()
        : S0(100.0), K(100.0), r(0.05), T(1.0),
          v0(0.04), kappa(2.0), theta(0.04), xi(0.3), rho(-0.7) {}

    HestonParams(double s0, double k, double rate, double mat,
                 double var0, double kap, double th, double volofvol, double corr)
        : S0(s0), K(k), r(rate), T(mat),
          v0(var0), kappa(kap), theta(th), xi(volofvol), rho(corr) {}

    /// Feller condition: 2*kappa*theta > xi^2 ensures variance stays positive
    bool feller_satisfied() const {
        return 2.0 * kappa * theta > xi * xi;
    }
};

/// Merton jump-diffusion parameters — Ch 15.1
struct MertonJumpParams {
    double S0;          // Initial spot
    double K;           // Strike
    double r;           // Risk-free rate
    double T;           // Maturity
    double sigma;       // Diffusion volatility
    double lambda;      // Jump intensity (expected jumps per year)
    double mu_j;        // Mean of log-jump size
    double sigma_j;     // Std dev of log-jump size

    MertonJumpParams()
        : S0(100.0), K(100.0), r(0.05), T(1.0),
          sigma(0.2), lambda(1.0), mu_j(-0.1), sigma_j(0.15) {}

    MertonJumpParams(double s0, double k, double rate, double mat,
                     double vol, double lam, double muj, double sigj)
        : S0(s0), K(k), r(rate), T(mat),
          sigma(vol), lambda(lam), mu_j(muj), sigma_j(sigj) {}
};

/// Bates model parameters — Heston stochastic vol + Merton jumps (Bates, 1996)
/// dS = (r - λk)S dt + √v S dW_S + J S dN
/// dv = κ(θ - v)dt + ξ√v dW_v,  Corr(dW_S, dW_v) = ρ
struct BatesParams {
    // Heston parameters
    double S0;          // Initial spot
    double K;           // Strike
    double r;           // Risk-free rate
    double T;           // Maturity
    double v0;          // Initial variance
    double kappa;       // Mean reversion speed of variance
    double theta;       // Long-run variance
    double xi;          // Vol-of-vol (σ_v)
    double rho;         // Correlation between asset and variance Brownians

    // Merton jump parameters
    double lambda;      // Jump intensity (expected jumps per year)
    double mu_j;        // Mean of log-jump size
    double sigma_j;     // Std dev of log-jump size

    BatesParams()
        : S0(100.0), K(100.0), r(0.05), T(1.0),
          v0(0.04), kappa(2.0), theta(0.04), xi(0.3), rho(-0.7),
          lambda(1.0), mu_j(-0.05), sigma_j(0.1) {}

    BatesParams(double s0, double k, double rate, double mat,
                double var0, double kap, double th, double volofvol, double corr,
                double lam, double muj, double sigj)
        : S0(s0), K(k), r(rate), T(mat),
          v0(var0), kappa(kap), theta(th), xi(volofvol), rho(corr),
          lambda(lam), mu_j(muj), sigma_j(sigj) {}

    /// Feller condition: 2*kappa*theta > xi^2 ensures variance stays positive
    bool feller_satisfied() const {
        return 2.0 * kappa * theta > xi * xi;
    }
};

/// FDM mesh parameters — Ch 17.2
struct FDMParams {
    size_t N_space;     // Number of spatial grid points
    size_t N_time;      // Number of time steps
    double S_max;       // Upper boundary for spot price
    double theta_fdm;   // 0.0 = explicit, 0.5 = Crank-Nicolson, 1.0 = implicit

    FDMParams()
        : N_space(200), N_time(1000), S_max(300.0), theta_fdm(0.5) {}

    FDMParams(size_t ns, size_t nt, double smax, double th)
        : N_space(ns), N_time(nt), S_max(smax), theta_fdm(th) {}
};

#endif // QUANTPRICER_OPTION_H

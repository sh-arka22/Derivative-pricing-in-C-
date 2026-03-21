#ifndef QUANTPRICER_FDM_H
#define QUANTPRICER_FDM_H

// ============================================================================
// Finite Difference Method (FDM) Solver — Book Chapter 17
// Ch 17.1: Black-Scholes PDE for European call:
//          ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
// Ch 17.2: Finite difference discretisation (explicit, implicit, C-N)
// Ch 17.3: Full implementation — PayOff, VanillaOption, PDE, FDM classes
// Ch 17.4: Execution and output
// Uses: Ch 9.3 Thomas algorithm for tridiagonal system solve
//       Ch 4 PayOff hierarchy for terminal condition
//       Ch 5 Template classes for matrix storage
// ============================================================================

#include "payoff/payoff.h"
#include "option/option.h"
#include "matrix/matrix.h"  // For Thomas algorithm
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>

/// PDE coefficient interface — Ch 17.3.3
/// For Black-Scholes: a(S) = ½σ²S², b(S) = rS, c(S) = -r
class PDECoefficients {
public:
    virtual ~PDECoefficients() = default;
    virtual double diffusion(double S, double t) const = 0;   // σ²S²/2
    virtual double convection(double S, double t) const = 0;  // rS
    virtual double reaction(double S, double t) const = 0;    // -r
    virtual double source(double S, double t) const = 0;      // 0 for BS
};

/// Black-Scholes PDE coefficients — Ch 17.3.3
class BlackScholesPDE : public PDECoefficients {
public:
    BlackScholesPDE(double r, double sigma) : r_(r), sigma_(sigma) {}

    double diffusion(double S, double /*t*/) const override {
        return 0.5 * sigma_ * sigma_ * S * S;
    }
    double convection(double S, double /*t*/) const override {
        return r_ * S;
    }
    double reaction(double /*S*/, double /*t*/) const override {
        return -r_;
    }
    double source(double /*S*/, double /*t*/) const override {
        return 0.0;
    }

    /// Accessor for rate — needed by FDM solver for boundary conditions
    double rate() const { return r_; }

private:
    double r_, sigma_;
};

/// FDM result container
struct FDMResult {
    std::vector<double> spot_grid;     // S values
    std::vector<double> option_values; // V(S, 0) at t=0
    double price_at_spot;              // Interpolated price at S0
};

/// Full FDM solver for Black-Scholes PDE — Ch 17.3.4
/// Supports: theta=0.0 (explicit), theta=0.5 (Crank-Nicolson), theta=1.0 (implicit)
class FDMSolver {
public:
    FDMSolver(const PayOff& payoff, const BlackScholesPDE& pde,
              double S0, double T, const FDMParams& params)
        : payoff_(payoff), pde_(pde), S0_(S0), T_(T), params_(params)
    {
        setup_grid();
    }

    /// Solve the PDE and return the option values at t=0
    FDMResult solve() {
        // Terminal condition: V(S, T) = payoff(S) — Ch 17.3.1
        std::vector<double> V(N_ + 1);
        for (size_t i = 0; i <= N_; ++i) {
            V[i] = payoff_(S_grid_[i]);
        }

        double dt = T_ / M_;
        double dS = params_.S_max / N_;
        double theta = params_.theta_fdm;
        double r = pde_.rate();

        // Detect option type from payoff to set correct boundary conditions
        // Put-like: payoff(0) > 0, strike K = payoff(0)
        // Call-like: payoff(0) == 0, strike K = S_max - payoff(S_max)
        // Ref: Duffy Ch 15.3-15.5, Wilmott Ch 77-78
        double payoff_at_zero = payoff_(0.0);
        bool is_put_like = (payoff_at_zero > 0.0);
        double K = is_put_like ? payoff_at_zero
                               : (S_grid_[N_] - payoff_(S_grid_[N_]));

        // Time-stepping: march backwards from T to 0 — Ch 17.2
        for (size_t m = 0; m < M_; ++m) {
            double t = T_ - m * dt;

            // Time-dependent boundary values at the NEW time level — Ch 17.3.5
            // tau_new = time remaining to maturity after this step
            double tau_new = (m + 1) * dt;
            double V_lower_new, V_upper_new;
            if (is_put_like) {
                V_lower_new = K * std::exp(-r * tau_new);  // V(0,t) = K*e^{-r*tau}
                V_upper_new = 0.0;                          // V(S_max,t) = 0
            } else {
                V_lower_new = 0.0;                          // V(0,t) = 0
                V_upper_new = S_grid_[N_] - K * std::exp(-r * tau_new); // V(S_max,t)
            }

            // Build tridiagonal coefficients — Ch 17.3.4
            std::vector<double> sub(N_ - 2);    // a_i (below diagonal)
            std::vector<double> main_d(N_ - 1); // b_i (main diagonal)
            std::vector<double> sup(N_ - 2);    // c_i (above diagonal)
            std::vector<double> rhs(N_ - 1);

            for (size_t i = 1; i < N_; ++i) {
                double S = S_grid_[i];
                double diff = pde_.diffusion(S, t);
                double conv = pde_.convection(S, t);
                double react = pde_.reaction(S, t);

                // FDM coefficients for central differences — Ch 17.2
                double alpha = dt * (diff / (dS * dS) - conv / (2.0 * dS));
                double beta  = dt * (-2.0 * diff / (dS * dS) + react);
                double gamma_coeff = dt * (diff / (dS * dS) + conv / (2.0 * dS));

                size_t idx = i - 1;  // 0-indexed for tridiagonal system

                // Theta-scheme: combines explicit and implicit — Ch 17.2
                // (I - θΔtA) V^{n} = (I + (1-θ)ΔtA) V^{n+1}
                main_d[idx] = 1.0 - theta * beta;

                if (idx > 0)
                    sub[idx - 1] = -theta * alpha;
                if (idx < N_ - 2)
                    sup[idx] = -theta * gamma_coeff;

                // Right-hand side: explicit part
                rhs[idx] = (1.0 - theta) * alpha * V[i - 1]
                         + (1.0 + (1.0 - theta) * beta) * V[i]
                         + (1.0 - theta) * gamma_coeff * V[i + 1];
            }

            // Boundary corrections — single theta factor (Duffy Ch 19.5)
            // Lower boundary: implicit contribution from node i=1
            double alpha_1 = dt * (pde_.diffusion(S_grid_[1], t) / (dS * dS)
                                  - pde_.convection(S_grid_[1], t) / (2.0 * dS));
            rhs[0] += theta * alpha_1 * V_lower_new;

            // Upper boundary: implicit contribution from node i=N-1
            double gamma_N1 = dt * (pde_.diffusion(S_grid_[N_-1], t) / (dS * dS)
                                   + pde_.convection(S_grid_[N_-1], t) / (2.0 * dS));
            rhs[N_ - 2] += theta * gamma_N1 * V_upper_new;

            // Solve tridiagonal system using Thomas algorithm — Ch 9.3
            if (theta > 1e-10) {
                auto solution = solve_thomas(sub, main_d, sup, rhs);
                for (size_t i = 0; i < N_ - 1; ++i) {
                    V[i + 1] = solution[i];
                }
            } else {
                // Pure explicit: just copy RHS
                for (size_t i = 0; i < N_ - 1; ++i) {
                    V[i + 1] = rhs[i];
                }
            }

            // Re-apply time-dependent boundary conditions
            V[0] = V_lower_new;
            V[N_] = V_upper_new;
        }

        // Interpolate to find price at S0
        double price = interpolate(V);

        return {S_grid_, V, price};
    }

    /// Write the full grid to CSV for visualization
    void write_csv(const std::string& filename, const FDMResult& result) const {
        std::ofstream ofs(filename);
        ofs << "Spot,OptionValue\n";
        for (size_t i = 0; i <= N_; ++i) {
            ofs << result.spot_grid[i] << "," << result.option_values[i] << "\n";
        }
    }

private:
    const PayOff& payoff_;
    const BlackScholesPDE& pde_;
    double S0_, T_;
    FDMParams params_;
    size_t N_, M_;
    std::vector<double> S_grid_;

    void setup_grid() {
        N_ = params_.N_space;
        M_ = params_.N_time;
        double dS = params_.S_max / N_;
        S_grid_.resize(N_ + 1);
        for (size_t i = 0; i <= N_; ++i) {
            S_grid_[i] = i * dS;
        }
    }

    /// Linear interpolation to find V at S0
    double interpolate(const std::vector<double>& V) const {
        double dS = params_.S_max / N_;
        size_t idx = static_cast<size_t>(S0_ / dS);
        if (idx >= N_) return V[N_];
        double frac = (S0_ - S_grid_[idx]) / dS;
        return V[idx] * (1.0 - frac) + V[idx + 1] * frac;
    }
};

#endif // QUANTPRICER_FDM_H

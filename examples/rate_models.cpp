// ============================================================================
// Day 14 Example: Interest Rate Models — Vasicek, CIR & Hull-White
//
// Demonstrates:
// 1. Vasicek yield curve and MC vs analytic bond pricing
// 2. CIR yield curve with Feller condition analysis
// 3. Hull-White calibration to market curve
// 4. Model comparison: Vasicek vs CIR yield curves
// 5. Bond option pricing (Vasicek analytic)
// 6. Negative rate probability (Vasicek)
// ============================================================================

#include "rates/rate_models.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace rates;

int main() {
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "======================================================================\n"
              << "  Day 14: Interest Rate Models — Vasicek, CIR & Hull-White\n"
              << "======================================================================\n\n";

    // ================================================================
    // 1. Vasicek Model — Analytic vs MC Bond Prices
    // ================================================================
    std::cout << "=== 1. Vasicek: Analytic vs MC Bond Prices ===\n\n";

    VasicekParams vp(0.05, 0.5, 0.05, 0.015);
    std::cout << "Vasicek: r0=" << vp.r0 << " kappa=" << vp.kappa
              << " theta=" << vp.theta << " sigma=" << vp.sigma << "\n\n";

    auto vas_paths = simulate_vasicek(vp, 30.0, 360, 100000);

    std::cout << std::setw(10) << "Maturity"
              << std::setw(14) << "Analytic P"
              << std::setw(14) << "MC P"
              << std::setw(10) << "MC SE"
              << std::setw(14) << "Analytic r"
              << std::setw(14) << "MC r" << "\n";
    std::cout << std::string(76, '-') << "\n";

    for (double T : {1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0}) {
        double anal_P = vasicek_bond_price(T, vp);
        double anal_r = vasicek_zero_rate(T, vp);

        // MC: subset paths to maturity T
        size_t steps_T = static_cast<size_t>(T * 12);
        std::vector<std::vector<double>> sub_paths(vas_paths.size());
        for (size_t i = 0; i < vas_paths.size(); ++i) {
            sub_paths[i].assign(vas_paths[i].begin(), vas_paths[i].begin() + steps_T + 1);
        }
        auto mc = mc_bond_price(sub_paths, T);

        std::cout << std::setw(10) << T
                  << std::setw(14) << anal_P
                  << std::setw(14) << mc.price
                  << std::setw(10) << mc.std_error
                  << std::setw(13) << anal_r * 100 << "%"
                  << std::setw(13) << mc.zero_rate * 100 << "%\n";
    }

    // ================================================================
    // 2. CIR Model — Feller Condition Analysis
    // ================================================================
    std::cout << "\n=== 2. CIR Model & Feller Condition ===\n\n";

    CIRParams cp_feller(0.05, 0.5, 0.05, 0.05);
    CIRParams cp_no_feller(0.05, 0.5, 0.05, 0.30);

    std::cout << "Case 1: sigma=" << cp_feller.sigma
              << " Feller: " << (cp_feller.feller_satisfied() ? "YES" : "NO")
              << " (2*kappa*theta=" << 2*cp_feller.kappa*cp_feller.theta
              << " vs sigma^2=" << cp_feller.sigma*cp_feller.sigma << ")\n";
    std::cout << "Case 2: sigma=" << cp_no_feller.sigma
              << " Feller: " << (cp_no_feller.feller_satisfied() ? "YES" : "NO")
              << " (2*kappa*theta=" << 2*cp_no_feller.kappa*cp_no_feller.theta
              << " vs sigma^2=" << cp_no_feller.sigma*cp_no_feller.sigma << ")\n\n";

    // CIR analytic vs MC
    std::cout << std::setw(10) << "Maturity"
              << std::setw(14) << "Analytic P"
              << std::setw(14) << "Zero Rate" << "\n";
    std::cout << std::string(38, '-') << "\n";

    for (double T : {1.0, 2.0, 5.0, 10.0, 20.0, 30.0}) {
        double P = cir_bond_price(T, cp_feller);
        double r = cir_zero_rate(T, cp_feller);
        std::cout << std::setw(10) << T
                  << std::setw(14) << P
                  << std::setw(13) << r * 100 << "%\n";
    }

    // ================================================================
    // 3. Model Comparison: Vasicek vs CIR Yield Curves
    // ================================================================
    std::cout << "\n=== 3. Vasicek vs CIR Yield Curves ===\n\n";

    // Same mean-reversion params, different dynamics
    VasicekParams vp2(0.03, 0.3, 0.06, 0.02);
    CIRParams cp2(0.03, 0.3, 0.06, 0.08);

    std::cout << std::setw(10) << "Maturity"
              << std::setw(14) << "Vasicek r"
              << std::setw(14) << "CIR r"
              << std::setw(14) << "Difference" << "\n";
    std::cout << std::string(52, '-') << "\n";

    for (double T : {0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0}) {
        double vr = vasicek_zero_rate(T, vp2);
        double cr = cir_zero_rate(T, cp2);
        std::cout << std::setw(10) << T
                  << std::setw(13) << vr * 100 << "%"
                  << std::setw(13) << cr * 100 << "%"
                  << std::setw(13) << (vr - cr) * 10000 << "bp\n";
    }

    // ================================================================
    // 4. Vasicek: Negative Rate Probability
    // ================================================================
    std::cout << "\n=== 4. Vasicek: Probability of Negative Rates ===\n\n";

    std::cout << "r(t) ~ N(mu(t), var(t)) under Vasicek\n";
    std::cout << "P(r < 0) = Phi(-mu/sqrt(var))\n\n";

    VasicekParams vp3(0.02, 0.3, 0.03, 0.02);  // Low initial rate
    std::cout << "Params: r0=" << vp3.r0 << " theta=" << vp3.theta
              << " sigma=" << vp3.sigma << "\n\n";

    auto N_cdf = [](double x) { return 0.5 * std::erfc(-x * M_SQRT1_2); };

    std::cout << std::setw(10) << "Time"
              << std::setw(12) << "E[r(t)]"
              << std::setw(12) << "Std[r(t)]"
              << std::setw(14) << "P(r<0)" << "\n";
    std::cout << std::string(48, '-') << "\n";

    for (double t : {0.5, 1.0, 2.0, 5.0, 10.0, 20.0}) {
        double mu = vasicek_mean(t, vp3);
        double sd = std::sqrt(vasicek_var(t, vp3));
        double p_neg = N_cdf(-mu / sd);
        std::cout << std::setw(10) << t
                  << std::setw(11) << mu * 100 << "%"
                  << std::setw(11) << sd * 100 << "%"
                  << std::setw(13) << p_neg * 100 << "%\n";
    }
    std::cout << "\nNote: negative rate probability converges to Phi(-theta/sigma_inf)\n"
              << "where sigma_inf = sigma/sqrt(2*kappa) = "
              << vp3.sigma / std::sqrt(2 * vp3.kappa) * 100 << "%\n";

    // ================================================================
    // 5. Vasicek Bond Option Pricing
    // ================================================================
    std::cout << "\n=== 5. Vasicek Bond Option (Jamshidian) ===\n\n";

    VasicekParams vp_opt(0.05, 0.5, 0.05, 0.015);
    double S = 5.0;  // Bond maturity
    double T_opt = 1.0;  // Option expiry
    double P_S = vasicek_bond_price(S, vp_opt);

    std::cout << "ZCB P(0," << S << ") = " << P_S << "\n";
    std::cout << "Option expiry T=" << T_opt << " on ZCB maturing at S=" << S << "\n\n";

    std::cout << std::setw(10) << "Strike"
              << std::setw(14) << "Call"
              << std::setw(14) << "Put"
              << std::setw(14) << "Parity Err" << "\n";
    std::cout << std::string(52, '-') << "\n";

    double P_T = vasicek_bond_price(T_opt, vp_opt);
    for (double X : {0.78, 0.80, 0.82, 0.84, 0.86}) {
        double call = vasicek_bond_call(T_opt, S, X, vp_opt);
        double put = vasicek_bond_put(T_opt, S, X, vp_opt);
        // Put-call parity: C - P = P(0,S) - X*P(0,T)
        double parity_err = std::abs((call - put) - (P_S - X * P_T));
        std::cout << std::setw(10) << X
                  << std::setw(14) << call
                  << std::setw(14) << put
                  << std::setw(14) << parity_err << "\n";
    }

    // ================================================================
    // 6. Hull-White — Fitting the Market Curve
    // ================================================================
    std::cout << "\n=== 6. Hull-White: Calibration to Market Curve ===\n\n";

    // Build a market curve via Nelson-Siegel
    fixed_income::NelsonSiegelParams ns(0.045, -0.015, 0.02, 2.0);
    auto mkt_curve = fixed_income::build_nelson_siegel_curve(ns);

    HullWhiteParams hwp(0.1, 0.01);
    std::cout << "Market curve: Nelson-Siegel (b0=4.5%, b1=-1.5%, b2=2%)\n";
    std::cout << "Hull-White: a=" << hwp.a << " sigma=" << hwp.sigma << "\n\n";

    // Show theta(t) calibration
    std::cout << std::setw(8) << "Time"
              << std::setw(14) << "Mkt Fwd"
              << std::setw(14) << "theta(t)" << "\n";
    std::cout << std::string(36, '-') << "\n";

    for (double t : {0.5, 1.0, 2.0, 5.0, 10.0, 20.0}) {
        double fwd = mkt_curve.inst_forward(t);
        double th = hw_theta(t, hwp, mkt_curve);
        std::cout << std::setw(8) << t
                  << std::setw(13) << fwd * 100 << "%"
                  << std::setw(13) << th * 100 << "%\n";
    }

    std::cout << "\nHull-White theta(t) adapts to match the market curve exactly.\n"
              << "This is why it's the industry standard for rates derivatives.\n";

    return 0;
}

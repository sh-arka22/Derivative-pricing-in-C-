// ============================================================================
// Day 13 Example: Fixed Income — Yield Curves & Bond Pricing
//
// Demonstrates:
// 1. Nelson-Siegel yield curve construction
// 2. Yield curve bootstrapping from market instruments
// 3. Bond pricing from discount factors
// 4. Duration, convexity, DV01 risk measures
// 5. Duration-convexity price approximation accuracy
// 6. Yield curve shape analysis (normal, flat, inverted)
// 7. Forward rate curve
// ============================================================================

#include "fixed_income/fixed_income.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace fixed_income;

int main() {
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "======================================================================\n"
              << "  Day 13: Fixed Income — Yield Curves & Bond Pricing\n"
              << "======================================================================\n\n";

    // ================================================================
    // 1. Nelson-Siegel Yield Curve
    // ================================================================
    std::cout << "=== 1. Nelson-Siegel Yield Curve ===\n\n";

    // Normal upward-sloping curve
    NelsonSiegelParams ns(0.045, -0.015, 0.02, 2.0);
    auto curve = build_nelson_siegel_curve(ns);

    std::cout << "Parameters: b0=" << ns.beta0 << " b1=" << ns.beta1
              << " b2=" << ns.beta2 << " tau=" << ns.tau << "\n";
    std::cout << "Short rate r(0) = b0+b1 = " << (ns.beta0 + ns.beta1) * 100 << "%\n";
    std::cout << "Long rate r(inf) = b0 = " << ns.beta0 * 100 << "%\n\n";

    std::cout << std::setw(10) << "Tenor"
              << std::setw(12) << "Zero Rate"
              << std::setw(14) << "Discount"
              << std::setw(14) << "Fwd 1y" << "\n";
    std::cout << std::string(50, '-') << "\n";

    for (double t : {0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0}) {
        double fwd = (t >= 1.0) ? curve.forward_rate(t - 0.5, t + 0.5) : curve.zero_rate(t);
        std::cout << std::setw(10) << t
                  << std::setw(11) << curve.zero_rate(t) * 100 << "%"
                  << std::setw(14) << curve.discount(t)
                  << std::setw(13) << fwd * 100 << "%\n";
    }

    // ================================================================
    // 2. Bootstrapping from Market Instruments
    // ================================================================
    std::cout << "\n=== 2. Yield Curve Bootstrapping ===\n\n";

    std::vector<BootstrapInstrument> instruments = {
        {0.25, 0.0300, true},   // 3-month deposit
        {0.50, 0.0320, true},   // 6-month deposit
        {1.00, 0.0350, true},   // 1-year deposit
        {2.00, 0.0380, false},  // 2-year swap
        {3.00, 0.0400, false},  // 3-year swap
        {5.00, 0.0420, false},  // 5-year swap
        {7.00, 0.0435, false},  // 7-year swap
        {10.0, 0.0450, false},  // 10-year swap
        {20.0, 0.0460, false},  // 20-year swap
        {30.0, 0.0465, false},  // 30-year swap
    };

    auto boot_curve = bootstrap(instruments);

    std::cout << "Market instruments → bootstrapped zero rates:\n\n";
    std::cout << std::setw(10) << "Tenor"
              << std::setw(14) << "Mkt Rate"
              << std::setw(14) << "Zero Rate"
              << std::setw(14) << "Discount"
              << std::setw(10) << "Type" << "\n";
    std::cout << std::string(62, '-') << "\n";

    for (size_t i = 0; i < instruments.size(); ++i) {
        auto& inst = instruments[i];
        std::cout << std::setw(10) << inst.maturity
                  << std::setw(13) << inst.rate * 100 << "%"
                  << std::setw(13) << boot_curve.zeros()[i] * 100 << "%"
                  << std::setw(14) << boot_curve.discount(inst.maturity)
                  << std::setw(10) << (inst.is_deposit ? "Deposit" : "Swap") << "\n";
    }

    // ================================================================
    // 3. Bond Pricing
    // ================================================================
    std::cout << "\n=== 3. Bond Pricing from Yield Curve ===\n\n";

    std::vector<Bond> bonds = {
        {100, 0.00, 5, 2},   // 5Y zero-coupon
        {100, 0.03, 5, 2},   // 5Y 3% coupon
        {100, 0.05, 5, 2},   // 5Y 5% coupon (near par)
        {100, 0.07, 5, 2},   // 5Y 7% coupon (premium)
        {100, 0.05, 2, 2},   // 2Y 5% coupon
        {100, 0.05, 10, 2},  // 10Y 5% coupon
        {100, 0.05, 30, 2},  // 30Y 5% coupon
    };

    std::cout << std::setw(12) << "Bond"
              << std::setw(10) << "Price"
              << std::setw(10) << "YTM"
              << std::setw(10) << "MacDur"
              << std::setw(10) << "ModDur"
              << std::setw(10) << "Convex"
              << std::setw(10) << "DV01" << "\n";
    std::cout << std::string(72, '-') << "\n";

    for (auto& b : bonds) {
        double p = bond_price(b, boot_curve);
        double y = yield_to_maturity(b, p);
        auto risk = bond_risk(b, y);

        char label[32];
        snprintf(label, sizeof(label), "%.0fy %.0f%%", b.maturity, b.coupon * 100);

        std::cout << std::setw(12) << label
                  << std::setw(10) << risk.price
                  << std::setw(9) << risk.ytm * 100 << "%"
                  << std::setw(10) << risk.macaulay_duration
                  << std::setw(10) << risk.modified_duration
                  << std::setw(10) << risk.convexity
                  << std::setw(10) << risk.dv01 << "\n";
    }

    std::cout << "\nKey insight: zero-coupon 5Y has duration = 5.0 (upper bound)\n"
              << "Higher coupon → lower duration (coupons pull avg time forward)\n"
              << "Longer maturity → higher duration and convexity\n";

    // ================================================================
    // 4. Duration-Convexity Approximation Accuracy
    // ================================================================
    std::cout << "\n=== 4. Duration-Convexity Approximation ===\n\n";

    Bond test_bond(100, 0.05, 10, 2);
    double test_price = bond_price(test_bond, boot_curve);
    double test_ytm = yield_to_maturity(test_bond, test_price);
    auto test_risk = bond_risk(test_bond, test_ytm);

    std::cout << "10Y 5% bond: Price=" << test_price << " YTM="
              << test_ytm*100 << "% Duration=" << test_risk.modified_duration
              << " Convexity=" << test_risk.convexity << "\n\n";

    std::cout << std::setw(10) << "dy (bp)"
              << std::setw(14) << "Exact dP"
              << std::setw(14) << "Dur Only"
              << std::setw(14) << "Dur+Conv"
              << std::setw(14) << "Dur Err"
              << std::setw(14) << "D+C Err" << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (int bp : {-200, -100, -50, -10, 10, 50, 100, 200}) {
        double dy = bp * 0.0001;
        double new_price = bond_price_at_yield(test_bond, test_ytm + dy);
        double exact_dp = new_price - test_price;

        // Duration-only approximation: dP ≈ -D_mod * P * dy
        double dur_dp = -test_risk.modified_duration * test_price * dy;
        // Duration + convexity: dP ≈ -D_mod*P*dy + 0.5*C*P*dy^2
        double dc_dp = price_change_approx(test_risk, dy);

        std::cout << std::setw(10) << bp
                  << std::setw(14) << exact_dp
                  << std::setw(14) << dur_dp
                  << std::setw(14) << dc_dp
                  << std::setw(14) << std::abs(dur_dp - exact_dp)
                  << std::setw(14) << std::abs(dc_dp - exact_dp) << "\n";
    }
    std::cout << "\nDuration alone: good for |dy| < 50bp\n"
              << "Duration+convexity: excellent for |dy| < 200bp\n";

    // ================================================================
    // 5. Yield Curve Shapes
    // ================================================================
    std::cout << "\n=== 5. Yield Curve Shapes ===\n\n";

    NelsonSiegelParams normal_ns(0.045, -0.015, 0.01, 2.0);  // Upward sloping
    NelsonSiegelParams flat_ns(0.04, 0.0, 0.0, 2.0);         // Flat
    NelsonSiegelParams inv_ns(0.03, 0.02, -0.01, 2.0);       // Inverted
    NelsonSiegelParams hump_ns(0.04, -0.01, 0.04, 1.5);      // Humped

    auto c_normal = build_nelson_siegel_curve(normal_ns);
    auto c_flat = build_nelson_siegel_curve(flat_ns);
    auto c_inv = build_nelson_siegel_curve(inv_ns);
    auto c_hump = build_nelson_siegel_curve(hump_ns);

    std::cout << std::setw(10) << "Tenor"
              << std::setw(12) << "Normal"
              << std::setw(12) << "Flat"
              << std::setw(12) << "Inverted"
              << std::setw(12) << "Humped" << "\n";
    std::cout << std::string(58, '-') << "\n";

    for (double t : {0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0}) {
        std::cout << std::setw(10) << t
                  << std::setw(11) << c_normal.zero_rate(t)*100 << "%"
                  << std::setw(11) << c_flat.zero_rate(t)*100 << "%"
                  << std::setw(11) << c_inv.zero_rate(t)*100 << "%"
                  << std::setw(11) << c_hump.zero_rate(t)*100 << "%\n";
    }

    std::cout << "\n10y-2y spread (recession indicator):\n";
    std::cout << "  Normal:   " << (c_normal.zero_rate(10)-c_normal.zero_rate(2))*100 << "bp\n";
    std::cout << "  Flat:     " << (c_flat.zero_rate(10)-c_flat.zero_rate(2))*100 << "bp\n";
    std::cout << "  Inverted: " << (c_inv.zero_rate(10)-c_inv.zero_rate(2))*100 << "bp (RECESSION SIGNAL)\n";

    // ================================================================
    // 6. Par Rates & Curve Scenarios
    // ================================================================
    std::cout << "\n=== 6. Par Rates & DV01 Sensitivity ===\n\n";

    std::cout << std::setw(10) << "Maturity"
              << std::setw(12) << "Par Rate"
              << std::setw(12) << "DV01/100" << "\n";
    std::cout << std::string(34, '-') << "\n";

    for (double m : {1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0}) {
        double pr = par_rate(boot_curve, m);
        Bond par_bond(100, pr, m, 2);
        double p = bond_price(par_bond, boot_curve);
        double y = yield_to_maturity(par_bond, p);
        auto risk = bond_risk(par_bond, y);
        std::cout << std::setw(10) << m
                  << std::setw(11) << pr * 100 << "%"
                  << std::setw(12) << risk.dv01 << "\n";
    }

    std::cout << "\nDV01 interpretation: 30Y par bond DV01 ~ $0.17\n"
              << "→ $17,000 P&L per bp on a $10M position\n";

    return 0;
}

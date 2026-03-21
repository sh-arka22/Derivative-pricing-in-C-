// ============================================================================
// Day 10 Example: Barrier Options
//
// Demonstrates:
// 1. Analytic pricing for all 8 barrier option types
// 2. In-out parity validation
// 3. Monte Carlo barrier pricing with discrete monitoring
// 4. Broadie-Glasserman-Kou continuity correction
// 5. Barrier sensitivity analysis (price vs spot, varying barrier)
// ============================================================================

#include "barrier/barrier.h"
#include "greeks/black_scholes.h"
#include <iostream>
#include <iomanip>
#include <string>

using namespace barrier;

int main() {
    std::cout << std::fixed << std::setprecision(4);

    double S0 = 100.0, K = 100.0, r = 0.05, T = 1.0, sigma = 0.2;

    std::cout << "======================================================================\n"
              << "  Day 10: Barrier Options\n"
              << "======================================================================\n\n";

    double bs_call = bs::call_price(S0, K, r, T, sigma);
    double bs_put  = bs::put_price(S0, K, r, T, sigma);
    std::cout << "Vanilla reference: Call = " << bs_call
              << ", Put = " << bs_put << "\n\n";

    // ================================================================
    // 1. Analytic pricing — all 8 types
    // ================================================================
    std::cout << "=== 1. Analytic Barrier Prices (Continuous Monitoring) ===\n\n";

    struct Case {
        const char* name;
        BarrierType bt;
        OptionType ot;
        double H;
    };

    std::vector<Case> cases = {
        {"Down-and-Out Call", BarrierType::DownAndOut, OptionType::Call, 90.0},
        {"Down-and-In  Call", BarrierType::DownAndIn,  OptionType::Call, 90.0},
        {"Up-and-Out   Call", BarrierType::UpAndOut,   OptionType::Call, 120.0},
        {"Up-and-In    Call", BarrierType::UpAndIn,    OptionType::Call, 120.0},
        {"Down-and-Out Put ", BarrierType::DownAndOut, OptionType::Put,  90.0},
        {"Down-and-In  Put ", BarrierType::DownAndIn,  OptionType::Put,  90.0},
        {"Up-and-Out   Put ", BarrierType::UpAndOut,   OptionType::Put,  120.0},
        {"Up-and-In    Put ", BarrierType::UpAndIn,    OptionType::Put,  120.0},
    };

    std::cout << std::setw(22) << "Type"
              << std::setw(10) << "Barrier"
              << std::setw(12) << "Price"
              << std::setw(12) << "Vanilla"
              << std::setw(14) << "Discount(%)" << "\n";
    std::cout << std::string(70, '-') << "\n";

    for (auto& c : cases) {
        BarrierParams bp(S0, K, c.H, r, T, sigma, c.bt, c.ot);
        double price = barrier_price(bp);
        double vanilla = (c.ot == OptionType::Call) ? bs_call : bs_put;
        double discount_pct = 100.0 * (1.0 - price / vanilla);
        std::cout << std::setw(22) << c.name
                  << std::setw(10) << c.H
                  << std::setw(12) << price
                  << std::setw(12) << vanilla
                  << std::setw(13) << discount_pct << "%\n";
    }

    // ================================================================
    // 2. In-Out Parity Validation
    // ================================================================
    std::cout << "\n=== 2. In-Out Parity: V_in + V_out = V_vanilla ===\n\n";

    std::cout << std::setw(22) << "Pair"
              << std::setw(10) << "V_out"
              << std::setw(10) << "V_in"
              << std::setw(10) << "Sum"
              << std::setw(10) << "Vanilla"
              << std::setw(12) << "Error" << "\n";
    std::cout << std::string(74, '-') << "\n";

    auto check_parity = [&](const char* name, BarrierType out_t, BarrierType in_t,
                            OptionType ot, double H) {
        BarrierParams p_out(S0, K, H, r, T, sigma, out_t, ot);
        BarrierParams p_in(S0, K, H, r, T, sigma, in_t, ot);
        double v_out = barrier_price(p_out);
        double v_in  = barrier_price(p_in);
        double vanilla = (ot == OptionType::Call) ? bs_call : bs_put;
        std::cout << std::setw(22) << name
                  << std::setw(10) << v_out
                  << std::setw(10) << v_in
                  << std::setw(10) << (v_out + v_in)
                  << std::setw(10) << vanilla
                  << std::setw(12) << std::abs(v_out + v_in - vanilla) << "\n";
    };

    check_parity("Down Call H=90",  BarrierType::DownAndOut, BarrierType::DownAndIn,
                 OptionType::Call, 90.0);
    check_parity("Up Call H=120",   BarrierType::UpAndOut, BarrierType::UpAndIn,
                 OptionType::Call, 120.0);
    check_parity("Down Put H=90",   BarrierType::DownAndOut, BarrierType::DownAndIn,
                 OptionType::Put, 90.0);
    check_parity("Up Put H=120",    BarrierType::UpAndOut, BarrierType::UpAndIn,
                 OptionType::Put, 120.0);

    // ================================================================
    // 3. MC vs Analytic Comparison
    // ================================================================
    std::cout << "\n=== 3. MC (Discrete) vs Analytic (Continuous) ===\n\n";

    BarrierParams doc(S0, K, 90.0, r, T, sigma, BarrierType::DownAndOut, OptionType::Call);
    double analytic = barrier_price(doc);

    std::cout << std::setw(10) << "Steps"
              << std::setw(12) << "MC Price"
              << std::setw(10) << "SE"
              << std::setw(12) << "Analytic"
              << std::setw(12) << "MC-Anal"
              << std::setw(12) << "BGK Price"
              << std::setw(12) << "Knock(%)" << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (size_t steps : {12, 50, 100, 252, 500, 1000}) {
        auto mc = mc_barrier(doc, 200000, steps, 42);
        double bgk = barrier_price_bgk(doc, steps);
        std::cout << std::setw(10) << steps
                  << std::setw(12) << mc.price
                  << std::setw(10) << mc.std_error
                  << std::setw(12) << analytic
                  << std::setw(12) << (mc.price - analytic)
                  << std::setw(12) << bgk
                  << std::setw(12) << mc.knock_pct << "\n";
    }

    // ================================================================
    // 4. BGK Correction Accuracy
    // ================================================================
    std::cout << "\n=== 4. BGK Correction: Analytic(adjusted) vs MC(discrete) ===\n\n";

    std::cout << "Down-and-Out Call, S0=100, K=100, H=90, sigma=0.2\n\n";
    std::cout << std::setw(10) << "Steps"
              << std::setw(14) << "Continuous"
              << std::setw(14) << "BGK(disc)"
              << std::setw(14) << "MC(disc)"
              << std::setw(14) << "|BGK-MC|" << "\n";
    std::cout << std::string(66, '-') << "\n";

    for (size_t steps : {12, 50, 100, 252, 1000}) {
        double continuous = barrier_price(doc);
        double bgk = barrier_price_bgk(doc, steps);
        auto mc = mc_barrier(doc, 300000, steps, 42);
        std::cout << std::setw(10) << steps
                  << std::setw(14) << continuous
                  << std::setw(14) << bgk
                  << std::setw(14) << mc.price
                  << std::setw(14) << std::abs(bgk - mc.price) << "\n";
    }

    // ================================================================
    // 5. Barrier Sensitivity: Price vs Barrier Level
    // ================================================================
    std::cout << "\n=== 5. Down-and-Out Call: Price vs Barrier Level ===\n\n";

    std::cout << std::setw(10) << "Barrier"
              << std::setw(14) << "DO Call"
              << std::setw(14) << "DI Call"
              << std::setw(14) << "Vanilla"
              << std::setw(14) << "Parity Err" << "\n";
    std::cout << std::string(66, '-') << "\n";

    for (double H : {50.0, 60.0, 70.0, 80.0, 85.0, 90.0, 95.0, 98.0, 99.0}) {
        BarrierParams p_do(S0, K, H, r, T, sigma, BarrierType::DownAndOut, OptionType::Call);
        BarrierParams p_di(S0, K, H, r, T, sigma, BarrierType::DownAndIn,  OptionType::Call);
        double v_do = barrier_price(p_do);
        double v_di = barrier_price(p_di);
        double parity_err = std::abs(v_do + v_di - bs_call);
        std::cout << std::setw(10) << H
                  << std::setw(14) << v_do
                  << std::setw(14) << v_di
                  << std::setw(14) << bs_call
                  << std::setw(14) << parity_err << "\n";
    }
    std::cout << "\nNote: as H -> 0, DO Call -> Vanilla (barrier never hit)\n"
              << "      as H -> S0, DO Call -> 0 (almost certain knock-out)\n";

    // ================================================================
    // 6. Up-and-Out Put (common in structured products)
    // ================================================================
    std::cout << "\n=== 6. Up-and-Out Put: Popular Structured Product ===\n\n";

    std::cout << std::setw(10) << "Barrier"
              << std::setw(14) << "UO Put"
              << std::setw(14) << "Vanilla Put"
              << std::setw(14) << "Savings(%)" << "\n";
    std::cout << std::string(52, '-') << "\n";

    for (double H : {105.0, 110.0, 115.0, 120.0, 130.0, 150.0}) {
        BarrierParams p_uo(S0, K, H, r, T, sigma, BarrierType::UpAndOut, OptionType::Put);
        double v_uo = barrier_price(p_uo);
        double savings = 100.0 * (1.0 - v_uo / bs_put);
        std::cout << std::setw(10) << H
                  << std::setw(14) << v_uo
                  << std::setw(14) << bs_put
                  << std::setw(13) << savings << "%\n";
    }

    return 0;
}

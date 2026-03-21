// ============================================================================
// Day 16 Example: American Options Pricing
//
// Demonstrates:
// 1. CRR Binomial Tree — European vs American puts
// 2. Convergence of binomial tree to analytic BS
// 3. Longstaff-Schwartz MC for American puts
// 4. Early exercise premium analysis
// 5. Tree-implied Greeks for American options
// ============================================================================

#include "tree/binomial_tree.h"
#include "mc/american_mc.h"
#include "greeks/black_scholes.h"
#include "greeks/greeks_engine.h"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << std::fixed << std::setprecision(4);

    double S0 = 100.0, K = 100.0, r = 0.05, T = 1.0, sigma = 0.2;
    PayOffPut put(K);
    PayOffCall call(K);

    std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
              << "║     Day 16: American Options & Binomial Trees               ║\n"
              << "╚══════════════════════════════════════════════════════════════╝\n\n";

    // ================================================================
    // 1. Binomial Tree Convergence Study
    // ================================================================
    std::cout << "=== 1. Binomial Tree Convergence (European Call) ===\n\n";
    double bs_call = bs::call_price(S0, K, r, T, sigma);
    double bs_put = bs::put_price(S0, K, r, T, sigma);

    std::cout << "  Analytic BS Call: " << bs_call << "\n";
    std::cout << "  Analytic BS Put:  " << bs_put << "\n\n";

    std::cout << std::setw(10) << "Steps"
              << std::setw(14) << "Tree Call"
              << std::setw(14) << "Error"
              << std::setw(14) << "Tree Put"
              << std::setw(14) << "Error" << "\n";
    std::cout << std::string(66, '-') << "\n";

    for (size_t steps : {10, 50, 100, 200, 500, 1000, 2000}) {
        auto tree_call = binomial_european(call, S0, K, r, T, sigma, steps);
        auto tree_put = binomial_european(put, S0, K, r, T, sigma, steps);
        std::cout << std::setw(10) << steps
                  << std::setw(14) << tree_call.price
                  << std::setw(14) << (tree_call.price - bs_call)
                  << std::setw(14) << tree_put.price
                  << std::setw(14) << (tree_put.price - bs_put) << "\n";
    }

    // ================================================================
    // 2. American vs European Put (early exercise premium)
    // ================================================================
    std::cout << "\n=== 2. American vs European Put (Binomial Tree) ===\n\n";

    std::cout << std::setw(10) << "Steps"
              << std::setw(14) << "Eur Put"
              << std::setw(14) << "Ame Put"
              << std::setw(14) << "EE Premium" << "\n";
    std::cout << std::string(52, '-') << "\n";

    for (size_t steps : {100, 200, 500, 1000, 2000}) {
        auto eur = binomial_european(put, S0, K, r, T, sigma, steps);
        auto ame = binomial_american(put, S0, K, r, T, sigma, steps);
        double eep = ame.price - eur.price;
        std::cout << std::setw(10) << steps
                  << std::setw(14) << eur.price
                  << std::setw(14) << ame.price
                  << std::setw(14) << eep << "\n";
    }

    // ================================================================
    // 3. American Put across moneyness
    // ================================================================
    std::cout << "\n=== 3. American Put across Moneyness (N=1000) ===\n\n";

    std::cout << std::setw(10) << "Spot"
              << std::setw(14) << "Eur Put"
              << std::setw(14) << "Ame Put"
              << std::setw(14) << "EE Premium"
              << std::setw(14) << "Premium %" << "\n";
    std::cout << std::string(66, '-') << "\n";

    for (double spot : {80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0}) {
        auto eur = binomial_european(put, spot, K, r, T, sigma, 1000);
        auto ame = binomial_american(put, spot, K, r, T, sigma, 1000);
        double eep = ame.price - eur.price;
        double pct = (eur.price > 0.01) ? 100.0 * eep / eur.price : 0.0;
        std::cout << std::setw(10) << spot
                  << std::setw(14) << eur.price
                  << std::setw(14) << ame.price
                  << std::setw(14) << eep
                  << std::setw(13) << pct << "%" << "\n";
    }

    // ================================================================
    // 4. Tree-implied Greeks for American Put
    // ================================================================
    std::cout << "\n=== 4. American Put Greeks from Binomial Tree (N=1000) ===\n\n";

    auto ame_greeks = binomial_american(put, S0, K, r, T, sigma, 1000);
    auto eur_greeks = analytic_greeks_put(S0, K, r, T, sigma);

    std::cout << std::setw(12) << "Greek"
              << std::setw(16) << "American (Tree)"
              << std::setw(16) << "European (BS)" << "\n";
    std::cout << std::string(44, '-') << "\n";
    std::cout << std::setw(12) << "Price"
              << std::setw(16) << ame_greeks.price
              << std::setw(16) << bs_put << "\n";
    std::cout << std::setw(12) << "Delta"
              << std::setw(16) << ame_greeks.delta
              << std::setw(16) << eur_greeks.delta << "\n";
    std::cout << std::setw(12) << "Gamma"
              << std::setw(16) << ame_greeks.gamma
              << std::setw(16) << eur_greeks.gamma << "\n";
    std::cout << std::setw(12) << "Theta"
              << std::setw(16) << ame_greeks.theta
              << std::setw(16) << eur_greeks.theta << "\n";

    // ================================================================
    // 5. Longstaff-Schwartz MC for American Put
    // ================================================================
    std::cout << "\n=== 5. Longstaff-Schwartz MC: American Put ===\n\n";

    std::cout << std::setw(10) << "Paths"
              << std::setw(14) << "LSM Price"
              << std::setw(12) << "SE"
              << std::setw(14) << "EE Premium"
              << std::setw(18) << "95% CI" << "\n";
    std::cout << std::string(68, '-') << "\n";

    for (size_t paths : {10000, 50000, 100000, 200000}) {
        auto lsm = mc_american_lsm(put, S0, r, T, sigma, paths, 50, 3, 42);
        std::cout << std::setw(10) << paths
                  << std::setw(14) << lsm.price
                  << std::setw(12) << lsm.std_error
                  << std::setw(14) << lsm.early_exercise_premium
                  << "    [" << lsm.confidence_lo << ", " << lsm.confidence_hi << "]\n";
    }

    // ================================================================
    // 6. American Call — no early exercise premium (no dividends)
    // ================================================================
    std::cout << "\n=== 6. American Call: No Early Exercise (No Dividends) ===\n\n";

    auto eur_call = binomial_european(call, S0, K, r, T, sigma, 1000);
    auto ame_call = binomial_american(call, S0, K, r, T, sigma, 1000);
    double call_eep = ame_call.price - eur_call.price;

    std::cout << "  European Call (tree): " << eur_call.price << "\n";
    std::cout << "  American Call (tree): " << ame_call.price << "\n";
    std::cout << "  EE Premium:          " << call_eep
              << " (should be ~0 for non-dividend paying stock)\n";

    // ================================================================
    // 7. Cross-validation: Tree vs LSM for American Put
    // ================================================================
    std::cout << "\n=== 7. Cross-Validation: Tree vs LSM ===\n\n";

    auto tree_ref = binomial_american(put, S0, K, r, T, sigma, 2000);
    auto lsm_ref = mc_american_lsm(put, S0, r, T, sigma, 200000, 50, 3, 42);

    std::cout << "  Binomial Tree (N=2000):    " << tree_ref.price << "\n";
    std::cout << "  LSM MC (200K paths):       " << lsm_ref.price
              << " +/- " << lsm_ref.std_error << "\n";
    std::cout << "  Difference:                "
              << std::abs(tree_ref.price - lsm_ref.price) << "\n";

    return 0;
}

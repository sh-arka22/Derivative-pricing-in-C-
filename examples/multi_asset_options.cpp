// ============================================================================
// Day 11 Example: Multi-Asset & Basket Options
//
// Demonstrates:
// 1. Margrabe exchange option — analytic vs MC validation
// 2. Basket call pricing — MC vs lognormal approximation
// 3. Best-of / worst-of options — correlation sensitivity
// 4. Spread options — vol of spread relationship
// 5. Correlation impact study — the "correlation trade"
// ============================================================================

#include "multi_asset/multi_asset.h"
#include "greeks/black_scholes.h"
#include <iostream>
#include <iomanip>

using namespace multi_asset;

int main() {
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "======================================================================\n"
              << "  Day 11: Multi-Asset & Basket Options\n"
              << "======================================================================\n\n";

    double S0 = 100.0, K = 100.0, r = 0.05, T = 1.0, sigma = 0.2;
    size_t num_paths = 300000;

    // ================================================================
    // 1. Margrabe Exchange Option — Analytic Validation
    // ================================================================
    std::cout << "=== 1. Margrabe Exchange Option: max(S1 - S2, 0) ===\n\n";
    std::cout << "Key insight: price does NOT depend on r (S2 is numeraire)\n\n";

    double S1 = 100.0, S2 = 100.0, sigma1 = 0.2, sigma2 = 0.3;

    std::cout << std::setw(8) << "Rho"
              << std::setw(14) << "Analytic"
              << std::setw(14) << "MC Price"
              << std::setw(10) << "SE"
              << std::setw(14) << "Spread Vol"
              << std::setw(12) << "|Error|" << "\n";
    std::cout << std::string(72, '-') << "\n";

    for (double rho : {-0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 0.9}) {
        auto marg = margrabe_price(S1, S2, sigma1, sigma2, rho, T);

        MultiAssetParams p;
        p.S0 = {S1, S2};
        p.sigma = {sigma1, sigma2};
        p.weights = {1.0, 1.0};
        p.corr = uniform_corr_matrix(2, rho);
        p.r = r; p.T = T; p.K = 0.0;

        auto mc = mc_multi_asset(p, PayoffType::Exchange, num_paths);

        std::cout << std::setw(8) << rho
                  << std::setw(14) << marg.price
                  << std::setw(14) << mc.price
                  << std::setw(10) << mc.std_error
                  << std::setw(14) << marg.sigma_spread
                  << std::setw(12) << std::abs(marg.price - mc.price) << "\n";
    }
    std::cout << "\nNote: higher rho -> lower spread vol -> cheaper exchange option\n";

    // ================================================================
    // 2. Basket Call — MC vs Lognormal Approximation
    // ================================================================
    std::cout << "\n=== 2. Equal-Weight Basket Call (N assets, rho=0.5) ===\n\n";

    std::cout << std::setw(10) << "N assets"
              << std::setw(14) << "MC Price"
              << std::setw(10) << "SE"
              << std::setw(14) << "LN Approx"
              << std::setw(14) << "Basket Vol"
              << std::setw(14) << "BS Single" << "\n";
    std::cout << std::string(76, '-') << "\n";

    double bs_single = bs::call_price(S0, K, r, T, sigma);

    for (size_t N : {2, 3, 5, 10, 20}) {
        auto params = make_basket_params(N, S0, sigma, 0.5, r, T, K);
        auto mc = mc_multi_asset(params, PayoffType::BasketCall, num_paths);
        double approx = basket_call_approx(params);
        double bvol = basket_vol(params);

        std::cout << std::setw(10) << N
                  << std::setw(14) << mc.price
                  << std::setw(10) << mc.std_error
                  << std::setw(14) << approx
                  << std::setw(14) << bvol
                  << std::setw(14) << bs_single << "\n";
    }
    std::cout << "\nNote: basket vol < individual vol due to diversification\n"
              << "      basket call < single-asset call (averaging reduces vol)\n";

    // ================================================================
    // 3. Best-of vs Worst-of Calls
    // ================================================================
    std::cout << "\n=== 3. Best-of vs Worst-of Call (2 assets, S0=100 each) ===\n\n";

    std::cout << std::setw(8) << "Rho"
              << std::setw(14) << "Best-of"
              << std::setw(14) << "Worst-of"
              << std::setw(14) << "Basket"
              << std::setw(14) << "Single BS" << "\n";
    std::cout << std::string(64, '-') << "\n";

    for (double rho : {-0.5, 0.0, 0.3, 0.5, 0.7, 0.9, 0.99}) {
        MultiAssetParams p;
        p.S0 = {S0, S0};
        p.sigma = {sigma, sigma};
        p.weights = {0.5, 0.5};
        p.corr = uniform_corr_matrix(2, rho);
        p.r = r; p.T = T; p.K = K;

        auto best  = mc_multi_asset(p, PayoffType::BestOfCall, num_paths);
        auto worst = mc_multi_asset(p, PayoffType::WorstOfCall, num_paths);
        auto basket = mc_multi_asset(p, PayoffType::BasketCall, num_paths);

        std::cout << std::setw(8) << rho
                  << std::setw(14) << best.price
                  << std::setw(14) << worst.price
                  << std::setw(14) << basket.price
                  << std::setw(14) << bs_single << "\n";
    }
    std::cout << "\nCorrelation trade insight:\n"
              << "  - Best-of DECREASES with rho (less dispersion)\n"
              << "  - Worst-of INCREASES with rho (less chance of one lagging)\n"
              << "  - Basket INCREASES with rho (less diversification)\n"
              << "  - At rho=1: best-of = worst-of = single BS (all assets identical)\n";

    // ================================================================
    // 4. Spread Option
    // ================================================================
    std::cout << "\n=== 4. Spread Call: max(S1 - S2 - K, 0) ===\n\n";

    std::cout << std::setw(10) << "Strike"
              << std::setw(14) << "Spread Call"
              << std::setw(10) << "SE"
              << std::setw(14) << "Margrabe(K=0)" << "\n";
    std::cout << std::string(48, '-') << "\n";

    double rho_spread = 0.5;
    auto marg_ref = margrabe_price(S0, S0, sigma, sigma, rho_spread, T);

    for (double spread_K : {0.0, 5.0, 10.0, 20.0, 30.0}) {
        MultiAssetParams p;
        p.S0 = {S0, S0};
        p.sigma = {sigma, sigma};
        p.weights = {1.0, 1.0};
        p.corr = uniform_corr_matrix(2, rho_spread);
        p.r = r; p.T = T; p.K = spread_K;

        auto mc = mc_multi_asset(p, PayoffType::SpreadCall, num_paths);
        std::cout << std::setw(10) << spread_K
                  << std::setw(14) << mc.price
                  << std::setw(10) << mc.std_error
                  << std::setw(14) << (spread_K == 0.0 ? marg_ref.price : 0.0) << "\n";
    }

    // ================================================================
    // 5. Diversification Effect — Basket Vol vs N
    // ================================================================
    std::cout << "\n=== 5. Diversification: Basket Vol as N Increases ===\n\n";

    std::cout << std::setw(10) << "N assets"
              << std::setw(14) << "rho=0.0"
              << std::setw(14) << "rho=0.3"
              << std::setw(14) << "rho=0.5"
              << std::setw(14) << "rho=0.8"
              << std::setw(14) << "rho=1.0" << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (size_t N : {2, 5, 10, 20, 50, 100}) {
        std::cout << std::setw(10) << N;
        for (double rho : {0.0, 0.3, 0.5, 0.8, 1.0}) {
            auto params = make_basket_params(N, S0, sigma, rho, r, T, K);
            std::cout << std::setw(14) << basket_vol(params);
        }
        std::cout << "\n";
    }
    std::cout << "\nLimit: basket_vol -> sigma * sqrt(rho) as N -> infinity\n"
              << "  rho=0: vol -> 0 (full diversification)\n"
              << "  rho=1: vol -> sigma (no diversification)\n"
              << "  rho=0.5: vol -> " << sigma * std::sqrt(0.5) << " = sigma*sqrt(0.5)\n";

    return 0;
}

// ============================================================================
// QuantPricer — Visualization Data Generator
//
// Generates CSV files for all key visual metrics:
//   1. Volatility Surface (Heston implied vol across strikes & maturities)
//   2. Greeks Dashboard (Delta, Gamma, Vega, Theta vs Spot)
//   3. Monte Carlo Convergence (price vs num_paths)
//   4. Barrier Analysis (price & knock prob vs barrier level)
//   5. Correlation Impact (multi-asset prices vs rho)
//   6. FDM Solution Grid (option value vs spot)
//   7. American vs European (early exercise premium vs spot)
//   8. Model Comparison (BS vs Heston vs Merton across strikes)
//   9. Payoff Profiles (call, put, spread, straddle)
//  10. Term Structure of Basket Vol
// ============================================================================

#include "payoff/payoff.h"
#include "option/option.h"
#include "greeks/black_scholes.h"
#include "greeks/greeks_engine.h"
#include "mc/monte_carlo.h"
#include "mc/american_mc.h"
#include "vol/implied_vol.h"
#include "fdm/fdm.h"
#include "tree/binomial_tree.h"
#include "barrier/barrier.h"
#include "multi_asset/multi_asset.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;

// ============================================================================
// 1. Volatility Surface — Heston implied vols
// ============================================================================
void gen_vol_surface() {
    std::cout << "  [1/10] Volatility surface..." << std::flush;
    std::ofstream ofs("viz_vol_surface.csv");
    ofs << "Strike,Maturity,ImpliedVol,HestonPrice\n";

    HestonParams hp;
    hp.S0 = 100; hp.r = 0.05;
    hp.v0 = 0.04; hp.kappa = 2.0; hp.theta = 0.04; hp.xi = 0.3; hp.rho = -0.7;

    std::vector<double> strikes = {75,80,85,90,95,100,105,110,115,120,125};
    std::vector<double> mats = {0.25, 0.5, 0.75, 1.0, 1.5, 2.0};

    for (double T : mats) {
        for (double K : strikes) {
            hp.K = K; hp.T = T;
            PayOffCall call(K);
            auto mc = mc_heston(call, hp, 50000);
            auto iv = implied_vol_newton(mc.price, hp.S0, K, hp.r, T);
            if (iv.converged && iv.sigma > 0.01 && iv.sigma < 1.0) {
                ofs << K << "," << T << "," << iv.sigma << "," << mc.price << "\n";
            }
        }
    }
    std::cout << " done\n";
}

// ============================================================================
// 2. Greeks Dashboard — all Greeks vs Spot
// ============================================================================
void gen_greeks() {
    std::cout << "  [2/10] Greeks dashboard..." << std::flush;
    std::ofstream ofs("viz_greeks.csv");
    ofs << "Spot,CallPrice,PutPrice,DeltaCall,DeltaPut,Gamma,Vega,ThetaCall,ThetaPut,RhoCall\n";

    double K=100, r=0.05, T=1.0, sigma=0.2;

    for (double S = 50; S <= 150; S += 0.5) {
        double cp = bs::call_price(S, K, r, T, sigma);
        double pp = bs::put_price(S, K, r, T, sigma);
        double dc = bs::delta_call(S, K, r, T, sigma);
        double dp = bs::delta_put(S, K, r, T, sigma);
        double g  = bs::gamma(S, K, r, T, sigma);
        double v  = bs::vega(S, K, r, T, sigma);
        double tc = bs::theta_call(S, K, r, T, sigma);
        double tp = bs::theta_put(S, K, r, T, sigma);
        double rc = bs::rho_call(S, K, r, T, sigma);
        ofs << S << "," << cp << "," << pp << ","
            << dc << "," << dp << "," << g << "," << v << ","
            << tc << "," << tp << "," << rc << "\n";
    }
    std::cout << " done\n";
}

// ============================================================================
// 3. Monte Carlo Convergence
// ============================================================================
void gen_mc_convergence() {
    std::cout << "  [3/10] MC convergence..." << std::flush;
    std::ofstream ofs("viz_mc_convergence.csv");
    ofs << "NumPaths,MCPrice,StdError,ConfLo,ConfHi,AnalyticPrice\n";

    double S=100, K=100, r=0.05, T=1.0, sigma=0.2;
    PayOffCall call(K);
    double analytic = bs::call_price(S, K, r, T, sigma);

    for (size_t paths = 100; paths <= 800000; paths = static_cast<size_t>(paths * 1.3)) {
        auto mc = mc_european(call, S, r, T, sigma, paths);
        ofs << paths << "," << mc.price << "," << mc.std_error << ","
            << mc.confidence_lo << "," << mc.confidence_hi << ","
            << analytic << "\n";
    }
    std::cout << " done\n";
}

// ============================================================================
// 4. Barrier Analysis
// ============================================================================
void gen_barrier() {
    std::cout << "  [4/10] Barrier analysis..." << std::flush;
    std::ofstream ofs("viz_barrier.csv");
    ofs << "Barrier,DOCallPrice,DICallPrice,VanillaCall,KnockPct\n";

    double S=100, K=100, r=0.05, T=1.0, sigma=0.2;
    double vanilla = bs::call_price(S, K, r, T, sigma);

    for (double H = 50; H <= 99.5; H += 1.0) {
        barrier::BarrierParams doc(S, K, H, r, T, sigma,
            barrier::BarrierType::DownAndOut, barrier::OptionType::Call);
        barrier::BarrierParams dic(S, K, H, r, T, sigma,
            barrier::BarrierType::DownAndIn, barrier::OptionType::Call);

        double do_price = barrier::barrier_price(doc);
        double di_price = barrier::barrier_price(dic);
        auto mc = barrier::mc_barrier(doc, 100000, 252);

        ofs << H << "," << do_price << "," << di_price << ","
            << vanilla << "," << mc.knock_pct << "\n";
    }
    std::cout << " done\n";
}

// ============================================================================
// 5. Correlation Impact — multi-asset
// ============================================================================
void gen_correlation() {
    std::cout << "  [5/10] Correlation impact..." << std::flush;
    std::ofstream ofs("viz_correlation.csv");
    ofs << "Rho,BasketCall,BestOfCall,WorstOfCall,ExchangeOption,BasketVol\n";

    double S0=100, K=100, r=0.05, T=1.0, sigma=0.2;

    for (double rho = -0.4; rho <= 0.99; rho += 0.05) {
        auto params = multi_asset::make_basket_params(2, S0, sigma, rho, r, T, K);
        auto basket = multi_asset::mc_multi_asset(params, multi_asset::PayoffType::BasketCall, 80000);
        auto best   = multi_asset::mc_multi_asset(params, multi_asset::PayoffType::BestOfCall, 80000);
        auto worst  = multi_asset::mc_multi_asset(params, multi_asset::PayoffType::WorstOfCall, 80000);

        multi_asset::MultiAssetParams ep;
        ep.S0 = {S0, S0}; ep.sigma = {sigma, sigma};
        ep.weights = {1.0, 1.0};
        ep.corr = multi_asset::uniform_corr_matrix(2, rho);
        ep.r = r; ep.T = T; ep.K = 0.0;
        auto exch = multi_asset::mc_multi_asset(ep, multi_asset::PayoffType::Exchange, 80000);

        double bvol = multi_asset::basket_vol(params);

        ofs << rho << "," << basket.price << "," << best.price << ","
            << worst.price << "," << exch.price << "," << bvol << "\n";
    }
    std::cout << " done\n";
}

// ============================================================================
// 6. FDM Solution Grid
// ============================================================================
void gen_fdm_grid() {
    std::cout << "  [6/10] FDM solution grid..." << std::flush;
    std::ofstream ofs("viz_fdm_grid.csv");
    ofs << "Spot,FDMCall,FDMPut,AnalyticCall,AnalyticPut,Intrinsic\n";

    double S0=100, K=100, r=0.05, T=1.0, sigma=0.2;

    PayOffCall call_payoff(K);
    PayOffPut put_payoff(K);
    BlackScholesPDE pde(r, sigma);
    FDMParams params(400, 2000, 300.0, 0.5);

    FDMSolver solver_call(call_payoff, pde, S0, T, params);
    FDMSolver solver_put(put_payoff, pde, S0, T, params);
    auto result_call = solver_call.solve();
    auto result_put = solver_put.solve();

    for (size_t i = 0; i <= params.N_space; i += 2) {
        double S = result_call.spot_grid[i];
        if (S < 1 || S > 200) continue;
        double ac = bs::call_price(S, K, r, T, sigma);
        double ap = bs::put_price(S, K, r, T, sigma);
        double intr = std::max(S - K, 0.0);
        ofs << S << "," << result_call.option_values[i] << ","
            << result_put.option_values[i] << ","
            << ac << "," << ap << "," << intr << "\n";
    }
    std::cout << " done\n";
}

// ============================================================================
// 7. American vs European — Early Exercise Premium
// ============================================================================
void gen_american() {
    std::cout << "  [7/10] American vs European..." << std::flush;
    std::ofstream ofs("viz_american.csv");
    ofs << "Spot,EurPut,AmePut,EEPremium,EurCall,AmeCall\n";

    double K=100, r=0.05, T=1.0, sigma=0.2;
    PayOffPut put(K);
    PayOffCall call(K);

    for (double S = 60; S <= 140; S += 1.0) {
        auto eur_p = binomial_european(put, S, K, r, T, sigma, 1000);
        auto ame_p = binomial_american(put, S, K, r, T, sigma, 1000);
        auto eur_c = binomial_european(call, S, K, r, T, sigma, 1000);
        auto ame_c = binomial_american(call, S, K, r, T, sigma, 1000);
        double eep = ame_p.price - eur_p.price;

        ofs << S << "," << eur_p.price << "," << ame_p.price << ","
            << eep << "," << eur_c.price << "," << ame_c.price << "\n";
    }
    std::cout << " done\n";
}

// ============================================================================
// 8. Model Comparison — BS vs Heston vs Merton across strikes
// ============================================================================
void gen_model_comparison() {
    std::cout << "  [8/10] Model comparison..." << std::flush;
    std::ofstream ofs("viz_model_comparison.csv");
    ofs << "Strike,BSPrice,BSImpliedVol,HestonPrice,HestonIV,MertonPrice,MertonIV\n";

    double S=100, r=0.05, T=1.0, sigma=0.2;

    HestonParams hp;
    hp.S0=S; hp.r=r; hp.T=T;
    hp.v0=0.04; hp.kappa=2.0; hp.theta=0.04; hp.xi=0.3; hp.rho=-0.7;

    MertonJumpParams jp;
    jp.S0=S; jp.r=r; jp.T=T; jp.sigma=sigma;
    jp.lambda=1.0; jp.mu_j=-0.1; jp.sigma_j=0.15;

    for (double K = 70; K <= 130; K += 2.0) {
        PayOffCall call(K);
        double bs_price = bs::call_price(S, K, r, T, sigma);

        hp.K = K;
        auto heston = mc_heston(call, hp, 80000);
        jp.K = K;
        auto merton = mc_merton_jump(call, jp, 80000);

        auto hiv = implied_vol_newton(heston.price, S, K, r, T);
        auto miv = implied_vol_newton(merton.price, S, K, r, T);

        ofs << K << "," << bs_price << "," << sigma << ","
            << heston.price << "," << (hiv.converged ? hiv.sigma : 0.0) << ","
            << merton.price << "," << (miv.converged ? miv.sigma : 0.0) << "\n";
    }
    std::cout << " done\n";
}

// ============================================================================
// 9. Payoff Profiles
// ============================================================================
void gen_payoffs() {
    std::cout << "  [9/10] Payoff profiles..." << std::flush;
    std::ofstream ofs("viz_payoffs.csv");
    ofs << "Spot,CallPayoff,PutPayoff,Straddle,CallValue,PutValue,StraddleValue\n";

    double K=100, r=0.05, T=1.0, sigma=0.2;

    for (double S = 50; S <= 150; S += 0.5) {
        double call_payoff = std::max(S - K, 0.0);
        double put_payoff = std::max(K - S, 0.0);
        double straddle = call_payoff + put_payoff;
        double call_val = bs::call_price(S, K, r, T, sigma);
        double put_val = bs::put_price(S, K, r, T, sigma);
        double straddle_val = call_val + put_val;

        ofs << S << "," << call_payoff << "," << put_payoff << ","
            << straddle << "," << call_val << "," << put_val << ","
            << straddle_val << "\n";
    }
    std::cout << " done\n";
}

// ============================================================================
// 10. Basket Vol Term Structure
// ============================================================================
void gen_basket_vol() {
    std::cout << "  [10/10] Basket vol term structure..." << std::flush;
    std::ofstream ofs("viz_basket_vol.csv");
    ofs << "N,Rho0,Rho03,Rho05,Rho08,Rho1\n";

    double sigma = 0.2;
    for (size_t N : {2,3,5,7,10,15,20,30,50,75,100}) {
        ofs << N;
        for (double rho : {0.0, 0.3, 0.5, 0.8, 1.0}) {
            auto params = multi_asset::make_basket_params(N, 100, sigma, rho, 0.05, 1.0, 100);
            ofs << "," << multi_asset::basket_vol(params);
        }
        ofs << "\n";
    }
    std::cout << " done\n";
}

// ============================================================================
// Main
// ============================================================================
int main() {
    auto t0 = Clock::now();

    std::cout << "======================================================================\n"
              << "  QuantPricer — Generating Visualization Data\n"
              << "======================================================================\n\n";

    gen_vol_surface();
    gen_greeks();
    gen_mc_convergence();
    gen_barrier();
    gen_correlation();
    gen_fdm_grid();
    gen_american();
    gen_model_comparison();
    gen_payoffs();
    gen_basket_vol();

    auto t1 = Clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\n  All CSV files generated in " << std::fixed << std::setprecision(1)
              << secs << "s\n"
              << "  Run: python3 tools/visualize.py\n"
              << "======================================================================\n";

    return 0;
}

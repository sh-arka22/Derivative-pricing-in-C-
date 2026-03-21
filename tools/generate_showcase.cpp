// ============================================================================
// QuantPricer — GitHub Showcase Data Generator
//
// Generates all results with 2-3 input variations per analysis.
// Output: CSV files in showcase_data/ directory
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
#include "risk/risk.h"
#include "fixed_income/fixed_income.h"
#include "rates/rate_models.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <sys/stat.h>

using Clock = std::chrono::high_resolution_clock;

void ensure_dir(const char* path) { mkdir(path, 0755); }

// ============================================================================
// 1. Volatility Surface — 3 rho variations
// ============================================================================
void gen_vol_surface() {
    std::cout << "  [1/12] Vol surface (3 rho variations)..." << std::flush;

    for (double rho : {-0.9, -0.5, 0.0}) {
        std::string fname = "showcase_data/vol_surface_rho_" +
            std::to_string(static_cast<int>(rho * 10)) + ".csv";
        std::ofstream ofs(fname);
        ofs << "Strike,Maturity,ImpliedVol\n";

        HestonParams hp;
        hp.S0 = 100; hp.r = 0.05;
        hp.v0 = 0.04; hp.kappa = 2.0; hp.theta = 0.04; hp.xi = 0.3; hp.rho = rho;

        for (double T : {0.25, 0.5, 0.75, 1.0, 1.5, 2.0}) {
            for (double K = 75; K <= 125; K += 2.5) {
                hp.K = K; hp.T = T;
                PayOffCall call(K);
                auto mc = mc_heston(call, hp, 40000);
                auto iv = implied_vol_newton(mc.price, hp.S0, K, hp.r, T);
                if (iv.converged && iv.sigma > 0.01 && iv.sigma < 1.0)
                    ofs << K << "," << T << "," << iv.sigma << "\n";
            }
        }
    }
    std::cout << " done\n";
}

// ============================================================================
// 2. Greeks — Call vs Put
// ============================================================================
void gen_greeks() {
    std::cout << "  [2/12] Greeks (call & put)..." << std::flush;
    std::ofstream ofs("showcase_data/greeks.csv");
    ofs << "Spot,CallPrice,PutPrice,DeltaCall,DeltaPut,Gamma,Vega,ThetaCall,ThetaPut\n";

    double K=100, r=0.05, T=1.0, sigma=0.2;
    for (double S = 50; S <= 150; S += 0.5) {
        ofs << S << ","
            << bs::call_price(S,K,r,T,sigma) << ","
            << bs::put_price(S,K,r,T,sigma) << ","
            << bs::delta_call(S,K,r,T,sigma) << ","
            << bs::delta_put(S,K,r,T,sigma) << ","
            << bs::gamma(S,K,r,T,sigma) << ","
            << bs::vega(S,K,r,T,sigma) << ","
            << bs::theta_call(S,K,r,T,sigma) << ","
            << bs::theta_put(S,K,r,T,sigma) << "\n";
    }
    std::cout << " done\n";
}

// ============================================================================
// 3. MC Convergence — 3 volatility levels
// ============================================================================
void gen_mc_convergence() {
    std::cout << "  [3/12] MC convergence (3 vol levels)..." << std::flush;
    std::ofstream ofs("showcase_data/mc_convergence.csv");
    ofs << "NumPaths,Sigma,MCPrice,StdError,ConfLo,ConfHi,Analytic\n";

    double S=100, K=100, r=0.05, T=1.0;
    for (double sigma : {0.15, 0.25, 0.40}) {
        PayOffCall call(K);
        double analytic = bs::call_price(S, K, r, T, sigma);
        for (size_t paths = 200; paths <= 500000; paths = static_cast<size_t>(paths * 1.6)) {
            auto mc = mc_european(call, S, r, T, sigma, paths);
            ofs << paths << "," << sigma << "," << mc.price << ","
                << mc.std_error << "," << mc.confidence_lo << ","
                << mc.confidence_hi << "," << analytic << "\n";
        }
    }
    std::cout << " done\n";
}

// ============================================================================
// 4. Model Comparison — BS vs Heston vs Merton
// ============================================================================
void gen_model_comparison() {
    std::cout << "  [4/12] Model comparison..." << std::flush;
    std::ofstream ofs("showcase_data/model_comparison.csv");
    ofs << "Strike,BSPrice,BSIV,HestonPrice,HestonIV,MertonPrice,MertonIV\n";

    double S=100, r=0.05, T=1.0, sigma=0.2;
    HestonParams hp;
    hp.S0=S; hp.r=r; hp.T=T; hp.v0=0.04; hp.kappa=2.0; hp.theta=0.04; hp.xi=0.3; hp.rho=-0.7;
    MertonJumpParams jp;
    jp.S0=S; jp.r=r; jp.T=T; jp.sigma=sigma; jp.lambda=1.0; jp.mu_j=-0.1; jp.sigma_j=0.15;

    for (double K = 70; K <= 130; K += 2) {
        PayOffCall call(K);
        hp.K = K; jp.K = K;
        auto heston = mc_heston(call, hp, 60000);
        auto merton = mc_merton_jump(call, jp, 60000);
        auto hiv = implied_vol_newton(heston.price, S, K, r, T);
        auto miv = implied_vol_newton(merton.price, S, K, r, T);

        ofs << K << "," << bs::call_price(S,K,r,T,sigma) << "," << sigma << ","
            << heston.price << "," << (hiv.converged ? hiv.sigma : 0.0) << ","
            << merton.price << "," << (miv.converged ? miv.sigma : 0.0) << "\n";
    }
    std::cout << " done\n";
}

// ============================================================================
// 5. Barrier Options — 3 vol levels
// ============================================================================
void gen_barrier() {
    std::cout << "  [5/12] Barrier options (3 vol levels)..." << std::flush;
    std::ofstream ofs("showcase_data/barrier.csv");
    ofs << "Barrier,Sigma,DOCall,DICall,Vanilla,KnockPct\n";

    double S=100, K=100, r=0.05, T=1.0;
    for (double sigma : {0.15, 0.25, 0.35}) {
        double vanilla = bs::call_price(S, K, r, T, sigma);
        for (double H = 55; H <= 99; H += 2) {
            barrier::BarrierParams doc(S, K, H, r, T, sigma,
                barrier::BarrierType::DownAndOut, barrier::OptionType::Call);
            barrier::BarrierParams dic(S, K, H, r, T, sigma,
                barrier::BarrierType::DownAndIn, barrier::OptionType::Call);
            auto mc = barrier::mc_barrier(doc, 50000, 252);
            ofs << H << "," << sigma << ","
                << barrier::barrier_price(doc) << ","
                << barrier::barrier_price(dic) << ","
                << vanilla << "," << mc.knock_pct << "\n";
        }
    }
    std::cout << " done\n";
}

// ============================================================================
// 6. Correlation — 2 vs 5 assets
// ============================================================================
void gen_correlation() {
    std::cout << "  [6/12] Correlation (2 & 5 assets)..." << std::flush;
    std::ofstream ofs("showcase_data/correlation.csv");
    ofs << "Rho,N,BasketCall,BestOfCall,WorstOfCall,BasketVol\n";

    double S0=100, K=100, r=0.05, T=1.0, sigma=0.2;
    for (int N : {2, 5}) {
        double rho_min = (N == 2) ? -0.3 : -0.15;  // min rho for PSD: > -1/(N-1)
        for (double rho = rho_min; rho <= 0.98; rho += 0.04) {
            auto params = multi_asset::make_basket_params(N, S0, sigma, rho, r, T, K);
            auto basket = multi_asset::mc_multi_asset(params, multi_asset::PayoffType::BasketCall, 60000);
            auto best = multi_asset::mc_multi_asset(params, multi_asset::PayoffType::BestOfCall, 60000);
            auto worst = multi_asset::mc_multi_asset(params, multi_asset::PayoffType::WorstOfCall, 60000);
            ofs << rho << "," << N << "," << basket.price << ","
                << best.price << "," << worst.price << ","
                << multi_asset::basket_vol(params) << "\n";
        }
    }
    std::cout << " done\n";
}

// ============================================================================
// 7. American vs European — 3 rate levels
// ============================================================================
void gen_american() {
    std::cout << "  [7/12] American options (3 rate levels)..." << std::flush;
    std::ofstream ofs("showcase_data/american.csv");
    ofs << "Spot,Rate,EurPut,AmePut,EEPremium\n";

    double K=100, T=1.0, sigma=0.2;
    PayOffPut put(K);
    for (double r : {0.02, 0.05, 0.10}) {
        for (double S = 60; S <= 140; S += 1) {
            auto eur = binomial_european(put, S, K, r, T, sigma, 800);
            auto ame = binomial_american(put, S, K, r, T, sigma, 800);
            ofs << S << "," << r << "," << eur.price << ","
                << ame.price << "," << (ame.price - eur.price) << "\n";
        }
    }
    std::cout << " done\n";
}

// ============================================================================
// 8. FDM vs Analytic
// ============================================================================
void gen_fdm() {
    std::cout << "  [8/12] FDM solution..." << std::flush;
    std::ofstream ofs("showcase_data/fdm.csv");
    ofs << "Spot,FDMCall,FDMPut,AnalyticCall,AnalyticPut\n";

    double K=100, r=0.05, T=1.0, sigma=0.2;
    PayOffCall call_po(K); PayOffPut put_po(K);
    BlackScholesPDE pde(r, sigma);
    FDMParams params(400, 2000, 300.0, 0.5);
    FDMSolver sc(call_po, pde, 100, T, params);
    FDMSolver sp(put_po, pde, 100, T, params);
    auto rc = sc.solve(); auto rp = sp.solve();

    for (size_t i = 0; i <= params.N_space; i += 2) {
        double S = rc.spot_grid[i];
        if (S < 1 || S > 200) continue;
        ofs << S << "," << rc.option_values[i] << "," << rp.option_values[i] << ","
            << bs::call_price(S,K,r,T,sigma) << "," << bs::put_price(S,K,r,T,sigma) << "\n";
    }
    std::cout << " done\n";
}

// ============================================================================
// 9. Yield Curves — 3 shapes
// ============================================================================
void gen_yield_curves() {
    std::cout << "  [9/12] Yield curves (3 shapes)..." << std::flush;
    std::ofstream ofs("showcase_data/yield_curves.csv");
    ofs << "Tenor,Normal,Flat,Inverted\n";

    fixed_income::NelsonSiegelParams normal(0.045, -0.015, 0.01, 2.0);
    fixed_income::NelsonSiegelParams flat(0.04, 0.0, 0.0, 2.0);
    fixed_income::NelsonSiegelParams inv(0.03, 0.02, -0.01, 2.0);
    auto cn = fixed_income::build_nelson_siegel_curve(normal);
    auto cf = fixed_income::build_nelson_siegel_curve(flat);
    auto ci = fixed_income::build_nelson_siegel_curve(inv);

    for (double t = 0.25; t <= 30.0; t += 0.25) {
        ofs << t << "," << cn.zero_rate(t) << ","
            << cf.zero_rate(t) << "," << ci.zero_rate(t) << "\n";
    }
    std::cout << " done\n";
}

// ============================================================================
// 10. Rate Models — Vasicek vs CIR
// ============================================================================
void gen_rate_models() {
    std::cout << "  [10/12] Rate models (Vasicek vs CIR)..." << std::flush;
    std::ofstream ofs("showcase_data/rate_models.csv");
    ofs << "Tenor,VasicekRate,CIRRate,VasicekBond,CIRBond\n";

    rates::VasicekParams vp(0.03, 0.3, 0.06, 0.02);
    rates::CIRParams cp(0.03, 0.3, 0.06, 0.08);

    for (double T = 0.25; T <= 30; T += 0.25) {
        ofs << T << ","
            << rates::vasicek_zero_rate(T, vp) << ","
            << rates::cir_zero_rate(T, cp) << ","
            << rates::vasicek_bond_price(T, vp) << ","
            << rates::cir_bond_price(T, cp) << "\n";
    }
    std::cout << " done\n";
}

// ============================================================================
// 11. Payoff Profiles
// ============================================================================
void gen_payoffs() {
    std::cout << "  [11/12] Payoff profiles..." << std::flush;
    std::ofstream ofs("showcase_data/payoffs.csv");
    ofs << "Spot,CallPayoff,PutPayoff,Straddle,CallValue,PutValue,StraddleValue\n";

    double K=100, r=0.05, T=1.0, sigma=0.2;
    for (double S = 50; S <= 150; S += 0.5) {
        double cp = std::max(S-K, 0.0), pp = std::max(K-S, 0.0);
        double cv = bs::call_price(S,K,r,T,sigma), pv = bs::put_price(S,K,r,T,sigma);
        ofs << S << "," << cp << "," << pp << "," << (cp+pp) << ","
            << cv << "," << pv << "," << (cv+pv) << "\n";
    }
    std::cout << " done\n";
}

// ============================================================================
// 12. Basket Vol Diversification
// ============================================================================
void gen_basket_vol() {
    std::cout << "  [12/12] Basket vol..." << std::flush;
    std::ofstream ofs("showcase_data/basket_vol.csv");
    ofs << "N,Rho0,Rho03,Rho05,Rho08,Rho1\n";

    for (size_t N : {2,3,5,7,10,15,20,30,50,75,100}) {
        ofs << N;
        for (double rho : {0.0, 0.3, 0.5, 0.8, 1.0}) {
            auto p = multi_asset::make_basket_params(N, 100, 0.2, rho, 0.05, 1.0, 100);
            ofs << "," << multi_asset::basket_vol(p);
        }
        ofs << "\n";
    }
    std::cout << " done\n";
}

// ============================================================================
// Results Summary Table
// ============================================================================
void gen_results_table() {
    std::ofstream ofs("showcase_data/results_summary.csv");
    ofs << "Metric,Value\n";

    double S=100, K=100, r=0.05, T=1.0, sigma=0.2;

    // BS prices
    ofs << "BS Call Price," << bs::call_price(S,K,r,T,sigma) << "\n";
    ofs << "BS Put Price," << bs::put_price(S,K,r,T,sigma) << "\n";
    ofs << "Put-Call Parity Error," << std::abs(bs::call_price(S,K,r,T,sigma)
        - bs::put_price(S,K,r,T,sigma) - S + K*std::exp(-r*T)) << "\n";

    // Greeks
    ofs << "Delta (Call)," << bs::delta_call(S,K,r,T,sigma) << "\n";
    ofs << "Gamma," << bs::gamma(S,K,r,T,sigma) << "\n";
    ofs << "Vega," << bs::vega(S,K,r,T,sigma) << "\n";

    // MC
    PayOffCall call(K);
    auto mc = mc_european(call, S, r, T, sigma, 500000);
    ofs << "MC Price (500k paths)," << mc.price << "\n";
    ofs << "MC Std Error," << mc.std_error << "\n";

    // Heston
    HestonParams hp; hp.S0=S; hp.K=K; hp.r=r; hp.T=T;
    auto hest = mc_heston(call, hp, 100000);
    ofs << "Heston Price," << hest.price << "\n";

    // FDM
    PayOffCall cpo(K); BlackScholesPDE pde(r,sigma);
    FDMParams fp(400,2000,300,0.5);
    FDMSolver solver(cpo,pde,S,T,fp);
    auto fdm = solver.solve();
    ofs << "FDM C-N Price," << fdm.price_at_spot << "\n";
    ofs << "FDM Error vs BS," << std::abs(fdm.price_at_spot - bs::call_price(S,K,r,T,sigma)) << "\n";

    // Tree
    auto tree = binomial_european(call, S, K, r, T, sigma, 1000);
    ofs << "Tree Price (N=1000)," << tree.price << "\n";

    // American
    PayOffPut put(K);
    auto ame = binomial_american(put, S, K, r, T, sigma, 1000);
    auto eur = binomial_european(put, S, K, r, T, sigma, 1000);
    ofs << "American Put," << ame.price << "\n";
    ofs << "European Put," << eur.price << "\n";
    ofs << "Early Exercise Premium," << (ame.price - eur.price) << "\n";

    // Barrier
    barrier::BarrierParams doc(S, K, 90, r, T, sigma,
        barrier::BarrierType::DownAndOut, barrier::OptionType::Call);
    ofs << "DO Call (H=90)," << barrier::barrier_price(doc) << "\n";

    // Margrabe
    auto marg = multi_asset::margrabe_price(100, 100, 0.2, 0.3, 0.5, 1.0);
    ofs << "Margrabe Exchange," << marg.price << "\n";
}

// ============================================================================
int main() {
    auto t0 = Clock::now();

    std::cout << "======================================================================\n"
              << "  QuantPricer — Generating Showcase Data\n"
              << "======================================================================\n\n";

    ensure_dir("showcase_data");

    gen_vol_surface();
    gen_greeks();
    gen_mc_convergence();
    gen_model_comparison();
    gen_barrier();
    gen_correlation();
    gen_american();
    gen_fdm();
    gen_yield_curves();
    gen_rate_models();
    gen_payoffs();
    gen_basket_vol();
    gen_results_table();

    auto t1 = Clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\n  All showcase data generated in " << std::fixed
              << std::setprecision(1) << secs << "s\n"
              << "  Run: python3 tools/render_showcase.py\n"
              << "======================================================================\n";
    return 0;
}

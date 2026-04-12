// Bates model — Heston stochastic vol + Merton jumps (Bates, 1996)
// Generates CSV data for visualization + console output
#include "mc/monte_carlo.h"
#include "greeks/black_scholes.h"
#include "vol/implied_vol.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

int main() {
    std::cout << std::fixed << std::setprecision(4);
    PayOffCall call(100);

    // ---------------------------------------------------------------
    // 1. Nesting test: verify Bates reduces to Heston when lambda=0
    // ---------------------------------------------------------------
    std::cout << "=== Nesting: Bates(lambda=0) vs Heston ===\n";
    BatesParams bp;
    bp.lambda = 0.0;
    HestonParams hp;
    auto bates_no_jump = mc_bates(call, bp, 200000);
    auto heston        = mc_heston(call, hp, 200000);
    std::cout << "Bates(lambda=0): " << bates_no_jump.price
              << "  SE=" << bates_no_jump.std_error << "\n"
              << "Heston:          " << heston.price
              << "  SE=" << heston.std_error << "\n\n";

    // ---------------------------------------------------------------
    // 2. Jump intensity sweep — CSV + console
    // ---------------------------------------------------------------
    std::cout << "=== Bates: varying jump intensity (lambda) ===\n";
    std::cout << "Lambda  Price   SE\n";
    {
        std::ofstream ofs("bates_lambda_sweep.csv");
        ofs << "Lambda,BatesPrice,StdError,ConfLo,ConfHi,HestonPrice\n";
        double heston_price = mc_heston(call, hp, 200000).price;
        for (double lam : {0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0}) {
            BatesParams p;
            p.lambda = lam;
            auto res = mc_bates(call, p, 200000);
            ofs << lam << "," << res.price << "," << res.std_error
                << "," << res.confidence_lo << "," << res.confidence_hi
                << "," << heston_price << "\n";
            std::cout << std::setw(6) << lam << "  "
                      << std::setw(7) << res.price << "  "
                      << std::setw(7) << res.std_error << "\n";
        }
    }

    // ---------------------------------------------------------------
    // 3. Smile comparison across strikes — CSV + console
    //    BS vs Heston vs Merton vs Bates (prices + implied vols)
    // ---------------------------------------------------------------
    std::cout << "\n=== Smile: BS vs Heston vs Merton vs Bates across strikes ===\n";
    std::cout << "Strike  BS      Heston  Merton  Bates\n";
    {
        std::ofstream ofs("bates_smile.csv");
        ofs << "Strike,BSPrice,HestonPrice,MertonPrice,BatesPrice,"
            << "BSIV,HestonIV,MertonIV,BatesIV\n";

        BatesParams bp_full;
        MertonJumpParams mp;
        mp.sigma = std::sqrt(hp.v0);  // match diffusion vol to Heston's v0
        mp.lambda = bp_full.lambda;
        mp.mu_j = bp_full.mu_j;
        mp.sigma_j = bp_full.sigma_j;

        double bs_flat_vol = std::sqrt(hp.v0);  // 0.2

        for (double K : {70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0,
                         105.0, 110.0, 115.0, 120.0, 125.0, 130.0}) {
            PayOffCall c(K);
            hp.K = K;  bp_full.K = K;  mp.K = K;

            double bs_p = bs::call_price(hp.S0, K, hp.r, hp.T, bs_flat_vol);
            auto h = mc_heston(c, hp, 300000);
            auto m = mc_merton_jump(c, mp, 300000);
            auto b = mc_bates(c, bp_full, 300000);

            // Implied vols via Newton-Raphson
            double bs_iv  = bs_flat_vol;  // trivially 20%
            double h_iv   = implied_vol_newton(h.price, hp.S0, K, hp.r, hp.T).sigma;
            double m_iv   = implied_vol_newton(m.price, hp.S0, K, hp.r, hp.T).sigma;
            double b_iv   = implied_vol_newton(b.price, hp.S0, K, hp.r, hp.T).sigma;

            ofs << K << "," << bs_p << "," << h.price << "," << m.price << "," << b.price
                << "," << bs_iv << "," << h_iv << "," << m_iv << "," << b_iv << "\n";

            std::cout << std::setw(6) << K << "  "
                      << std::setw(7) << bs_p << "  "
                      << std::setw(7) << h.price << "  "
                      << std::setw(7) << m.price << "  "
                      << std::setw(7) << b.price << "\n";
        }
    }

    // ---------------------------------------------------------------
    // 4. Jump size impact — vary mu_j at fixed lambda
    // ---------------------------------------------------------------
    std::cout << "\n=== Bates: varying mean jump size (mu_j) ===\n";
    {
        std::ofstream ofs("bates_jump_size.csv");
        ofs << "Strike,MuJ_n15,MuJ_n10,MuJ_n05,MuJ_0,MuJ_p05,MuJ_p10\n";

        for (double K : {80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0}) {
            PayOffCall c(K);
            ofs << K;
            for (double muj : {-0.15, -0.10, -0.05, 0.0, 0.05, 0.10}) {
                BatesParams p;
                p.K = K;
                p.mu_j = muj;
                double iv = implied_vol_newton(
                    mc_bates(c, p, 200000).price, p.S0, K, p.r, p.T).sigma;
                ofs << "," << iv;
            }
            ofs << "\n";
        }
    }

    std::cout << "\nCSV files written: bates_lambda_sweep.csv, bates_smile.csv, bates_jump_size.csv\n";
    std::cout << "Run: python3 tools/plot_bates.py\n";
}

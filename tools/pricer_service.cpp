// ============================================================================
// QuantPricer C++ Service — stdin/stdout bridge for Python dashboard
//
// Protocol: line-based key-value input, CSV output
// Input:  COMMAND: <name>\nkey: value\n...\n---
// Output: CSV to stdout (headers + data rows)
//
// Supported commands:
//   heston_smile    — Heston MC across strikes → prices + implied vols
//   merton_smile    — Merton jump-diffusion across strikes → prices + IVs
//   barrier_curve   — barrier option price vs barrier level (analytic)
//   multi_asset     — basket/best-of/worst-of MC pricing vs correlation
//   batch_iv        — compute IV for a list of (price,K,T) triples
//   mc_european     — single European MC price
//   vasicek_curve   — Vasicek model yield curve
// ============================================================================

#include "payoff/payoff.h"
#include "option/option.h"
#include "greeks/black_scholes.h"
#include "mc/monte_carlo.h"
#include "vol/implied_vol.h"
#include "barrier/barrier.h"
#include "multi_asset/multi_asset.h"
#include "rates/rate_models.h"

#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <cmath>
#include <cstdlib>

// ============================================================================
// Parse input: read key-value pairs until "---"
// ============================================================================
std::map<std::string, std::string> parse_input() {
    std::map<std::string, std::string> params;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "---" || line.empty()) break;
        auto colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = line.substr(0, colon);
            std::string val = line.substr(colon + 1);
            // Trim whitespace
            while (!key.empty() && key.back() == ' ') key.pop_back();
            while (!val.empty() && val.front() == ' ') val.erase(val.begin());
            params[key] = val;
        }
    }
    return params;
}

double get_d(const std::map<std::string, std::string>& p, const std::string& key, double def = 0.0) {
    auto it = p.find(key);
    return (it != p.end()) ? std::stod(it->second) : def;
}

int get_i(const std::map<std::string, std::string>& p, const std::string& key, int def = 0) {
    auto it = p.find(key);
    return (it != p.end()) ? std::stoi(it->second) : def;
}

// ============================================================================
// Command: heston_smile
// ============================================================================
void cmd_heston_smile(const std::map<std::string, std::string>& p) {
    double S0 = get_d(p, "S0", 100);
    double K_min = get_d(p, "K_min", 70);
    double K_max = get_d(p, "K_max", 130);
    double K_step = get_d(p, "K_step", 2);
    double r = get_d(p, "r", 0.05);
    double T = get_d(p, "T", 1.0);
    size_t num_paths = static_cast<size_t>(get_d(p, "num_paths", 80000));

    HestonParams hp;
    hp.S0 = S0; hp.r = r; hp.T = T;
    hp.v0 = get_d(p, "v0", 0.04);
    hp.kappa = get_d(p, "kappa", 2.0);
    hp.theta = get_d(p, "theta", 0.04);
    hp.xi = get_d(p, "xi", 0.3);
    hp.rho = get_d(p, "rho", -0.7);

    std::cout << "Strike,HestonPrice,ImpliedVol,BSPrice\n";
    for (double K = K_min; K <= K_max + 0.01; K += K_step) {
        hp.K = K;
        PayOffCall call(K);
        auto mc = mc_heston(call, hp, num_paths);
        auto iv = implied_vol_newton(mc.price, S0, K, r, T);
        double bs = bs::call_price(S0, K, r, T, std::sqrt(hp.v0));

        std::cout << K << "," << mc.price << ","
                  << (iv.converged ? iv.sigma : 0.0) << "," << bs << "\n";
    }
}

// ============================================================================
// Command: merton_smile
// ============================================================================
void cmd_merton_smile(const std::map<std::string, std::string>& p) {
    double S0 = get_d(p, "S0", 100);
    double K_min = get_d(p, "K_min", 70);
    double K_max = get_d(p, "K_max", 130);
    double K_step = get_d(p, "K_step", 2);
    size_t num_paths = static_cast<size_t>(get_d(p, "num_paths", 80000));

    MertonJumpParams jp;
    jp.S0 = S0;
    jp.r = get_d(p, "r", 0.05);
    jp.T = get_d(p, "T", 1.0);
    jp.sigma = get_d(p, "sigma", 0.2);
    jp.lambda = get_d(p, "lambda", 1.0);
    jp.mu_j = get_d(p, "mu_j", -0.1);
    jp.sigma_j = get_d(p, "sigma_j", 0.15);

    std::cout << "Strike,MertonPrice,ImpliedVol,BSPrice\n";
    for (double K = K_min; K <= K_max + 0.01; K += K_step) {
        jp.K = K;
        PayOffCall call(K);
        auto mc = mc_merton_jump(call, jp, num_paths);
        auto iv = implied_vol_newton(mc.price, jp.S0, K, jp.r, jp.T);
        double bs = bs::call_price(jp.S0, K, jp.r, jp.T, jp.sigma);

        std::cout << K << "," << mc.price << ","
                  << (iv.converged ? iv.sigma : 0.0) << "," << bs << "\n";
    }
}

// ============================================================================
// Command: barrier_curve
// ============================================================================
void cmd_barrier_curve(const std::map<std::string, std::string>& p) {
    double S = get_d(p, "S0", 100);
    double K = get_d(p, "K", 100);
    double r = get_d(p, "r", 0.05);
    double T = get_d(p, "T", 1.0);
    double sigma = get_d(p, "sigma", 0.2);
    double H_min = get_d(p, "H_min", 50);
    double H_max = get_d(p, "H_max", 99);
    double H_step = get_d(p, "H_step", 1);
    double vanilla = bs::call_price(S, K, r, T, sigma);

    std::cout << "Barrier,DOCall,DICall,Vanilla,KnockPct\n";
    for (double H = H_min; H <= H_max + 0.01; H += H_step) {
        barrier::BarrierParams doc(S, K, H, r, T, sigma,
            barrier::BarrierType::DownAndOut, barrier::OptionType::Call);
        barrier::BarrierParams dic(S, K, H, r, T, sigma,
            barrier::BarrierType::DownAndIn, barrier::OptionType::Call);

        double do_price = barrier::barrier_price(doc);
        double di_price = barrier::barrier_price(dic);
        auto mc = barrier::mc_barrier(doc, 50000, 252);

        std::cout << H << "," << do_price << "," << di_price << ","
                  << vanilla << "," << mc.knock_pct << "\n";
    }
}

// ============================================================================
// Command: multi_asset — correlation sweep
// ============================================================================
void cmd_multi_asset(const std::map<std::string, std::string>& p) {
    double S0 = get_d(p, "S0", 100);
    double K = get_d(p, "K", 100);
    double r = get_d(p, "r", 0.05);
    double T = get_d(p, "T", 1.0);
    double sigma = get_d(p, "sigma", 0.2);
    int N = get_i(p, "N", 2);
    double rho_min = get_d(p, "rho_min", -0.4);
    double rho_max = get_d(p, "rho_max", 0.99);
    double rho_step = get_d(p, "rho_step", 0.05);
    size_t num_paths = static_cast<size_t>(get_d(p, "num_paths", 80000));

    std::cout << "Rho,BasketCall,BestOfCall,WorstOfCall,BasketVol\n";
    for (double rho = rho_min; rho <= rho_max + 0.001; rho += rho_step) {
        auto params = multi_asset::make_basket_params(N, S0, sigma, rho, r, T, K);
        auto basket = multi_asset::mc_multi_asset(params, multi_asset::PayoffType::BasketCall, num_paths);
        auto best = multi_asset::mc_multi_asset(params, multi_asset::PayoffType::BestOfCall, num_paths);
        auto worst = multi_asset::mc_multi_asset(params, multi_asset::PayoffType::WorstOfCall, num_paths);
        double bvol = multi_asset::basket_vol(params);

        std::cout << rho << "," << basket.price << "," << best.price << ","
                  << worst.price << "," << bvol << "\n";
    }
}

// ============================================================================
// Command: batch_iv — compute IV for multiple options
// Input expects: lines of "price,strike,T,is_call" after the header params
// ============================================================================
void cmd_batch_iv(const std::map<std::string, std::string>& p) {
    double S = get_d(p, "S0", 100);
    double r = get_d(p, "r", 0.05);
    int count = get_i(p, "count", 0);

    std::cout << "Strike,T,IV,Type\n";
    for (int i = 0; i < count; ++i) {
        std::string line;
        if (!std::getline(std::cin, line)) break;
        double price, K, T;
        int is_call;
        std::sscanf(line.c_str(), "%lf,%lf,%lf,%d", &price, &K, &T, &is_call);

        double iv_val = 0.0;
        if (is_call) {
            auto res = implied_vol_newton(price, S, K, r, T);
            iv_val = res.converged ? res.sigma : 0.0;
        } else {
            // For puts, use put-call parity to get call price, then solve
            double call_price = price + S - K * std::exp(-r * T);
            auto res = implied_vol_newton(call_price, S, K, r, T);
            iv_val = res.converged ? res.sigma : 0.0;
        }
        std::cout << K << "," << T << "," << iv_val << ","
                  << (is_call ? "call" : "put") << "\n";
    }
}

// ============================================================================
// Command: vasicek_curve
// ============================================================================
void cmd_vasicek_curve(const std::map<std::string, std::string>& p) {
    rates::VasicekParams vp;
    vp.r0 = get_d(p, "r0", 0.05);
    vp.kappa = get_d(p, "kappa", 0.5);
    vp.theta = get_d(p, "theta", 0.05);
    vp.sigma = get_d(p, "sigma", 0.015);

    std::cout << "Tenor,ZeroRate,BondPrice\n";
    for (double T : {0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0}) {
        double P = rates::vasicek_bond_price(T, vp);
        double r = rates::vasicek_zero_rate(T, vp);
        std::cout << T << "," << r << "," << P << "\n";
    }
}

// ============================================================================
// Main — dispatch by command
// ============================================================================
int main() {
    auto params = parse_input();
    auto it = params.find("COMMAND");
    if (it == params.end()) {
        std::cerr << "ERROR: no COMMAND specified\n";
        return 1;
    }

    const std::string& cmd = it->second;

    if (cmd == "heston_smile")    cmd_heston_smile(params);
    else if (cmd == "merton_smile")   cmd_merton_smile(params);
    else if (cmd == "barrier_curve")  cmd_barrier_curve(params);
    else if (cmd == "multi_asset")    cmd_multi_asset(params);
    else if (cmd == "batch_iv")       cmd_batch_iv(params);
    else if (cmd == "vasicek_curve")  cmd_vasicek_curve(params);
    else {
        std::cerr << "ERROR: unknown command '" << cmd << "'\n";
        return 1;
    }

    return 0;
}

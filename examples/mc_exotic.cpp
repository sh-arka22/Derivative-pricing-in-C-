// Exotic options MC — Chapters 12, 15
#include "mc/monte_carlo.h"
#include "greeks/black_scholes.h"
#include <iostream>
#include <iomanip>
int main() {
    double S=100, K=100, r=0.05, T=1.0, sigma=0.2;
    PayOffCall call(K);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Asian Options (Ch 12) ===\n";
    auto arith = mc_asian_arithmetic(call, S, r, T, sigma, 252, 500000);
    auto geo = mc_asian_geometric(call, S, r, T, sigma, 252, 500000);
    std::cout << "Arithmetic Asian: " << arith.price << " +/- " << arith.std_error << "\n";
    std::cout << "Geometric Asian:  " << geo.price << " +/- " << geo.std_error << "\n";
    std::cout << "European (ref):   " << bs::call_price(S,K,r,T,sigma) << "\n";
    std::cout << "\n=== Jump-Diffusion (Ch 15) ===\n";
    MertonJumpParams jp;
    auto jd = mc_merton_jump(call, jp, 500000);
    std::cout << "Merton JD: " << jd.price << " +/- " << jd.std_error << "\n";
}

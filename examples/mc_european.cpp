// European Monte Carlo pricing — Chapters 10, 14
#include "mc/monte_carlo.h"
#include "greeks/black_scholes.h"
#include <iostream>
#include <iomanip>
int main() {
    double S=100, K=100, r=0.05, T=1.0, sigma=0.2;
    PayOffCall call(K);
    PayOffPut put(K);
    std::cout << std::fixed << std::setprecision(6);
    for (size_t paths : {10000UL, 100000UL, 500000UL, 1000000UL}) {
        auto c = mc_european(call, S, r, T, sigma, paths);
        auto p = mc_european(put, S, r, T, sigma, paths);
        std::cout << "Paths=" << std::setw(8) << paths
                  << "  Call=" << c.price << " (SE=" << c.std_error << ")"
                  << "  Put=" << p.price << " (SE=" << p.std_error << ")\n";
    }
    std::cout << "\nAnalytic: Call=" << bs::call_price(S,K,r,T,sigma)
              << "  Put=" << bs::put_price(S,K,r,T,sigma) << "\n";
}

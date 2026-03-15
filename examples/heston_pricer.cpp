// Heston stochastic volatility — Chapter 16
#include "mc/monte_carlo.h"
#include "greeks/black_scholes.h"
#include <iostream>
#include <iomanip>
int main() {
    std::cout << std::fixed << std::setprecision(4);
    PayOffCall call(100);
    HestonParams hp;
    std::cout << "=== Heston vs BS across correlation values ===\n";
    std::cout << "Rho     Heston  SE      BS(flat)\n";
    for (double rho : {-0.9, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7}) {
        hp.rho = rho;
        auto res = mc_heston(call, hp, 200000);
        std::cout << std::setw(6) << rho << "  "
                  << std::setw(7) << res.price << "  "
                  << std::setw(7) << res.std_error << "  "
                  << bs::call_price(hp.S0, hp.K, hp.r, hp.T, std::sqrt(hp.v0)) << "\n";
    }
    std::cout << "\nFeller condition (2*kappa*theta > xi^2): "
              << (hp.feller_satisfied() ? "YES" : "NO") << "\n";
}

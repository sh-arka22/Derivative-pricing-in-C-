// Merton jump-diffusion — Chapter 15
#include "mc/monte_carlo.h"
#include "greeks/black_scholes.h"
#include <iostream>
#include <iomanip>
int main() {
    std::cout << std::fixed << std::setprecision(4);
    PayOffCall call(100);
    MertonJumpParams jp;
    std::cout << "=== Jump intensity sensitivity ===\n";
    std::cout << "Lambda  Price   SE      BS(GBM)\n";
    double bs_ref = bs::call_price(jp.S0, jp.K, jp.r, jp.T, jp.sigma);
    for (double lam : {0.0, 0.5, 1.0, 2.0, 5.0, 10.0}) {
        jp.lambda = lam;
        auto res = mc_merton_jump(call, jp, 200000);
        std::cout << std::setw(6) << lam << "  "
                  << std::setw(7) << res.price << "  "
                  << std::setw(7) << res.std_error << "  "
                  << bs_ref << "\n";
    }
}

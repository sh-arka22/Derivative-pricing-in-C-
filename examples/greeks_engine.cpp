// Greeks comparison — Chapter 11
#include "greeks/greeks_engine.h"
#include <iostream>
#include <iomanip>
int main() {
    double S=100, K=100, r=0.05, T=1.0, sigma=0.2;
    std::cout << std::fixed << std::setprecision(6);
    auto a = analytic_greeks_call(S, K, r, T, sigma);
    FDGreeks fd([](double S, double K, double r, double T, double sigma) {
        return bs::call_price(S, K, r, T, sigma);
    });
    auto f = fd.all(S, K, r, T, sigma);
    double mc_d = mc_delta_pathwise(S, K, r, T, sigma, 1000000);
    double mc_v = mc_vega_pathwise(S, K, r, T, sigma, 1000000);
    std::cout << "          Analytic       FD          MC\n";
    std::cout << "Delta:    " << a.delta << "  " << f.delta << "  " << mc_d << "\n";
    std::cout << "Gamma:    " << a.gamma << "  " << f.gamma << "\n";
    std::cout << "Vega:     " << a.vega  << "  " << f.vega  << "  " << mc_v << "\n";
    std::cout << "Theta:    " << a.theta << "  " << f.theta << "\n";
    std::cout << "Rho:      " << a.rho   << "  " << f.rho   << "\n";
}

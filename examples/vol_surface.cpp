// Implied volatility & vol surface — Chapter 13
#include "vol/implied_vol.h"
#include "mc/monte_carlo.h"
#include <iostream>
#include <iomanip>
int main() {
    double S=100, r=0.05;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Strike  T=0.25  T=0.5   T=1.0   T=2.0\n";
    std::cout << std::string(48, '-') << "\n";
    HestonParams hp;
    hp.S0=S; hp.r=r; hp.v0=0.04; hp.kappa=2.0; hp.theta=0.04; hp.xi=0.3; hp.rho=-0.7;
    for (double K : {85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0}) {
        std::cout << std::setw(6) << K;
        for (double T : {0.25, 0.5, 1.0, 2.0}) {
            hp.K = K; hp.T = T;
            PayOffCall payoff(K);
            auto mc = mc_heston(payoff, hp, 100000);
            auto iv = implied_vol_newton(mc.price, S, K, r, T);
            std::cout << std::setw(8) << (iv.converged ? iv.sigma : 0.0);
        }
        std::cout << "\n";
    }
}

// FDM pricing — Chapter 17
#include "fdm/fdm.h"
#include "greeks/black_scholes.h"
#include <iostream>
#include <iomanip>
int main() {
    double S0=100, K=100, r=0.05, T=1.0, sigma=0.2;
    PayOffCall payoff(K);
    BlackScholesPDE pde(r, sigma);
    FDMParams params(400, 2000, 300.0, 0.5);
    FDMSolver solver(payoff, pde, S0, T, params);
    auto result = solver.solve();
    solver.write_csv("fdm_output.csv", result);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "FDM (Crank-Nicolson) Price: " << result.price_at_spot << "\n";
    std::cout << "Analytic Price:             " << bs::call_price(S0,K,r,T,sigma) << "\n";
    std::cout << "Written grid to fdm_output.csv\n";
}

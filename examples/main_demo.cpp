// ============================================================================
// QuantPricer — Full Demo
// Demonstrates every module in the pricing engine.
// Maps to all 17 chapters of "C++ For Quantitative Finance".
// ============================================================================

#include "payoff/payoff.h"
#include "option/option.h"
#include "matrix/matrix.h"
#include "rng/rng.h"
#include "mc/monte_carlo.h"
#include "greeks/black_scholes.h"
#include "greeks/greeks_engine.h"
#include "vol/implied_vol.h"
#include "fdm/fdm.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <memory>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(72, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(72, '=') << "\n";
}

void print_subheader(const std::string& title) {
    std::cout << "\n--- " << title << " ---\n";
}

// ============================================================================
// Demo 1: PayOff Hierarchy (Ch 3-4)
// ============================================================================
void demo_payoffs() {
    print_header("PAYOFF HIERARCHY (Chapters 3-4)");
    std::cout << "Ch 3: OOP, VanillaOption class, const correctness\n"
              << "Ch 4: Inheritance, abstract base class, virtual destructors, operator()\n\n";

    double K = 100.0;
    PayOffCall call(K);
    PayOffPut put(K);
    PayOffDigitalCall digi_call(K);
    PayOffDoubleDigital dbl_digi(90.0, 110.0);
    PayOffPower power(K, 1.5);

    std::vector<double> spots = {80, 90, 95, 100, 105, 110, 120};

    std::cout << std::setw(8) << "Spot" << std::setw(10) << "Call"
              << std::setw(10) << "Put" << std::setw(10) << "DigiCall"
              << std::setw(12) << "DblDigital" << std::setw(10) << "Power\n";
    std::cout << std::string(60, '-') << "\n";

    for (double S : spots) {
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(8) << S
                  << std::setw(10) << call(S)
                  << std::setw(10) << put(S)
                  << std::setw(10) << digi_call(S)
                  << std::setw(12) << dbl_digi(S)
                  << std::setw(10) << power(S) << "\n";
    }

    // Demonstrate polymorphic cloning (smart pointers)
    std::unique_ptr<PayOff> cloned = call.clone();
    std::cout << "\nCloned call payoff at S=110: " << (*cloned)(110.0) << "\n";
}

// ============================================================================
// Demo 2: Matrix Operations & Linear Algebra (Ch 5, 8-9)
// ============================================================================
void demo_matrix() {
    print_header("MATRIX & LINEAR ALGEBRA (Chapters 5, 8-9)");
    std::cout << "Ch 5: Template classes, generic programming\n"
              << "Ch 8: Custom matrix class, operator overloading, STL storage\n"
              << "Ch 9: LU decomposition, Thomas algorithm, Cholesky\n\n";

    // Matrix arithmetic — Ch 8
    QMatrix<double> A(3, 3);
    A(0,0)=4; A(0,1)=12; A(0,2)=-16;
    A(1,0)=12; A(1,1)=37; A(1,2)=-43;
    A(2,0)=-16; A(2,1)=-43; A(2,2)=98;

    print_subheader("Cholesky Decomposition (Ch 9.4)");
    std::cout << "Matrix A (SPD):\n";
    A.print(2);
    auto L = cholesky(A);
    std::cout << "\nL (lower triangular):\n";
    L.print(4);
    auto LLT = L * L.transpose();
    std::cout << "\nL*L^T (should equal A):\n";
    LLT.print(2);

    // LU solve — Ch 9.2
    print_subheader("LU Solve (Ch 9.2)");
    QMatrix<double> B(3, 3);
    B(0,0)=2; B(0,1)=1; B(0,2)=-1;
    B(1,0)=-3; B(1,1)=-1; B(1,2)=2;
    B(2,0)=-2; B(2,1)=1; B(2,2)=2;
    std::vector<double> b = {8, -11, -3};

    auto x = solve_lu(B, b);
    std::cout << "Solving Bx = b:\n  x = [";
    for (size_t i = 0; i < x.size(); ++i)
        std::cout << x[i] << (i < x.size()-1 ? ", " : "");
    std::cout << "]\n  Expected: [2, 3, -1]\n";

    // Thomas algorithm — Ch 9.3
    print_subheader("Thomas Algorithm (Ch 9.3)");
    std::vector<double> sub = {1, 1, 1};
    std::vector<double> diag = {4, 4, 4, 4};
    std::vector<double> sup = {1, 1, 1};
    std::vector<double> d = {5, 5, 5, 5};
    auto xt = solve_thomas(sub, diag, sup, d);
    std::cout << "Tridiagonal solve: x = [";
    for (size_t i = 0; i < xt.size(); ++i)
        std::cout << std::setprecision(6) << xt[i] << (i < xt.size()-1 ? ", " : "");
    std::cout << "]\n";
}

// ============================================================================
// Demo 3: Black-Scholes Analytics & Greeks (Ch 3, 10, 11)
// ============================================================================
void demo_black_scholes() {
    print_header("BLACK-SCHOLES ANALYTICS & GREEKS (Chapters 3, 10-11)");
    std::cout << "Ch 3:  VanillaOption pricing\n"
              << "Ch 10: Analytic pricing formula\n"
              << "Ch 11: Analytic + FD + MC Greeks\n\n";

    double S=100, K=100, r=0.05, T=1.0, sigma=0.2;

    // Analytic prices
    double call = bs::call_price(S, K, r, T, sigma);
    double put = bs::put_price(S, K, r, T, sigma);
    std::cout << "European Call (analytic): " << std::setprecision(6) << call << "\n";
    std::cout << "European Put  (analytic): " << put << "\n";
    std::cout << "Put-Call Parity check:    C - P = " << call - put
              << "  vs  S - K*exp(-rT) = " << S - K*std::exp(-r*T) << "\n";

    // Analytic Greeks — Ch 11.1
    print_subheader("Analytic Greeks (Ch 11.1)");
    auto greeks = analytic_greeks_call(S, K, r, T, sigma);
    std::cout << "Delta: " << greeks.delta << "\n"
              << "Gamma: " << greeks.gamma << "\n"
              << "Vega:  " << greeks.vega << "\n"
              << "Theta: " << greeks.theta << "\n"
              << "Rho:   " << greeks.rho << "\n";

    // FD Greeks — Ch 11.2
    print_subheader("FD Greeks vs Analytic (Ch 11.2)");
    FDGreeks fd_greeks([](double S, double K, double r, double T, double sigma) {
        return bs::call_price(S, K, r, T, sigma);
    });
    auto fd = fd_greeks.all(S, K, r, T, sigma);
    std::cout << std::setw(10) << "Greek" << std::setw(14) << "Analytic"
              << std::setw(14) << "FD" << std::setw(14) << "Error\n";
    std::cout << std::string(52, '-') << "\n";

    auto print_row = [](const char* name, double a, double f) {
        std::cout << std::setw(10) << name << std::setw(14) << a
                  << std::setw(14) << f << std::setw(14) << std::abs(a-f) << "\n";
    };
    print_row("Delta", greeks.delta, fd.delta);
    print_row("Gamma", greeks.gamma, fd.gamma);
    print_row("Vega",  greeks.vega,  fd.vega);
    print_row("Theta", greeks.theta, fd.theta);
    print_row("Rho",   greeks.rho,   fd.rho);

    // MC Greeks — Ch 11.3
    print_subheader("MC Pathwise Greeks (Ch 11.3)");
    size_t mc_paths = 500000;
    double mc_delta = mc_delta_pathwise(S, K, r, T, sigma, mc_paths);
    double mc_vega = mc_vega_pathwise(S, K, r, T, sigma, mc_paths);
    std::cout << "MC Delta (pathwise, " << mc_paths << " paths): " << mc_delta
              << "  (analytic: " << greeks.delta << ")\n";
    std::cout << "MC Vega  (pathwise, " << mc_paths << " paths): " << mc_vega
              << "  (analytic: " << greeks.vega << ")\n";
}

// ============================================================================
// Demo 4: Monte Carlo — European, Asian, Jump-Diffusion, Heston (Ch 10,12,15,16)
// ============================================================================
void demo_monte_carlo() {
    print_header("MONTE CARLO ENGINE (Chapters 10, 12, 15, 16)");

    double S0=100, K=100, r=0.05, T=1.0, sigma=0.2;
    size_t num_paths = 500000;

    // European MC — Ch 10
    print_subheader("European Call MC (Ch 10.3-10.4)");
    PayOffCall call(K);
    auto t0 = Clock::now();
    auto result = mc_european(call, S0, r, T, sigma, num_paths);
    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double analytic = bs::call_price(S0, K, r, T, sigma);
    std::cout << "MC Price:       " << std::setprecision(6) << result.price << "\n"
              << "Analytic Price: " << analytic << "\n"
              << "Std Error:      " << result.std_error << "\n"
              << "95% CI:         [" << result.confidence_lo << ", "
              << result.confidence_hi << "]\n"
              << "Time:           " << std::setprecision(1) << ms << " ms\n";

    // Asian MC — Ch 12
    print_subheader("Arithmetic Asian Call MC (Ch 12.5-12.6)");
    auto asian_arith = mc_asian_arithmetic(call, S0, r, T, sigma, 252, 200000);
    auto asian_geo = mc_asian_geometric(call, S0, r, T, sigma, 252, 200000);
    std::cout << "Arithmetic Asian: " << std::setprecision(4) << asian_arith.price
              << " (SE: " << asian_arith.std_error << ")\n"
              << "Geometric Asian:  " << asian_geo.price
              << " (SE: " << asian_geo.std_error << ")\n"
              << "Note: Asian < European (" << analytic
              << ") due to averaging effect\n";

    // Jump-Diffusion — Ch 15
    print_subheader("Merton Jump-Diffusion MC (Ch 15.2-15.3)");
    MertonJumpParams jp;
    jp.S0 = S0; jp.K = K; jp.r = r; jp.T = T; jp.sigma = sigma;
    jp.lambda = 1.0; jp.mu_j = -0.1; jp.sigma_j = 0.15;

    auto jd_result = mc_merton_jump(call, jp, 200000);
    std::cout << "Jump-Diffusion Price: " << jd_result.price
              << " (SE: " << jd_result.std_error << ")\n"
              << "vs Pure GBM:          " << analytic << "\n"
              << "Jump params: lambda=" << jp.lambda
              << ", mu_j=" << jp.mu_j << ", sigma_j=" << jp.sigma_j << "\n";

    // Heston — Ch 16
    print_subheader("Heston Stochastic Vol MC (Ch 16.5-16.7)");
    HestonParams hp;
    hp.S0 = S0; hp.K = K; hp.r = r; hp.T = T;
    hp.v0 = 0.04; hp.kappa = 2.0; hp.theta = 0.04; hp.xi = 0.3; hp.rho = -0.7;

    auto heston_result = mc_heston(call, hp, 200000);
    std::cout << "Heston Price:   " << heston_result.price
              << " (SE: " << heston_result.std_error << ")\n"
              << "vs BS (flat vol): " << analytic << "\n"
              << "Feller condition: " << (hp.feller_satisfied() ? "SATISFIED" : "VIOLATED") << "\n"
              << "Parameters: v0=" << hp.v0 << " kappa=" << hp.kappa
              << " theta=" << hp.theta << " xi=" << hp.xi << " rho=" << hp.rho << "\n";
}

// ============================================================================
// Demo 5: Implied Volatility (Ch 13)
// ============================================================================
void demo_implied_vol() {
    print_header("IMPLIED VOLATILITY (Chapter 13)");
    std::cout << "Ch 13.3: Interval bisection method\n"
              << "Ch 13.4: Newton-Raphson with Vega derivative\n\n";

    double S=100, K=100, r=0.05, T=1.0, true_sigma=0.25;
    double market_price = bs::call_price(S, K, r, T, true_sigma);

    std::cout << "True sigma: " << true_sigma << "\n"
              << "Market price (from BS): " << market_price << "\n\n";

    // Bisection — Ch 13.3
    auto t0 = Clock::now();
    auto bs_result = implied_vol_bisection(market_price, S, K, r, T);
    auto t1 = Clock::now();
    std::cout << "Bisection:      sigma=" << std::setprecision(8) << bs_result.sigma
              << "  iters=" << bs_result.iterations
              << "  residual=" << bs_result.residual
              << "  time=" << std::chrono::duration<double, std::micro>(t1-t0).count() << " us\n";

    // Newton-Raphson — Ch 13.4
    t0 = Clock::now();
    auto nr_result = implied_vol_newton(market_price, S, K, r, T);
    t1 = Clock::now();
    std::cout << "Newton-Raphson: sigma=" << std::setprecision(8) << nr_result.sigma
              << "  iters=" << nr_result.iterations
              << "  residual=" << nr_result.residual
              << "  time=" << std::chrono::duration<double, std::micro>(t1-t0).count() << " us\n";

    // Volatility smile — compute IV for range of strikes
    print_subheader("Volatility Smile (Extension)");
    std::cout << std::setw(10) << "Strike" << std::setw(14) << "Price"
              << std::setw(14) << "ImpliedVol\n";
    std::cout << std::string(38, '-') << "\n";

    // Generate synthetic smile using Heston prices
    HestonParams hp;
    hp.S0=100; hp.r=0.05; hp.T=1.0; hp.v0=0.04; hp.kappa=2.0;
    hp.theta=0.04; hp.xi=0.3; hp.rho=-0.7;

    for (double strike : {80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0}) {
        hp.K = strike;
        PayOffCall payoff(strike);
        auto mc = mc_heston(payoff, hp, 100000);
        auto iv = implied_vol_newton(mc.price, S, strike, r, T);
        std::cout << std::setprecision(4)
                  << std::setw(10) << strike
                  << std::setw(14) << mc.price
                  << std::setw(14) << (iv.converged ? iv.sigma : 0.0) << "\n";
    }
}

// ============================================================================
// Demo 6: FDM Solver (Ch 17)
// ============================================================================
void demo_fdm() {
    print_header("FINITE DIFFERENCE METHOD (Chapter 17)");
    std::cout << "Ch 17.1: Black-Scholes PDE\n"
              << "Ch 17.2: FD discretisation (explicit, implicit, Crank-Nicolson)\n"
              << "Ch 17.3: Full implementation using Thomas algorithm (Ch 9.3)\n\n";

    double S0=100, K=100, r=0.05, T=1.0, sigma=0.2;
    double analytic = bs::call_price(S0, K, r, T, sigma);

    PayOffCall payoff(K);
    BlackScholesPDE pde(r, sigma);

    // Compare three schemes — Ch 17.2
    struct Scheme { const char* name; double theta; };
    std::vector<Scheme> schemes = {
        {"Explicit (theta=0.0)", 0.0},
        {"Crank-Nicolson (theta=0.5)", 0.5},
        {"Implicit (theta=1.0)", 1.0}
    };

    std::cout << std::setw(30) << "Scheme" << std::setw(14) << "FDM Price"
              << std::setw(14) << "Analytic" << std::setw(14) << "Error"
              << std::setw(12) << "Time(ms)\n";
    std::cout << std::string(84, '-') << "\n";

    for (auto& scheme : schemes) {
        FDMParams params(200, 2000, 300.0, scheme.theta);
        FDMSolver solver(payoff, pde, S0, T, params);

        auto t0 = Clock::now();
        auto result = solver.solve();
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << std::setprecision(6)
                  << std::setw(30) << scheme.name
                  << std::setw(14) << result.price_at_spot
                  << std::setw(14) << analytic
                  << std::setw(14) << std::abs(result.price_at_spot - analytic)
                  << std::setw(12) << std::setprecision(1) << ms << "\n";
    }

    // Grid convergence study
    print_subheader("Grid Convergence (Crank-Nicolson)");
    std::cout << std::setw(10) << "N_space" << std::setw(10) << "N_time"
              << std::setw(14) << "Price" << std::setw(14) << "Error\n";
    std::cout << std::string(48, '-') << "\n";

    for (size_t n : {50, 100, 200, 400, 800}) {
        FDMParams params(n, 5 * n, 300.0, 0.5);
        FDMSolver solver(payoff, pde, S0, T, params);
        auto result = solver.solve();
        std::cout << std::setw(10) << n
                  << std::setw(10) << 5*n
                  << std::setprecision(6)
                  << std::setw(14) << result.price_at_spot
                  << std::setw(14) << std::abs(result.price_at_spot - analytic) << "\n";
    }
}

// ============================================================================
// Demo 7: RNG Quality (Ch 14)
// ============================================================================
void demo_rng() {
    print_header("RANDOM NUMBER GENERATION (Chapter 14)");
    std::cout << "Ch 14.2: RNG hierarchy — LCG vs Mersenne Twister\n"
              << "Ch 14.3: Statistical distributions — Standard Normal\n\n";

    size_t N = 1000000;

    // Compare LCG vs MT — Ch 14.2
    LinearCongruentialGenerator lcg(42);
    MersenneTwisterRNG mt(42);

    double lcg_sum = 0, mt_sum = 0;
    double lcg_sq = 0, mt_sq = 0;
    for (size_t i = 0; i < N; ++i) {
        double u1 = lcg.generate_uniform();
        double u2 = mt.generate_uniform();
        lcg_sum += u1; lcg_sq += u1*u1;
        mt_sum += u2;  mt_sq += u2*u2;
    }
    double lcg_mean = lcg_sum/N, mt_mean = mt_sum/N;
    double lcg_var = lcg_sq/N - lcg_mean*lcg_mean;
    double mt_var = mt_sq/N - mt_mean*mt_mean;

    std::cout << "U(0,1) samples (" << N << "):\n"
              << "  LCG:  mean=" << std::setprecision(6) << lcg_mean
              << "  var=" << lcg_var << "  (expected: 0.5, 0.0833)\n"
              << "  MT:   mean=" << mt_mean
              << "  var=" << mt_var << "  (expected: 0.5, 0.0833)\n";

    // Standard Normal CDF check — Ch 14.3.2
    StandardNormalDistribution snd;
    std::cout << "\nStandard Normal CDF checks:\n"
              << "  Phi(0)   = " << snd.cdf(0.0)   << "  (expected: 0.5)\n"
              << "  Phi(1)   = " << snd.cdf(1.0)   << "  (expected: 0.8413)\n"
              << "  Phi(-1)  = " << snd.cdf(-1.0)  << "  (expected: 0.1587)\n"
              << "  Phi(1.96)= " << snd.cdf(1.96)  << "  (expected: 0.975)\n";
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << std::fixed;
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
              << "║       QuantPricer v1.0 — Multi-Model Pricing Engine         ║\n"
              << "║   C++ For Quantitative Finance — All Chapters Combined      ║\n"
              << "╚══════════════════════════════════════════════════════════════╝\n";

    demo_payoffs();
    demo_matrix();
    demo_rng();
    demo_black_scholes();
    demo_monte_carlo();
    demo_implied_vol();
    demo_fdm();

    std::cout << "\n" << std::string(72, '=') << "\n"
              << "  All demos completed successfully.\n"
              << std::string(72, '=') << "\n";

    return 0;
}

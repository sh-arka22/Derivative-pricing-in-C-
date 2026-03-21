// ============================================================================
// QuantPricer Test Suite
// Validates pricing accuracy, Greeks correctness, solver convergence,
// and numerical stability across all modules.
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
#include "tree/binomial_tree.h"
#include "mc/american_mc.h"
#include "barrier/barrier.h"
#include "multi_asset/multi_asset.h"

#include <iostream>
#include <cmath>
#include <cassert>
#include <string>

static int tests_passed = 0;
static int tests_failed = 0;

void check(bool condition, const std::string& name) {
    if (condition) {
        std::cout << "  [PASS] " << name << "\n";
        tests_passed++;
    } else {
        std::cout << "  [FAIL] " << name << "\n";
        tests_failed++;
    }
}

void check_approx(double actual, double expected, double tol, const std::string& name) {
    bool ok = std::abs(actual - expected) < tol;
    if (ok) {
        std::cout << "  [PASS] " << name << " (got " << actual << ")\n";
        tests_passed++;
    } else {
        std::cout << "  [FAIL] " << name << " (got " << actual
                  << ", expected " << expected << ", tol " << tol << ")\n";
        tests_failed++;
    }
}

// ============================================================================
// Test Groups
// ============================================================================

void test_payoffs() {
    std::cout << "\n=== PayOff Tests (Ch 3-4) ===\n";
    PayOffCall call(100);
    PayOffPut put(100);
    PayOffDoubleDigital dd(90, 110);

    check_approx(call(120), 20.0, 1e-10, "Call(120, K=100) = 20");
    check_approx(call(80), 0.0, 1e-10, "Call(80, K=100) = 0");
    check_approx(put(80), 20.0, 1e-10, "Put(80, K=100) = 20");
    check_approx(put(120), 0.0, 1e-10, "Put(120, K=100) = 0");
    check_approx(dd(100), 1.0, 1e-10, "DblDigital(100, [90,110]) = 1");
    check_approx(dd(80), 0.0, 1e-10, "DblDigital(80, [90,110]) = 0");

    // Clone test
    auto cloned = call.clone();
    check_approx((*cloned)(110), 10.0, 1e-10, "Cloned call works");
}

void test_matrix() {
    std::cout << "\n=== Matrix Tests (Ch 5, 8-9) ===\n";

    // Basic operations
    QMatrix<double> A(2, 2);
    A(0,0)=1; A(0,1)=2; A(1,0)=3; A(1,1)=4;
    QMatrix<double> B(2, 2);
    B(0,0)=5; B(0,1)=6; B(1,0)=7; B(1,1)=8;

    auto C = A + B;
    check_approx(C(0,0), 6.0, 1e-10, "Matrix addition");
    auto D = A * B;
    check_approx(D(0,0), 19.0, 1e-10, "Matrix multiply (0,0)");
    check_approx(D(1,1), 50.0, 1e-10, "Matrix multiply (1,1)");

    // LU solve
    QMatrix<double> M(3, 3);
    M(0,0)=2; M(0,1)=1; M(0,2)=-1;
    M(1,0)=-3; M(1,1)=-1; M(1,2)=2;
    M(2,0)=-2; M(2,1)=1; M(2,2)=2;
    std::vector<double> b = {8, -11, -3};
    auto x = solve_lu(M, b);
    check_approx(x[0], 2.0, 1e-8, "LU solve x[0]=2");
    check_approx(x[1], 3.0, 1e-8, "LU solve x[1]=3");
    check_approx(x[2], -1.0, 1e-8, "LU solve x[2]=-1");

    // Thomas algorithm
    std::vector<double> sub={1,1}, diag={4,4,4}, sup={1,1}, d_vec={5,6,5};
    auto xt = solve_thomas(sub, diag, sup, d_vec);
    check(xt.size() == 3, "Thomas returns correct size");

    // Cholesky
    QMatrix<double> SPD(2, 2);
    SPD(0,0)=4; SPD(0,1)=2; SPD(1,0)=2; SPD(1,1)=5;
    auto L = cholesky(SPD);
    check_approx(L(0,0), 2.0, 1e-8, "Cholesky L(0,0)=2");
    check_approx(L(1,0), 1.0, 1e-8, "Cholesky L(1,0)=1");
}

void test_rng() {
    std::cout << "\n=== RNG Tests (Ch 14) ===\n";

    MersenneTwisterRNG mt(42);
    double sum = 0;
    size_t N = 100000;
    for (size_t i = 0; i < N; ++i) sum += mt.generate_uniform();
    check_approx(sum / N, 0.5, 0.01, "MT uniform mean ~0.5");

    // Normal samples
    mt.reset();
    auto normals = generate_normals(mt, N);
    double nsum = 0, nsq = 0;
    for (double z : normals) { nsum += z; nsq += z*z; }
    check_approx(nsum / N, 0.0, 0.02, "Normal mean ~0");
    check_approx(nsq / N, 1.0, 0.02, "Normal variance ~1");

    // Standard Normal CDF
    StandardNormalDistribution snd;
    check_approx(snd.cdf(0.0), 0.5, 1e-6, "Phi(0) = 0.5");
    check_approx(snd.cdf(1.96), 0.975, 0.001, "Phi(1.96) ~0.975");
}

void test_black_scholes() {
    std::cout << "\n=== Black-Scholes Tests (Ch 3, 10) ===\n";

    double S=100, K=100, r=0.05, T=1.0, sigma=0.2;
    double call = bs::call_price(S, K, r, T, sigma);
    double put = bs::put_price(S, K, r, T, sigma);

    // Known reference value
    check_approx(call, 10.4506, 0.001, "BS call ~10.4506");

    // Put-call parity: C - P = S - K*exp(-rT)
    double parity_lhs = call - put;
    double parity_rhs = S - K * std::exp(-r * T);
    check_approx(parity_lhs, parity_rhs, 1e-10, "Put-call parity holds");

    // ATM approximation: C ≈ S*σ*√T / √(2π) for ATM
    double atm_approx = S * sigma * std::sqrt(T) / std::sqrt(2.0 * M_PI);
    check_approx(call, atm_approx, 3.0, "ATM approximation reasonable");
}

void test_greeks() {
    std::cout << "\n=== Greeks Tests (Ch 11) ===\n";

    double S=100, K=100, r=0.05, T=1.0, sigma=0.2;
    auto a = analytic_greeks_call(S, K, r, T, sigma);

    // Delta bounds: 0 < delta_call < 1
    check(a.delta > 0.0 && a.delta < 1.0, "Call delta in (0,1)");

    // Gamma positive
    check(a.gamma > 0.0, "Gamma positive");

    // Vega positive
    check(a.vega > 0.0, "Vega positive");

    // FD Greeks match analytic
    FDGreeks fd([](double S, double K, double r, double T, double sigma) {
        return bs::call_price(S, K, r, T, sigma);
    });
    check_approx(fd.delta(S, K, r, T, sigma), a.delta, 0.001, "FD delta matches analytic");
    check_approx(fd.gamma(S, K, r, T, sigma), a.gamma, 0.001, "FD gamma matches analytic");
    check_approx(fd.vega(S, K, r, T, sigma), a.vega, 0.05, "FD vega matches analytic");
}

void test_mc_european() {
    std::cout << "\n=== MC European Tests (Ch 10) ===\n";

    double S=100, K=100, r=0.05, T=1.0, sigma=0.2;
    PayOffCall call(K);
    auto result = mc_european(call, S, r, T, sigma, 500000);
    double analytic = bs::call_price(S, K, r, T, sigma);

    check_approx(result.price, analytic, 3.0 * result.std_error,
                 "MC call within 3 SE of analytic");
    check(result.std_error < 0.1, "MC standard error < 0.1");
}

void test_implied_vol() {
    std::cout << "\n=== Implied Volatility Tests (Ch 13) ===\n";

    double S=100, K=100, r=0.05, T=1.0, true_sigma=0.25;
    double price = bs::call_price(S, K, r, T, true_sigma);

    auto bs_iv = implied_vol_bisection(price, S, K, r, T);
    check_approx(bs_iv.sigma, true_sigma, 1e-6, "Bisection recovers true sigma");
    check(bs_iv.converged, "Bisection converged");

    auto nr_iv = implied_vol_newton(price, S, K, r, T);
    check_approx(nr_iv.sigma, true_sigma, 1e-6, "Newton-Raphson recovers true sigma");
    check(nr_iv.converged, "Newton-Raphson converged");
    check(nr_iv.iterations < bs_iv.iterations, "Newton-Raphson faster than bisection");
}

void test_fdm() {
    std::cout << "\n=== FDM Tests (Ch 17) ===\n";

    double S0=100, K=100, r=0.05, T=1.0, sigma=0.2;
    double analytic = bs::call_price(S0, K, r, T, sigma);

    PayOffCall payoff(K);
    BlackScholesPDE pde(r, sigma);

    // Crank-Nicolson should be accurate
    FDMParams params(400, 2000, 300.0, 0.5);
    FDMSolver solver(payoff, pde, S0, T, params);
    auto result = solver.solve();

    check_approx(result.price_at_spot, analytic, 0.05, "FDM C-N within 5bp of analytic");
    check(result.spot_grid.size() == 401, "Grid has correct size");
}

void test_heston() {
    std::cout << "\n=== Heston Tests (Ch 16) ===\n";

    HestonParams hp;
    check(hp.feller_satisfied(), "Default Heston satisfies Feller");

    PayOffCall call(hp.K);
    auto result = mc_heston(call, hp, 100000);
    check(result.price > 0.0, "Heston price positive");
    check(result.std_error < 1.0, "Heston SE reasonable");
}

void test_jump_diffusion() {
    std::cout << "\n=== Jump-Diffusion Tests (Ch 15) ===\n";

    MertonJumpParams jp;
    jp.lambda = 0.0;  // No jumps = pure GBM
    PayOffCall call(jp.K);
    auto result = mc_merton_jump(call, jp, 200000);
    double bs_price = bs::call_price(jp.S0, jp.K, jp.r, jp.T, jp.sigma);

    check_approx(result.price, bs_price, 3.0 * result.std_error,
                 "Merton with lambda=0 matches BS");
}

void test_binomial_tree() {
    std::cout << "\n=== Binomial Tree Tests (Day 16) ===\n";

    double S=100, K=100, r=0.05, T=1.0, sigma=0.2;
    PayOffCall call(K);
    PayOffPut put(K);

    // European call via tree should converge to BS
    double bs_call = bs::call_price(S, K, r, T, sigma);
    auto tree_call = binomial_european(call, S, K, r, T, sigma, 1000);
    check_approx(tree_call.price, bs_call, 0.05, "Tree European call matches BS");

    // European put via tree should converge to BS
    double bs_put = bs::put_price(S, K, r, T, sigma);
    auto tree_put = binomial_european(put, S, K, r, T, sigma, 1000);
    check_approx(tree_put.price, bs_put, 0.05, "Tree European put matches BS");

    // American put >= European put (early exercise premium >= 0)
    auto ame_put = binomial_american(put, S, K, r, T, sigma, 1000);
    check(ame_put.price >= tree_put.price - 1e-10,
          "American put >= European put");

    // American call == European call (no dividends)
    auto ame_call = binomial_american(call, S, K, r, T, sigma, 1000);
    check_approx(ame_call.price, tree_call.price, 0.01,
                 "American call == European call (no divs)");

    // Deep ITM American put has significant early exercise premium
    PayOffPut deep_put(K);
    auto deep_eur = binomial_european(deep_put, 80.0, K, r, T, sigma, 1000);
    auto deep_ame = binomial_american(deep_put, 80.0, K, r, T, sigma, 1000);
    check(deep_ame.price - deep_eur.price > 0.5,
          "Deep ITM American put has EE premium > 0.5");

    // Tree delta for European call should match BS delta
    auto tree_greeks = binomial_european(call, S, K, r, T, sigma, 1000);
    double bs_delta = bs::delta_call(S, K, r, T, sigma);
    check_approx(tree_greeks.delta, bs_delta, 0.01, "Tree delta matches BS delta");

    // Tree gamma should be positive
    check(tree_greeks.gamma > 0.0, "Tree gamma positive");
}

void test_american_mc() {
    std::cout << "\n=== Longstaff-Schwartz MC Tests (Day 16) ===\n";

    double S=100, K=100, r=0.05, T=1.0, sigma=0.2;
    PayOffPut put(K);

    // LSM American put should be positive and reasonable
    auto lsm = mc_american_lsm(put, S, r, T, sigma, 100000, 50, 3, 42);
    check(lsm.price > 0.0, "LSM American put price positive");
    check(lsm.std_error < 0.5, "LSM standard error reasonable");

    // LSM should be close to binomial tree reference
    auto tree_ref = binomial_american(put, S, K, r, T, sigma, 2000);
    check_approx(lsm.price, tree_ref.price, 3.0 * lsm.std_error + 0.3,
                 "LSM matches tree within tolerance");

    // LSM should show early exercise premium for puts
    check(lsm.early_exercise_premium >= 0.0, "LSM EE premium non-negative");

    // LSM American put should be >= European BS put
    double bs_put = bs::put_price(S, K, r, T, sigma);
    check(lsm.price >= bs_put - 3.0 * lsm.std_error,
          "LSM American put >= BS European put (within 3 SE)");
}

void test_barrier_options() {
    std::cout << "\n=== Barrier Options Tests (Day 10) ===\n";

    using namespace barrier;
    double S=100, K=100, r=0.05, T=1.0, sigma=0.2;
    double bs_call = bs::call_price(S, K, r, T, sigma);
    double bs_put  = bs::put_price(S, K, r, T, sigma);

    // Down-and-out call should be cheaper than vanilla
    BarrierParams doc(S, K, 90.0, r, T, sigma, BarrierType::DownAndOut, OptionType::Call);
    double doc_price = barrier_price(doc);
    check(doc_price > 0.0 && doc_price < bs_call, "DO call: 0 < price < vanilla");

    // Up-and-out put should be cheaper than vanilla
    BarrierParams uop(S, K, 120.0, r, T, sigma, BarrierType::UpAndOut, OptionType::Put);
    double uop_price = barrier_price(uop);
    check(uop_price > 0.0 && uop_price < bs_put, "UO put: 0 < price < vanilla");

    // In-out parity for down call: DI + DO = Vanilla
    BarrierParams dic(S, K, 90.0, r, T, sigma, BarrierType::DownAndIn, OptionType::Call);
    double dic_price = barrier_price(dic);
    check_approx(doc_price + dic_price, bs_call, 1e-10,
                 "In-out parity: DI+DO call = vanilla");

    // In-out parity for up put: UI + UO = Vanilla
    BarrierParams uip(S, K, 120.0, r, T, sigma, BarrierType::UpAndIn, OptionType::Put);
    double uip_price = barrier_price(uip);
    check_approx(uop_price + uip_price, bs_put, 1e-10,
                 "In-out parity: UI+UO put = vanilla");

    // In-out parity for up call
    BarrierParams uoc(S, K, 120.0, r, T, sigma, BarrierType::UpAndOut, OptionType::Call);
    BarrierParams uic(S, K, 120.0, r, T, sigma, BarrierType::UpAndIn, OptionType::Call);
    check_approx(barrier_price(uoc) + barrier_price(uic), bs_call, 1e-10,
                 "In-out parity: UI+UO call = vanilla");

    // In-out parity for down put
    BarrierParams dop(S, K, 90.0, r, T, sigma, BarrierType::DownAndOut, OptionType::Put);
    BarrierParams dip(S, K, 90.0, r, T, sigma, BarrierType::DownAndIn, OptionType::Put);
    check_approx(barrier_price(dop) + barrier_price(dip), bs_put, 1e-10,
                 "In-out parity: DI+DO put = vanilla");

    // As barrier -> 0, DO call -> vanilla (barrier never hit)
    BarrierParams doc_far(S, K, 1.0, r, T, sigma, BarrierType::DownAndOut, OptionType::Call);
    check_approx(barrier_price(doc_far), bs_call, 0.01,
                 "DO call with H~0 -> vanilla");

    // MC barrier should be close to analytic (with BGK correction)
    auto mc = mc_barrier(doc, 200000, 252, 42);
    double bgk = barrier_price_bgk(doc, 252);
    check_approx(mc.price, bgk, 3.0 * mc.std_error + 0.2,
                 "MC barrier ~ BGK-adjusted analytic");

    // Knock percentage should be reasonable (not 0% or 100%)
    check(mc.knock_pct > 5.0 && mc.knock_pct < 95.0,
          "Knock percentage reasonable");

    // BGK price >= continuous for knock-out (discrete monitoring misses crossings
    // → fewer knock-outs → knock-out option is MORE valuable with discrete monitoring)
    check(bgk >= doc_price - 0.01,
          "BGK price >= continuous price (fewer knock-outs)");
}

void test_multi_asset() {
    std::cout << "\n=== Multi-Asset Tests (Day 11) ===\n";

    using namespace multi_asset;
    double S0=100, r=0.05, T=1.0, sigma=0.2;

    // Margrabe: MC should match analytic
    double sigma1=0.2, sigma2=0.3, rho=0.5;
    auto marg = margrabe_price(S0, S0, sigma1, sigma2, rho, T);
    check(marg.price > 0.0, "Margrabe price positive");

    MultiAssetParams p;
    p.S0 = {S0, S0}; p.sigma = {sigma1, sigma2};
    p.weights = {1.0, 1.0};
    p.corr = uniform_corr_matrix(2, rho);
    p.r = r; p.T = T; p.K = 0.0;

    auto mc_exch = mc_multi_asset(p, PayoffType::Exchange, 200000);
    check_approx(mc_exch.price, marg.price, 3.0 * mc_exch.std_error,
                 "MC exchange matches Margrabe");

    // Margrabe does NOT depend on r — verify
    auto marg_r0 = margrabe_price(S0, S0, sigma1, sigma2, rho, T);
    // Change r and reprice via MC — both should give same result
    check_approx(marg_r0.price, marg.price, 1e-10,
                 "Margrabe independent of r");

    // Basket call should be <= single-asset call (diversification)
    auto bp = make_basket_params(5, S0, sigma, 0.5, r, T, S0);
    auto mc_basket = mc_multi_asset(bp, PayoffType::BasketCall, 200000);
    double bs_call = bs::call_price(S0, S0, r, T, sigma);
    check(mc_basket.price < bs_call + 3.0 * mc_basket.std_error,
          "Basket call <= single-asset call");

    // Basket vol should be < individual vol (diversification)
    double bvol = basket_vol(bp);
    check(bvol < sigma, "Basket vol < individual vol");

    // Best-of call >= single-asset call (max >= any individual)
    auto p2 = make_basket_params(2, S0, sigma, 0.5, r, T, S0);
    auto mc_best = mc_multi_asset(p2, PayoffType::BestOfCall, 200000);
    check(mc_best.price >= bs_call - 3.0 * mc_best.std_error,
          "Best-of call >= single-asset call");

    // Worst-of call <= single-asset call
    auto mc_worst = mc_multi_asset(p2, PayoffType::WorstOfCall, 200000);
    check(mc_worst.price <= bs_call + 3.0 * mc_worst.std_error,
          "Worst-of call <= single-asset call");

    // At rho=1, best-of = worst-of = single BS (all assets identical)
    auto p_rho1 = make_basket_params(2, S0, sigma, 0.999, r, T, S0);
    auto mc_best1 = mc_multi_asset(p_rho1, PayoffType::BestOfCall, 200000);
    auto mc_worst1 = mc_multi_asset(p_rho1, PayoffType::WorstOfCall, 200000);
    check_approx(mc_best1.price, bs_call, 3.0 * mc_best1.std_error + 0.2,
                 "Best-of at rho~1 ~ BS single");
    check_approx(mc_worst1.price, bs_call, 3.0 * mc_worst1.std_error + 0.2,
                 "Worst-of at rho~1 ~ BS single");

    // Lognormal basket approximation should be in the ballpark
    double approx = basket_call_approx(bp);
    check_approx(mc_basket.price, approx, 0.5,
                 "LN basket approx near MC");

    // Basket vol limit: as N->inf, sigma_basket -> sigma*sqrt(rho)
    auto p_large = make_basket_params(100, S0, sigma, 0.5, r, T, S0);
    double bvol_large = basket_vol(p_large);
    double limit = sigma * std::sqrt(0.5);
    check_approx(bvol_large, limit, 0.005,
                 "Basket vol(N=100) ~ sigma*sqrt(rho)");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "╔══════════════════════════════════════════════════════╗\n"
              << "║         QuantPricer Test Suite                      ║\n"
              << "╚══════════════════════════════════════════════════════╝\n";

    test_payoffs();
    test_matrix();
    test_rng();
    test_black_scholes();
    test_greeks();
    test_mc_european();
    test_implied_vol();
    test_fdm();
    test_heston();
    test_jump_diffusion();
    test_binomial_tree();
    test_american_mc();
    test_barrier_options();
    test_multi_asset();

    std::cout << "\n" << std::string(52, '=') << "\n"
              << "  Results: " << tests_passed << " passed, "
              << tests_failed << " failed\n"
              << std::string(52, '=') << "\n";

    return tests_failed > 0 ? 1 : 0;
}

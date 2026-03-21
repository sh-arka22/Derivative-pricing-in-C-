// ============================================================================
// Day 12 Example: Risk Management — VaR, CVaR & Portfolio Risk
//
// Demonstrates:
// 1. Parametric vs Monte Carlo VaR comparison
// 2. VaR subadditivity violation (why VaR is not coherent)
// 3. Portfolio risk decomposition (marginal & component VaR)
// 4. Performance metrics (Sharpe, Sortino, Calmar, max drawdown)
// 5. Stress testing scenarios
// 6. Confidence level sensitivity
// ============================================================================

#include "risk/risk.h"
#include "multi_asset/multi_asset.h"
#include <iostream>
#include <iomanip>

using namespace risk;

int main() {
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "======================================================================\n"
              << "  Day 12: Risk Management — VaR, CVaR & Portfolio Risk\n"
              << "======================================================================\n\n";

    // ================================================================
    // 1. Parametric vs Monte Carlo VaR
    // ================================================================
    std::cout << "=== 1. Parametric vs Monte Carlo VaR ===\n\n";

    // 3-asset portfolio: equities, bonds, commodities
    std::vector<double> w = {0.6, 0.3, 0.1};  // 60/30/10 allocation
    std::vector<double> mu = {0.0004, 0.0001, 0.0003};  // daily expected returns
    std::vector<double> sigma = {0.015, 0.005, 0.020};  // daily vols

    // Correlation matrix
    QMatrix<double> corr(3, 3);
    corr(0,0)=1.0;  corr(0,1)=0.3;  corr(0,2)=0.2;
    corr(1,0)=0.3;  corr(1,1)=1.0;  corr(1,2)=-0.1;
    corr(2,0)=0.2;  corr(2,1)=-0.1; corr(2,2)=1.0;

    auto cov = build_cov_matrix(sigma, corr);

    std::cout << "Portfolio: 60% Equity, 30% Bonds, 10% Commodities\n";
    std::cout << "Daily vols: " << sigma[0]*100 << "%, "
              << sigma[1]*100 << "%, " << sigma[2]*100 << "%\n\n";

    std::cout << std::setw(14) << "Confidence"
              << std::setw(14) << "Param VaR"
              << std::setw(14) << "Param CVaR"
              << std::setw(14) << "MC VaR"
              << std::setw(14) << "MC CVaR" << "\n";
    std::cout << std::string(70, '-') << "\n";

    for (double alpha : {0.90, 0.95, 0.99, 0.995}) {
        auto pvar = portfolio_parametric_var(w, cov, mu, alpha);
        auto mcvar = mc_portfolio_var(w, mu, cov, alpha, 500000);

        std::cout << std::setw(13) << (alpha * 100) << "%"
                  << std::setw(14) << (pvar.var * 100) << "%"
                  << std::setw(14) << (pvar.cvar * 100) << "%"
                  << std::setw(14) << (mcvar.var * 100) << "%"
                  << std::setw(14) << (mcvar.cvar * 100) << "%\n";
    }
    std::cout << "\nNote: MC matches parametric well (both assume normal returns here)\n"
              << "In practice, MC captures fat tails that parametric misses.\n";

    // ================================================================
    // 2. VaR Subadditivity Violation
    // ================================================================
    std::cout << "\n=== 2. VaR Subadditivity Violation (Interview Classic) ===\n\n";

    auto demo = demonstrate_subadditivity();
    std::cout << "Two independent binary bets (lose $96 with 4% prob, gain $4 with 96%)\n";
    std::cout << "At 95% confidence:\n";
    std::cout << "  VaR(A)   = " << std::setw(8) << demo.var_A << "\n";
    std::cout << "  VaR(B)   = " << std::setw(8) << demo.var_B << "\n";
    std::cout << "  VaR(A+B) = " << std::setw(8) << demo.var_AB << "\n";
    std::cout << "  VaR(A) + VaR(B) = " << (demo.var_A + demo.var_B) << "\n";
    std::cout << "\n  Subadditivity violated: " << (demo.violated ? "YES" : "NO") << "\n";
    std::cout << "  VaR(A+B) > VaR(A) + VaR(B) means VaR PENALISES diversification!\n";
    std::cout << "  This is why Basel III switched to Expected Shortfall (CVaR).\n";

    // ================================================================
    // 3. Portfolio Risk Decomposition
    // ================================================================
    std::cout << "\n=== 3. Portfolio Risk Decomposition (Euler) ===\n\n";

    auto decomp = decompose_var(w, cov, 0.99);

    std::cout << std::setw(12) << "Asset"
              << std::setw(10) << "Weight"
              << std::setw(14) << "Marg VaR"
              << std::setw(14) << "Comp VaR"
              << std::setw(14) << "% of Total" << "\n";
    std::cout << std::string(64, '-') << "\n";

    const char* names[] = {"Equity", "Bonds", "Commodity"};
    for (size_t i = 0; i < 3; ++i) {
        double pct = 100.0 * decomp.component_var[i] / decomp.total_var;
        std::cout << std::setw(12) << names[i]
                  << std::setw(9) << (w[i]*100) << "%"
                  << std::setw(14) << (decomp.marginal_var[i]*100) << "%"
                  << std::setw(14) << (decomp.component_var[i]*100) << "%"
                  << std::setw(13) << pct << "%\n";
    }
    std::cout << std::setw(12) << "TOTAL" << std::setw(10) << ""
              << std::setw(14) << "" << std::setw(14) << (decomp.total_var*100) << "%\n";

    std::cout << "\nKey insight: Equity is 60% of weight but contributes ~"
              << std::setprecision(0)
              << (100.0 * decomp.component_var[0] / decomp.total_var) << "% of risk\n";
    std::cout << std::setprecision(4);

    // ================================================================
    // 4. Performance Metrics
    // ================================================================
    std::cout << "\n=== 4. Performance Metrics (Simulated 2-year daily returns) ===\n\n";

    // Simulate 504 days of correlated returns
    auto returns = simulate_returns(mu, cov, 504, 42);

    // Compute portfolio returns
    std::vector<double> port_returns(504);
    for (size_t d = 0; d < 504; ++d) {
        port_returns[d] = 0.0;
        for (size_t i = 0; i < 3; ++i)
            port_returns[d] += w[i] * returns[i][d];
    }

    auto metrics = compute_metrics(port_returns, 0.0, 252.0);

    std::cout << "  Annualised Return:  " << std::setw(10) << (metrics.mean_return*100) << "%\n"
              << "  Annualised Vol:     " << std::setw(10) << (metrics.volatility*100) << "%\n"
              << "  Sharpe Ratio:       " << std::setw(10) << metrics.sharpe << "\n"
              << "  Sortino Ratio:      " << std::setw(10) << metrics.sortino << "\n"
              << "  Max Drawdown:       " << std::setw(10) << (metrics.max_drawdown*100) << "%\n"
              << "  Calmar Ratio:       " << std::setw(10) << metrics.calmar << "\n"
              << "  Skewness:           " << std::setw(10) << metrics.skewness << "\n"
              << "  Excess Kurtosis:    " << std::setw(10) << metrics.kurtosis << "\n";

    // Also compute per-asset metrics
    std::cout << "\n  Per-asset comparison:\n";
    std::cout << std::setw(12) << "Asset"
              << std::setw(10) << "Return"
              << std::setw(10) << "Vol"
              << std::setw(10) << "Sharpe"
              << std::setw(10) << "MaxDD" << "\n";
    std::cout << "  " << std::string(50, '-') << "\n";

    for (size_t i = 0; i < 3; ++i) {
        auto m = compute_metrics(returns[i], 0.0, 252.0);
        std::cout << std::setw(12) << names[i]
                  << std::setw(9) << (m.mean_return*100) << "%"
                  << std::setw(9) << (m.volatility*100) << "%"
                  << std::setw(10) << m.sharpe
                  << std::setw(9) << (m.max_drawdown*100) << "%\n";
    }
    auto pm = compute_metrics(port_returns, 0.0, 252.0);
    std::cout << std::setw(12) << "Portfolio"
              << std::setw(9) << (pm.mean_return*100) << "%"
              << std::setw(9) << (pm.volatility*100) << "%"
              << std::setw(10) << pm.sharpe
              << std::setw(9) << (pm.max_drawdown*100) << "%\n";

    // ================================================================
    // 5. Stress Testing
    // ================================================================
    std::cout << "\n=== 5. Stress Testing ===\n\n";

    std::vector<std::pair<std::string, std::vector<double>>> scenarios = {
        {"2008 Crisis",      {-0.40, 0.05, -0.30}},
        {"Rates +200bp",     {-0.10, -0.15, -0.05}},
        {"Equity Crash -20%",{-0.20, 0.03, -0.10}},
        {"Commodity Spike",  {-0.05, -0.02, 0.30}},
        {"Risk-Off Flight",  {-0.15, 0.10, -0.20}},
        {"Bull Market",      {0.25, 0.03, 0.15}},
    };

    auto stress = stress_test(w, scenarios);

    std::cout << std::setw(22) << "Scenario"
              << std::setw(10) << "Equity"
              << std::setw(10) << "Bonds"
              << std::setw(10) << "Commod"
              << std::setw(14) << "Portfolio" << "\n";
    std::cout << std::string(66, '-') << "\n";

    for (auto& s : stress) {
        std::cout << std::setw(22) << s.name;
        for (auto r : s.asset_returns)
            std::cout << std::setw(9) << (r*100) << "%";
        std::cout << std::setw(13) << (s.portfolio_pnl*100) << "%\n";
    }

    // ================================================================
    // 6. VaR vs Confidence Level
    // ================================================================
    std::cout << "\n=== 6. VaR Sensitivity to Confidence Level ===\n\n";

    std::cout << std::setw(14) << "Confidence"
              << std::setw(14) << "VaR ($)"
              << std::setw(14) << "CVaR ($)"
              << std::setw(14) << "CVaR/VaR" << "\n";
    std::cout << std::string(56, '-') << "\n";

    double portfolio_value = 10000000;  // $10M portfolio
    for (double alpha : {0.90, 0.95, 0.975, 0.99, 0.995, 0.999}) {
        auto v = portfolio_parametric_var(w, cov, mu, alpha);
        double var_dollar = v.var * portfolio_value;
        double cvar_dollar = v.cvar * portfolio_value;
        std::cout << std::setw(13) << (alpha*100) << "%"
                  << std::setw(12) << std::setprecision(0) << var_dollar << "  "
                  << std::setw(12) << cvar_dollar << "  "
                  << std::setprecision(4)
                  << std::setw(12) << cvar_dollar / var_dollar << "\n";
    }

    std::cout << "\nCVaR/VaR ratio increases with confidence level — tail risk matters more.\n"
              << "At 99.9%, CVaR is ~" << std::setprecision(1);

    auto v999 = portfolio_parametric_var(w, cov, mu, 0.999);
    auto v99 = portfolio_parametric_var(w, cov, mu, 0.99);
    std::cout << (v999.cvar / v99.var) << "x the 99% VaR.\n";

    return 0;
}

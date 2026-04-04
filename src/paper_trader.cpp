// ============================================================================
// Paper Trader — Entry Point (Self-Simulated)
//
// No broker, no network. Replays historical CSV data through a simulated
// exchange built on the existing orderbook::OrderBook matching engine.
//
// Day 1 (revised): Verify config loads and all engine deps compile
//                   without Boost.Beast or OpenSSL.
// ============================================================================

#include "trading/config.h"

// Verify quantpricer headers work
#include "greeks/black_scholes.h"
#include "greeks/greeks_engine.h"
#include "risk/risk.h"
#include "vol/implied_vol.h"
#include "orderbook/orderbook.h"
#include "payoff/payoff.h"

#include <iostream>
#include <iomanip>
#include <string>

int main(int argc, char* argv[]) {
    std::cout << "=== QuantPricer Paper Trader (Self-Simulated) ===" << std::endl;

    // ----------------------------------------------------------------
    // 1. Load config
    // ----------------------------------------------------------------
    std::string config_path = "config/paper_trading.json";
    if (argc > 1) {
        config_path = argv[1];
    }

    trading::Config config;
    try {
        config = trading::Config::from_json_file(config_path);
        config.print_summary();
    } catch (const std::exception& e) {
        std::cerr << "CONFIG ERROR: " << e.what() << std::endl;
        return 1;
    }

    // ----------------------------------------------------------------
    // 2. Verify pricing engine
    // ----------------------------------------------------------------
    std::cout << std::fixed << std::setprecision(4);
    double call = bs::call_price(100.0, 100.0, 0.05, 1.0, 0.2);
    double delta = bs::delta_call(100.0, 100.0, 0.05, 1.0, 0.2);
    std::cout << "\n[ENGINE] BS Call = " << call
              << ", Delta = " << delta << std::endl;

    auto greeks = analytic_greeks_call(100.0, 100.0, 0.05, 1.0, 0.2);
    std::cout << "[ENGINE] Greeks: d=" << greeks.delta
              << " g=" << greeks.gamma
              << " v=" << greeks.vega
              << " t=" << greeks.theta
              << " r=" << greeks.rho << std::endl;

    // ----------------------------------------------------------------
    // 3. Verify risk engine
    // ----------------------------------------------------------------
    auto var = risk::parametric_var(0.0, 0.02, 0.99);
    std::cout << "[ENGINE] VaR(99%) = " << var.var
              << ", CVaR = " << var.cvar << std::endl;

    // ----------------------------------------------------------------
    // 4. Verify order book (simulated exchange core)
    // ----------------------------------------------------------------
    orderbook::OrderBook book("TEST");
    book.add_limit_order(orderbook::Side::Buy, 99.50, 100);
    book.add_limit_order(orderbook::Side::Buy, 99.40, 200);
    book.add_limit_order(orderbook::Side::Sell, 100.50, 100);
    book.add_limit_order(orderbook::Side::Sell, 100.60, 200);
    std::cout << "[ENGINE] OrderBook: bid=" << book.best_bid()
              << " ask=" << book.best_ask()
              << " spread=" << book.spread() << std::endl;

    // Test a market order fill
    book.add_market_order(orderbook::Side::Buy, 50);
    std::cout << "[ENGINE] After MKT BUY 50: trades=" << book.trade_count()
              << " last_price=" << book.trades().back().price << std::endl;

    // ----------------------------------------------------------------
    // 5. Verify implied vol
    // ----------------------------------------------------------------
    auto iv = implied_vol_newton(call, 100.0, 100.0, 0.05, 1.0);
    std::cout << "[ENGINE] IV recovery: " << iv.sigma
              << " (expected 0.2, converged=" << iv.converged << ")" << std::endl;

    std::cout << "\nAll systems operational. Ready for Day 2." << std::endl;
    return 0;
}

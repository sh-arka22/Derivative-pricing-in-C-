// ============================================================================
// Paper Trader — Entry Point (Self-Simulated)
//
// No broker, no network. Replays historical CSV data through a simulated
// exchange built on the existing orderbook::OrderBook matching engine.
//
// Day 1: Verify config loads and all engine deps compile
// Day 2: Verify MarketDataReplay, OptionDataReplay, MarketSnapshot,
//         find_best_option, fill_option_at_market
// ============================================================================

#include "trading/config.h"
#include "trading/market_data.h"
#include "trading/market_snapshot.h"
#include "trading/option_types.h"

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

    // ================================================================
    // Day 2: Market Data Replay + Option Data + MarketSnapshot
    // ================================================================

    std::cout << "\n=== Day 2: Market Data & Option Chain Verification ===" << std::endl;

    // ----------------------------------------------------------------
    // 6. Load OHLCV data via MarketDataReplay
    // ----------------------------------------------------------------
    trading::MarketDataReplay replay(config.data_files);
    replay.print_summary();

    // ----------------------------------------------------------------
    // 7. Load option chain data via OptionDataReplay
    // ----------------------------------------------------------------
    trading::OptionDataReplay option_replay(config.option_data_files);
    option_replay.print_summary();

    // ----------------------------------------------------------------
    // 8. Build MarketSnapshot for first 3 dates
    // ----------------------------------------------------------------
    std::cout << "\nFirst 3 MarketSnapshots:\n";
    int count = 0;
    while (replay.has_next() && count < 3) {
        auto bars = replay.next();
        std::string date = bars.begin()->second.date;

        // Build unified snapshot
        trading::MarketSnapshot snapshot;
        snapshot.date = date;
        snapshot.bars = bars;
        if (!config.option_data_files.empty()) {
            snapshot.option_chains = option_replay.get_chains(date);
        }

        std::cout << "\n[" << date << "] "
                  << snapshot.num_symbols() << " symbols, "
                  << snapshot.num_option_symbols() << " with options, "
                  << snapshot.total_option_quotes() << " option quotes\n";

        for (auto& [sym, bar] : bars) {
            std::cout << "  " << sym
                      << "  O=" << bar.open << " H=" << bar.high
                      << " L=" << bar.low << " C=" << bar.close
                      << " V=" << bar.volume;

            // Show option chain summary if available
            if (snapshot.has_options(sym)) {
                auto& chain = snapshot.option_chains.at(sym);
                auto exps = chain.expirations();
                std::cout << "  | opts=" << chain.size()
                          << " exps=" << exps.size();

                // Find ATM call
                auto* atm = chain.atm_call(bar.close);
                if (atm) {
                    std::cout << " | ATM call: $" << atm->strike
                              << " bid=" << atm->bid << " ask=" << atm->ask
                              << " iv=" << std::setprecision(2) << atm->implied_vol
                              << " d=" << atm->delta;
                    std::cout << std::setprecision(4);
                }
            }
            std::cout << "\n";
        }
        count++;
    }

    // ----------------------------------------------------------------
    // 9. Test find_best_option (ATM call ~30 DTE)
    // ----------------------------------------------------------------
    std::cout << "\n--- find_best_option test ---\n";

    // Reset replay to first date
    trading::MarketDataReplay replay2(config.data_files);
    auto first_bars = replay2.next();
    std::string first_date = first_bars.begin()->second.date;

    trading::MarketSnapshot snap;
    snap.date = first_date;
    snap.bars = first_bars;
    snap.option_chains = option_replay.get_chains(first_date);

    for (auto& [sym, bar] : first_bars) {
        if (!snap.has_options(sym)) continue;

        auto& chain = snap.option_chains.at(sym);
        auto best = trading::find_best_option(chain, bar.close, 30, true);

        if (best.has_value()) {
            auto& q = best.value();
            int dte = q.days_to_expiry_from(first_date);
            std::cout << "[FIND] " << sym << " spot=$" << bar.close
                      << " -> best call: $" << q.strike
                      << " exp=" << q.expiration
                      << " dte=" << dte
                      << " bid=" << q.bid << " ask=" << q.ask
                      << " iv=" << std::setprecision(2) << q.implied_vol
                      << " delta=" << q.delta << "\n";
            std::cout << std::setprecision(4);
        }
    }

    // ----------------------------------------------------------------
    // 10. Test fill_option_at_market
    // ----------------------------------------------------------------
    std::cout << "\n--- fill_option_at_market test ---\n";

    // Pick first symbol that has options
    for (auto& [sym, bar] : first_bars) {
        if (!snap.has_options(sym)) continue;

        auto& chain = snap.option_chains.at(sym);
        auto best = trading::find_best_option(chain, bar.close, 30, true);
        if (!best.has_value()) continue;

        auto& q = best.value();

        // Create a buy order for this option
        trading::OptionOrderRequest buy_req;
        buy_req.symbol     = sym;
        buy_req.expiration = q.expiration;
        buy_req.strike     = q.strike;
        buy_req.is_call    = true;
        buy_req.contracts  = 5;
        buy_req.is_buy     = true;

        auto buy_fill = trading::fill_option_at_market(buy_req, snap, first_date);
        if (buy_fill.has_value()) {
            auto& f = buy_fill.value();
            std::cout << "[FILL] BUY " << f.contracts << "x "
                      << f.symbol << " $" << f.strike
                      << (f.is_call ? "C" : "P")
                      << " exp=" << f.expiration
                      << " @ $" << f.price << " (ask)"
                      << " cost=$" << std::setprecision(2)
                      << (f.price * f.contracts * 100.0) << "\n";
            std::cout << std::setprecision(4);
        }

        // Create a sell order for the same option
        trading::OptionOrderRequest sell_req = buy_req;
        sell_req.is_buy = false;

        auto sell_fill = trading::fill_option_at_market(sell_req, snap, first_date);
        if (sell_fill.has_value()) {
            auto& f = sell_fill.value();
            std::cout << "[FILL] SELL " << f.contracts << "x "
                      << f.symbol << " $" << f.strike
                      << (f.is_call ? "C" : "P")
                      << " exp=" << f.expiration
                      << " @ $" << f.price << " (bid)"
                      << " spread cost=$" << std::setprecision(2)
                      << ((buy_fill.value().price - f.price) * f.contracts * 100.0)
                      << "\n";
            std::cout << std::setprecision(4);
        }

        break;  // one symbol is enough for verification
    }

    std::cout << "\nAll Day 2 systems operational. Ready for Day 3." << std::endl;
    return 0;
}

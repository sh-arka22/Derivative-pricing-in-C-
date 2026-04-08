// ============================================================================
// Paper Trader — Entry Point (Self-Simulated)
//
// No broker, no network. Replays historical CSV data through a simulated
// exchange built on the existing orderbook::OrderBook matching engine.
//
// Day 1: Verify config loads and all engine deps compile
// Day 2: Verify MarketDataReplay, OptionDataReplay, MarketSnapshot,
//         find_best_option, fill_option_at_market
// Day 3: Verify SimulatedExchange — seed liquidity, submit orders, get fills
// Day 4: Verify PositionTracker — stock + option P&L, equity invariant
// ============================================================================

#include "trading/config.h"
#include "trading/market_data.h"
#include "trading/market_snapshot.h"
#include "trading/option_types.h"
#include "trading/simulated_exchange.h"
#include "trading/position_tracker.h"

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

    // ================================================================
    // Day 3: Simulated Exchange — Seed Liquidity + Execute Orders
    // ================================================================

    std::cout << "\n=== Day 3: Simulated Exchange Verification ===" << std::endl;

    trading::SimulatedExchange exchange(config.sim);

    // Use first date's bars to seed and test
    for (auto& [sym, bar] : first_bars) {
        exchange.seed_liquidity(sym, bar);
        auto& ebook = exchange.get_book(sym);

        std::cout << "\n[EXCHANGE] " << sym << " seeded around $" << bar.close
                  << "  bid=" << std::setprecision(2) << ebook.best_bid()
                  << " ask=" << ebook.best_ask()
                  << " spread=$" << ebook.spread()
                  << " (" << std::setprecision(1) << (ebook.spread() / bar.close * 10000.0)
                  << " bps)\n";
        std::cout << std::setprecision(4);

        // Show book depth
        auto bids = ebook.get_bids(5);
        auto asks = ebook.get_asks(5);
        for (auto ait = asks.rbegin(); ait != asks.rend(); ++ait) {
            std::cout << "    ASK $" << std::setprecision(2) << ait->price
                      << "  x " << ait->quantity << "\n";
        }
        std::cout << "    ---- spread ----\n";
        for (auto& b : bids) {
            std::cout << "    BID $" << std::setprecision(2) << b.price
                      << "  x " << b.quantity << "\n";
        }
        std::cout << std::setprecision(4);

        // Test market buy
        trading::OrderRequest buy_order{sym, orderbook::Side::Buy, 50, 0.0};
        auto buy_fills = exchange.submit_order(buy_order, first_date);
        double total_cost = 0;
        int total_qty = 0;
        for (auto& f : buy_fills) {
            total_cost += f.price * f.quantity;
            total_qty += f.quantity;
        }
        std::cout << "[EXCHANGE] MKT BUY 50 " << sym << " → "
                  << buy_fills.size() << " fill(s), "
                  << total_qty << " shares, avg=$"
                  << std::setprecision(2) << (total_qty > 0 ? total_cost / total_qty : 0)
                  << "\n";

        // Test market sell
        trading::OrderRequest sell_order{sym, orderbook::Side::Sell, 150, 0.0};
        auto sell_fills = exchange.submit_order(sell_order, first_date);
        total_cost = 0;
        total_qty = 0;
        for (auto& f : sell_fills) {
            total_cost += f.price * f.quantity;
            total_qty += f.quantity;
        }
        std::cout << "[EXCHANGE] MKT SELL 150 " << sym << " → "
                  << sell_fills.size() << " fill(s), "
                  << total_qty << " shares, avg=$"
                  << std::setprecision(2) << (total_qty > 0 ? total_cost / total_qty : 0)
                  << "\n";
        std::cout << std::setprecision(4);

        // Test limit order (should rest if price is away from market)
        double away_price = bar.close - bar.close * 0.01;  // 1% below close
        trading::OrderRequest limit_order{sym, orderbook::Side::Buy, 100, away_price};
        auto limit_fills = exchange.submit_order(limit_order, first_date);
        std::cout << "[EXCHANGE] LMT BUY 100 " << sym << " @ $"
                  << std::setprecision(2) << away_price
                  << " → " << limit_fills.size() << " fills (expected 0 — resting)\n";
        std::cout << std::setprecision(4);

        break;  // one symbol is enough
    }

    // ================================================================
    // Day 4: Position Tracker — Stock + Option P&L Accounting
    // ================================================================

    std::cout << "\n=== Day 4: Position Tracker Verification ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);

    trading::PositionTracker tracker(config.sim.initial_capital);

    // ----------------------------------------------------------------
    // 11. Stock position P&L — all 4 cases
    // ----------------------------------------------------------------

    // Case 1: Open from flat — BUY 100 AAPL @ 185.00
    trading::Fill buy1{"AAPL", orderbook::Side::Buy, 185.00, 100, "2025-01-02"};
    tracker.on_fill(buy1);
    tracker.on_price("AAPL", 185.00);
    std::cout << "\n[CASE 1] BUY 100 AAPL @ 185.00 (open from flat):\n";
    tracker.print_positions();
    // Expected: qty=100, avg=185.00, cash=81500, equity=100000

    // Mark-to-market: price moves up
    tracker.on_price("AAPL", 190.00);
    std::cout << "\n[MTM] Price -> 190.00:\n";
    tracker.print_positions();
    // Expected: unrealized = 100 * (190 - 185) = +500, equity = 100500

    // Case 2: Add to position — BUY 50 more @ 192.00
    trading::Fill buy2{"AAPL", orderbook::Side::Buy, 192.00, 50, "2025-01-02"};
    tracker.on_fill(buy2);
    tracker.on_price("AAPL", 192.00);
    std::cout << "\n[CASE 2] BUY 50 AAPL @ 192.00 (add to long):\n";
    tracker.print_positions();
    // Expected: qty=150, avg = (185*100 + 192*50)/150 = 187.33

    // Case 3: Partial close — SELL 60 @ 190.00
    trading::Fill sell1{"AAPL", orderbook::Side::Sell, 190.00, 60, "2025-01-03"};
    tracker.on_fill(sell1);
    std::cout << "\n[CASE 3] SELL 60 AAPL @ 190.00 (partial close):\n";
    tracker.print_positions();
    // Expected: qty=90, avg still 187.33, realized=60*(190-187.33)=160

    // Case 4: Reverse through zero — SELL 200 @ 188.00
    // Closes remaining 90 long + opens 110 short
    trading::Fill sell2{"AAPL", orderbook::Side::Sell, 188.00, 200, "2025-01-03"};
    tracker.on_fill(sell2);
    tracker.on_price("AAPL", 188.00);
    std::cout << "\n[CASE 4] SELL 200 AAPL @ 188.00 (reverse through zero):\n";
    tracker.print_positions();
    // Expected: qty=-110, avg=188.00, realized=160+90*(188-187.33)=220

    // Verify equity invariant
    double expected_equity = tracker.cash() + tracker.get_position("AAPL").quantity
                             * tracker.get_position("AAPL").market_price;
    std::cout << "\n[INVARIANT] equity=" << tracker.total_equity()
              << "  cash+pos=" << expected_equity
              << "  match=" << (std::abs(tracker.total_equity() - expected_equity) < 0.01
                                ? "YES" : "NO") << "\n";

    // End of day snapshot
    tracker.on_day_end("2025-01-02");
    tracker.on_day_end("2025-01-03");
    auto returns = tracker.daily_return_series();
    std::cout << "[EQUITY] History snapshots: " << tracker.equity_history().size()
              << "  Daily returns: " << returns.size() << "\n";
    if (!returns.empty()) {
        std::cout << "[EQUITY] Day 1->2 return: " << std::setprecision(4)
                  << (returns[0] * 100.0) << "%\n";
    }

    // ----------------------------------------------------------------
    // 12. Option position P&L
    // ----------------------------------------------------------------
    std::cout << std::setprecision(2);
    std::cout << "\n--- Option Position Tests ---\n";

    // Use real option data if available
    trading::MarketDataReplay replay3(config.data_files);
    auto test_bars = replay3.next();
    std::string test_date = test_bars.begin()->second.date;

    trading::MarketSnapshot opt_snap;
    opt_snap.date = test_date;
    opt_snap.bars = test_bars;
    opt_snap.option_chains = option_replay.get_chains(test_date);

    bool tested_option = false;
    for (auto& [sym, bar] : test_bars) {
        if (!opt_snap.has_options(sym)) continue;

        auto& chain = opt_snap.option_chains.at(sym);
        auto best = trading::find_best_option(chain, bar.close, 30, true);
        if (!best.has_value()) continue;

        auto& q = best.value();

        // Buy 5 contracts of the ATM call
        trading::OptionOrderRequest buy_req;
        buy_req.symbol     = sym;
        buy_req.expiration = q.expiration;
        buy_req.strike     = q.strike;
        buy_req.is_call    = true;
        buy_req.contracts  = 5;
        buy_req.is_buy     = true;

        auto buy_fill = trading::fill_option_at_market(buy_req, opt_snap, test_date);
        if (!buy_fill.has_value()) continue;

        tracker.on_option_fill(buy_fill.value());

        // Update marks from chain
        tracker.update_option_marks(chain);

        std::cout << "[OPT BUY] " << buy_fill->contracts << "x "
                  << buy_fill->symbol << " $" << buy_fill->strike
                  << (buy_fill->is_call ? "C" : "P")
                  << " @ $" << buy_fill->price << "\n";
        tracker.print_positions();

        // Check net delta
        std::cout << "[NET DELTA] " << sym << ": " << tracker.net_delta(sym)
                  << " (stock=" << tracker.get_position(sym).quantity
                  << " + option delta)\n";

        // Portfolio Greeks
        auto greeks_sum = tracker.portfolio_greeks();
        std::cout << "[GREEKS] Portfolio: d=" << greeks_sum.delta
                  << " g=" << greeks_sum.gamma
                  << " t=" << greeks_sum.theta
                  << " v=" << greeks_sum.vega << "\n";

        // Sell to close — verify realized P&L
        trading::OptionOrderRequest sell_req = buy_req;
        sell_req.is_buy = false;
        sell_req.contracts = 3;  // partial close

        auto sell_fill = trading::fill_option_at_market(sell_req, opt_snap, test_date);
        if (sell_fill.has_value()) {
            tracker.on_option_fill(sell_fill.value());
            std::cout << "\n[OPT SELL] " << sell_fill->contracts << "x @ $"
                      << sell_fill->price << " (partial close)\n";
            tracker.print_positions();
        }

        tested_option = true;
        break;
    }

    if (!tested_option) {
        std::cout << "[OPT] No option data available — skipping option tests\n";
    }

    // Final equity invariant check
    double final_equity = tracker.total_equity();
    double manual_equity = tracker.cash();
    for (auto& [sym, pos] : tracker.positions()) {
        manual_equity += pos.quantity * pos.market_price;
    }
    for (auto& [key, pos] : tracker.option_positions()) {
        manual_equity += pos.contracts * 100.0 * pos.mark;
    }
    std::cout << "\n[FINAL INVARIANT] equity=" << final_equity
              << "  manual=" << manual_equity
              << "  match=" << (std::abs(final_equity - manual_equity) < 0.01
                                ? "YES" : "NO") << "\n";

    std::cout << "\nTotal trades: " << tracker.total_trades()
              << "  Total P&L: $" << tracker.total_pnl() << "\n";

    std::cout << "\nAll Day 4 systems operational. Ready for Day 5." << std::endl;
    return 0;
}

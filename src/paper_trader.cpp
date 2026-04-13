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
#include "trading/strategy.h"
#include "trading/strategies/mean_reversion.h"
#include "trading/risk_guard.h"
#include "trading/logging.h"

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

    // ================================================================
    // Day 5: Strategy Base + Mean Reversion
    // ================================================================

    std::cout << "\n=== Day 5: Mean Reversion Strategy Verification ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);

    // Fresh components for strategy test
    trading::MarketDataReplay strat_replay(config.data_files);
    trading::OptionDataReplay strat_option_replay(config.option_data_files);
    trading::SimulatedExchange strat_exchange(config.sim);
    trading::PositionTracker strat_tracker(config.sim.initial_capital);

    auto strategy = std::make_unique<trading::MeanReversionStrategy>();
    strategy->init(config);

    std::cout << "[STRATEGY] " << strategy->name()
              << "  lookback=" << config.strategy_params.mr_lookback
              << "  entry_z=" << config.strategy_params.mr_entry_z
              << "  exit_z=" << config.strategy_params.mr_exit_z
              << "  size=" << config.strategy_params.mr_position_size << "\n";

    // Run first 30 bars
    int bar_count = 0;
    int signal_count = 0;
    int fill_count = 0;

    while (strat_replay.has_next() && bar_count < 30) {
        auto bars = strat_replay.next();

        // Build MarketSnapshot for this date
        trading::MarketSnapshot snapshot;
        snapshot.date = bars.begin()->second.date;
        snapshot.bars = bars;
        // option_chains left empty — equity-only strategy

        // Seed exchange + update tracker prices
        for (auto& [sym, bar] : bars) {
            strat_tracker.on_price(sym, bar.close);
            strat_exchange.seed_liquidity(sym, bar);
        }

        auto strategy_orders = strategy->on_bar(snapshot, strat_tracker, strat_exchange);

        for (auto& order : strategy_orders.stock_orders) {
            std::cout << "[" << snapshot.date << "] SIGNAL "
                      << order.symbol << " "
                      << (order.side == orderbook::Side::Buy ? "BUY" : "SELL")
                      << " " << order.quantity << "\n";
            signal_count++;

            auto fills = strat_exchange.submit_order(order, snapshot.date);
            for (auto& fill : fills) {
                strat_tracker.on_fill(fill);
                strategy->on_fill(fill);
                fill_count++;
            }
        }

        strat_tracker.on_day_end(snapshot.date);
        bar_count++;
    }

    std::cout << "\n--- Day 5 Results (30 bars) ---\n";
    std::cout << "[STATS] Bars processed: " << bar_count
              << "  Signals: " << signal_count
              << "  Fills: " << fill_count << "\n";
    strat_tracker.print_positions();

    // Equity curve check
    auto strat_returns = strat_tracker.daily_return_series();
    std::cout << "[EQUITY] Snapshots: " << strat_tracker.equity_history().size()
              << "  Returns: " << strat_returns.size() << "\n";
    if (strat_tracker.equity_history().size() >= 2) {
        std::cout << "[EQUITY] Start: $" << strat_tracker.equity_history().front()
                  << "  End: $" << strat_tracker.equity_history().back()
                  << "  P&L: $" << (strat_tracker.equity_history().back()
                                    - strat_tracker.equity_history().front()) << "\n";
    }

    // Invariant: equity = cash + positions
    double strat_equity = strat_tracker.total_equity();
    double strat_manual = strat_tracker.cash();
    for (auto& [sym, pos] : strat_tracker.positions()) {
        strat_manual += pos.quantity * pos.market_price;
    }
    std::cout << "[INVARIANT] equity=" << strat_equity
              << "  manual=" << strat_manual
              << "  match=" << (std::abs(strat_equity - strat_manual) < 0.01
                                ? "YES" : "NO") << "\n";

    // ================================================================
    // Day 6: Risk Guard — Pre-Trade Checks + Portfolio Monitoring
    // ================================================================

    std::cout << "\n=== Day 6: Risk Guard Verification ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);

    trading::RiskGuard risk_guard(config.risk_limits);
    trading::PositionTracker rg_tracker(100000.0);
    rg_tracker.on_price("AAPL", 185.0);

    std::cout << "[LIMITS] max_order=" << config.risk_limits.max_order_size
              << "  max_pos=$" << config.risk_limits.max_position_notional
              << "  max_port=$" << config.risk_limits.max_portfolio_notional
              << "  max_loss=$" << config.risk_limits.max_loss
              << "  max_var=$" << config.risk_limits.max_var_99 << "\n";

    // Test 1: Order within limits — should approve
    trading::OrderRequest good_order{"AAPL", orderbook::Side::Buy, 100, 0.0};
    auto r1 = risk_guard.check_order(good_order, rg_tracker);
    std::cout << "\n[TEST 1] BUY 100 AAPL ($18.5k): "
              << (r1.approved ? "APPROVED" : "REJECTED: " + r1.reason) << "\n";

    // Test 2: Order exceeds max_order_size (500) — should reject
    trading::OrderRequest big_order{"AAPL", orderbook::Side::Buy, 600, 0.0};
    auto r2 = risk_guard.check_order(big_order, rg_tracker);
    std::cout << "[TEST 2] BUY 600 AAPL (size>500): "
              << (r2.approved ? "APPROVED" : "REJECTED: " + r2.reason) << "\n";

    // Test 3: Order breaches position notional ($50k limit)
    // 300 * $185 = $55,500 > $50,000
    trading::OrderRequest notional_order{"AAPL", orderbook::Side::Buy, 300, 0.0};
    auto r3 = risk_guard.check_order(notional_order, rg_tracker);
    std::cout << "[TEST 3] BUY 300 AAPL ($55.5k>$50k): "
              << (r3.approved ? "APPROVED" : "REJECTED: " + r3.reason) << "\n";

    // Test 4: Simulate a large loss, then check kill switch
    // Buy 100 @ 185, price drops to 130 → unrealized = 100*(130-185) = -$5,500
    trading::Fill rg_buy{"AAPL", orderbook::Side::Buy, 185.0, 100, "2025-01-02"};
    rg_tracker.on_fill(rg_buy);
    rg_tracker.on_price("AAPL", 130.0);
    std::cout << "\n[TEST 4] After BUY 100 @ $185, price → $130:\n";
    std::cout << "  P&L = $" << rg_tracker.total_pnl()
              << "  (loss limit = -$" << config.risk_limits.max_loss << ")\n";

    trading::OrderRequest loss_order{"AAPL", orderbook::Side::Buy, 10, 0.0};
    auto r4 = risk_guard.check_order(loss_order, rg_tracker);
    std::cout << "  BUY 10 more: "
              << (r4.approved ? "APPROVED" : "REJECTED: " + r4.reason) << "\n";

    // Test 5: Monitor — should trigger kill switch
    auto status = risk_guard.monitor(rg_tracker);
    std::cout << "\n[TEST 5] Monitor:\n";
    std::cout << "  P&L=$" << status.total_pnl
              << "  Notional=$" << status.portfolio_notional
              << "  VaR=$" << status.portfolio_var
              << "  Kill=" << (status.kill_switch ? "YES" : "NO") << "\n";
    if (!status.message.empty()) {
        std::cout << "  " << status.message << "\n";
    }

    // Test 6: Batch filter — mix of good and bad orders
    std::cout << "\n[TEST 6] Batch check (3 orders):\n";
    std::vector<trading::OrderRequest> batch = {
        {"MSFT", orderbook::Side::Buy, 50, 0.0},   // no price set → notional=0, but loss limit blocks
        {"AAPL", orderbook::Side::Buy, 600, 0.0},   // size limit
        {"AAPL", orderbook::Side::Sell, 50, 0.0},   // reduces position, but loss limit
    };
    // Set MSFT price so notional check works
    rg_tracker.on_price("MSFT", 430.0);

    std::vector<std::pair<trading::OrderRequest, std::string>> rejections;
    auto approved = risk_guard.check_orders(batch, rg_tracker, &rejections);
    std::cout << "  Approved: " << approved.size()
              << "  Rejected: " << rejections.size() << "\n";
    for (auto& [order, reason] : rejections) {
        std::cout << "  REJECT " << order.symbol << " "
                  << (order.side == orderbook::Side::Buy ? "BUY" : "SELL")
                  << " " << order.quantity << ": " << reason << "\n";
    }

    // ================================================================
    // Day 7: Trade Logger — Terminal + CSV Output
    // ================================================================

    std::cout << "\n=== Day 7: Trade Logger Verification ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);

    // Fresh components for logger test
    trading::MarketDataReplay log_replay(config.data_files);
    trading::SimulatedExchange log_exchange(config.sim);
    trading::PositionTracker log_tracker(config.sim.initial_capital);
    trading::RiskGuard log_risk_guard(config.risk_limits);
    trading::TradeLogger logger(config.log_dir, config.verbose);

    auto log_strategy = std::make_unique<trading::MeanReversionStrategy>();
    log_strategy->init(config);

    logger.log_header(config);

    // Run 25 bars with full logger wired in
    int log_bar_count = 0;
    bool killed = false;

    while (log_replay.has_next() && log_bar_count < 25 && !killed) {
        auto bars = log_replay.next();
        std::string date = bars.begin()->second.date;

        // Update prices + seed exchange
        for (auto& [sym, bar] : bars) {
            log_tracker.on_price(sym, bar.close);
            log_exchange.seed_liquidity(sym, bar);
        }

        // Log bars
        logger.log_bar(bars, log_tracker, log_risk_guard);

        // Build snapshot
        trading::MarketSnapshot snapshot;
        snapshot.date = date;
        snapshot.bars = bars;

        // Strategy signals
        auto orders = log_strategy->on_bar(snapshot, log_tracker, log_exchange);

        // Log signals
        for (auto& order : orders.stock_orders) {
            logger.log_signal(order, date);
        }

        // Risk filter
        std::vector<std::pair<trading::OrderRequest, std::string>> rejections;
        auto approved = log_risk_guard.check_orders(
            orders.stock_orders, log_tracker, &rejections);

        // Log rejections
        for (auto& [order, reason] : rejections) {
            logger.log_risk_rejection(order, reason, date);
        }

        // Execute approved orders
        for (auto& order : approved) {
            auto fills = log_exchange.submit_order(order, date);
            for (auto& fill : fills) {
                log_tracker.on_fill(fill);
                log_strategy->on_fill(fill);
                logger.log_fill(fill, log_tracker);
            }
        }

        // Check kill switch
        auto risk_status = log_risk_guard.monitor(log_tracker);
        if (risk_status.kill_switch) {
            logger.log_kill_switch(risk_status, date);
            killed = true;
        }

        log_tracker.on_day_end(date);
        log_bar_count++;
    }

    // Session summary with performance metrics
    auto log_returns = log_tracker.daily_return_series();
    risk::PerformanceMetrics log_metrics = {};
    if (log_returns.size() >= 2) {
        log_metrics = risk::compute_metrics(log_returns);
    }
    logger.log_session_summary(log_tracker, log_metrics, log_strategy->name(), log_bar_count);

    // Verify CSV was written
    std::cout << "[VERIFY] CSV path: " << logger.csv_path() << "\n";
    std::cout << "[VERIFY] Fills logged: " << logger.fill_count()
              << "  Rejections: " << logger.rejection_count() << "\n";

    // Check CSV file exists and has content
    {
        std::ifstream check(logger.csv_path());
        if (check.is_open()) {
            std::string header;
            std::getline(check, header);
            int lines = 0;
            std::string line;
            while (std::getline(check, line)) {
                if (!line.empty() && line[0] != '#') lines++;
            }
            std::cout << "[VERIFY] CSV header: " << header << "\n";
            std::cout << "[VERIFY] CSV data lines: " << lines << "\n";
        } else {
            std::cout << "[VERIFY] ERROR: CSV file not found!\n";
        }
    }

    std::cout << "\nAll Day 7 systems operational. Ready for Day 8." << std::endl;
    return 0;
}

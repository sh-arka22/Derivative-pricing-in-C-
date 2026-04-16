// ============================================================================
// Paper Trader — Full Event Loop (Self-Simulated)
//
// No broker, no network. Replays historical CSV data through a simulated
// exchange built on the existing orderbook::OrderBook matching engine.
//
// Usage:
//   ./paper_trader [config_path] [--strategy name] [--bars N]
//
// Default:
//   config_path = "config/paper_trading.json"
//   strategy    = value from config JSON (default: "mean_reversion")
//   bars        = 0 (run all available bars)
// ============================================================================

#include "trading/config.h"
#include "trading/market_data.h"
#include "trading/market_snapshot.h"
#include "trading/option_types.h"
#include "trading/simulated_exchange.h"
#include "trading/position_tracker.h"
#include "trading/strategy.h"
#include "trading/risk_guard.h"
#include "trading/logging.h"

// Strategy implementations
#include "trading/strategies/mean_reversion.h"
// Day 9 will add:
// #include "trading/strategies/delta_hedge.h"
// #include "trading/strategies/momentum.h"

// Existing quantpricer modules (used by risk guard + strategies)
#include "greeks/black_scholes.h"
#include "risk/risk.h"
#include "orderbook/orderbook.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <memory>
#include <csignal>

// ============================================================================
// SIGINT handler for clean shutdown (Ctrl+C)
// ============================================================================

static volatile bool g_running = true;

static void signal_handler(int signum) {
    (void)signum;
    g_running = false;
    std::cout << "\n[SIGINT] Shutting down gracefully...\n";
}

// ============================================================================
// Strategy factory
// ============================================================================

static std::unique_ptr<trading::Strategy> create_strategy(const std::string& name) {
    if (name == "mean_reversion") {
        return std::make_unique<trading::MeanReversionStrategy>();
    }
    // Day 9 will add:
    // if (name == "delta_hedge")
    //     return std::make_unique<trading::DeltaHedgeStrategy>();
    // if (name == "momentum")
    //     return std::make_unique<trading::MomentumStrategy>();

    std::cerr << "Unknown strategy: " << name << "\n";
    std::cerr << "Available: mean_reversion\n";
    return nullptr;
}

// ============================================================================
// CLI argument parsing
// ============================================================================

struct CLIArgs {
    std::string config_path = "config/paper_trading.json";
    std::string strategy_override;
    int max_bars = 0;  // 0 = run all

    static CLIArgs parse(int argc, char* argv[]) {
        CLIArgs args;
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--strategy" && i + 1 < argc) {
                args.strategy_override = argv[++i];
            } else if (arg == "--bars" && i + 1 < argc) {
                args.max_bars = std::stoi(argv[++i]);
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: paper_trader [config.json] [--strategy name] [--bars N]\n";
                std::cout << "  config.json    Path to config file (default: config/paper_trading.json)\n";
                std::cout << "  --strategy     Override strategy from config\n";
                std::cout << "  --bars N       Limit to N bars (0 = all)\n";
                std::cout << "  Available strategies: mean_reversion\n";
                std::exit(0);
            } else if (arg[0] != '-') {
                args.config_path = arg;
            }
        }
        return args;
    }
};

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);

    std::cout << "=== QuantPricer Paper Trader ===" << std::endl;

    // ----------------------------------------------------------------
    // 1. Parse CLI args and load config
    // ----------------------------------------------------------------
    auto cli = CLIArgs::parse(argc, argv);

    trading::Config config;
    try {
        config = trading::Config::from_json_file(cli.config_path);
    } catch (const std::exception& e) {
        std::cerr << "CONFIG ERROR: " << e.what() << "\n";
        return 1;
    }

    if (!cli.strategy_override.empty()) {
        config.strategy = cli.strategy_override;
    }

    // ----------------------------------------------------------------
    // 2. Create components
    // ----------------------------------------------------------------
    trading::MarketDataReplay replay(config.data_files);
    if (replay.total_bars() == 0) {
        std::cerr << "ERROR: No market data loaded. Check data_files in config.\n";
        return 1;
    }

    trading::OptionDataReplay option_replay(config.option_data_files);
    trading::SimulatedExchange exchange(config.sim);
    trading::PositionTracker tracker(config.sim.initial_capital);
    trading::RiskGuard risk_guard(config.risk_limits);
    trading::TradeLogger logger(config.log_dir, config.verbose);

    auto strategy = create_strategy(config.strategy);
    if (!strategy) return 1;

    // ----------------------------------------------------------------
    // 3. Initialize
    // ----------------------------------------------------------------
    strategy->init(config);
    logger.log_header(config);
    replay.print_summary();
    if (!config.option_data_files.empty()) {
        option_replay.print_summary();
    }
    std::cout << "\nRunning strategy: " << strategy->name() << "\n\n";

    // ----------------------------------------------------------------
    // 4. Main event loop
    // ----------------------------------------------------------------
    size_t bars_processed = 0;
    std::string current_date;

    while (replay.has_next() && g_running) {
        if (cli.max_bars > 0 && static_cast<int>(bars_processed) >= cli.max_bars) break;

        auto bars = replay.next();
        bars_processed++;

        if (bars.empty()) continue;
        current_date = bars.begin()->second.date;

        // 4a. Build MarketSnapshot (equity bars + option chains)
        trading::MarketSnapshot snapshot;
        snapshot.date = current_date;
        snapshot.bars = bars;
        snapshot.option_chains = option_replay.get_chains(current_date);

        // 4b. Update marks and seed exchange
        for (auto& [sym, bar] : bars) {
            tracker.on_price(sym, bar.close);
            exchange.seed_liquidity(sym, bar);
        }

        // 4c. Check portfolio health (pre-strategy)
        auto risk_status = risk_guard.monitor(tracker);
        if (risk_status.kill_switch) {
            logger.log_kill_switch(risk_status, current_date);
            break;
        }

        // 4d. Strategy generates orders
        auto orders = strategy->on_bar(snapshot, tracker, exchange);

        // 4e. Log signals
        for (auto& order : orders.stock_orders) {
            logger.log_signal(order, current_date);
        }

        // 4f. Risk guard filters stock orders
        std::vector<std::pair<trading::OrderRequest, std::string>> rejections;
        auto approved = risk_guard.check_orders(
            orders.stock_orders, tracker, &rejections);

        for (auto& [order, reason] : rejections) {
            logger.log_risk_rejection(order, reason, current_date);
        }

        // 4g. Execute approved stock orders via simulated exchange
        for (auto& order : approved) {
            auto fills = exchange.submit_order(order, current_date);
            for (auto& fill : fills) {
                tracker.on_fill(fill);
                strategy->on_fill(fill);
                logger.log_fill(fill, tracker);
            }
        }

        // 4h. Execute option orders directly at market bid/ask
        for (auto& opt_order : orders.option_orders) {
            auto opt_fill = trading::fill_option_at_market(
                opt_order, snapshot, current_date);
            if (opt_fill.has_value()) {
                tracker.on_option_fill(opt_fill.value());
                strategy->on_option_fill(opt_fill.value());
                if (config.verbose) {
                    auto& f = opt_fill.value();
                    std::cout << std::fixed << std::setprecision(2);
                    std::cout << "[" << current_date << "] OPT   "
                              << (f.is_buy ? "BUY" : "SELL") << " "
                              << f.contracts << "x " << f.symbol
                              << " $" << f.strike
                              << (f.is_call ? "C" : "P")
                              << " @ $" << f.price << "\n";
                }
            }
        }

        // 4i. Update option marks from latest chain data
        for (auto& [sym, chain] : snapshot.option_chains) {
            tracker.update_option_marks(chain);
        }

        // 4j. Log bar summary (after all fills)
        logger.log_bar(bars, tracker, risk_guard);

        // 4k. End-of-day equity snapshot
        tracker.on_day_end(current_date);
    }

    // ----------------------------------------------------------------
    // 5. Session summary
    // ----------------------------------------------------------------
    strategy->on_stop();

    auto daily_returns = tracker.daily_return_series();
    risk::PerformanceMetrics metrics = {};
    if (daily_returns.size() >= 2) {
        metrics = risk::compute_metrics(daily_returns);
    }

    logger.log_session_summary(tracker, metrics, strategy->name(), bars_processed);

    // Print OU calibration if mean reversion
    auto* mr = dynamic_cast<trading::MeanReversionStrategy*>(strategy.get());
    if (mr) {
        mr->print_calibration();
    }

    std::cout << "CSV log: " << logger.csv_path() << "\n";

    return 0;
}

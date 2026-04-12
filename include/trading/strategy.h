#ifndef QUANTPRICER_TRADING_STRATEGY_H
#define QUANTPRICER_TRADING_STRATEGY_H

// ============================================================================
// Strategy — Abstract base class for all trading strategies
//
// Strategies are PASSIVE: they receive a MarketSnapshot and return order
// requests (both stock and option). The main loop handles execution through
// the exchange and risk guard.
//
// Lifecycle:
//   1. init(config)                        — one-time setup
//   2. on_bar(snapshot, tracker, exchange)  — called each bar, returns orders
//   3. on_fill(fill)                        — notification of stock fills
//   4. on_option_fill(fill)                 — notification of option fills
//   5. on_stop()                            — called at end of simulation
// ============================================================================

#include "trading/config.h"
#include "trading/market_data.h"
#include "trading/market_snapshot.h"
#include "trading/option_types.h"
#include "trading/simulated_exchange.h"
#include "trading/position_tracker.h"

#include <string>
#include <vector>
#include <map>
#include <memory>

namespace trading {

// ============================================================================
// StrategyOrders — Return type for on_bar(), bundles stock + option orders
// ============================================================================

struct StrategyOrders {
    std::vector<OrderRequest> stock_orders;
    std::vector<OptionOrderRequest> option_orders;
};

// ============================================================================
// Strategy — Abstract base class
// ============================================================================

class Strategy {
public:
    virtual ~Strategy() = default;

    // Called once before the main loop starts
    virtual void init(const Config& config) = 0;

    // Called each bar — returns stock and option orders to submit
    virtual StrategyOrders on_bar(
        const MarketSnapshot& snapshot,
        const PositionTracker& tracker,
        const SimulatedExchange& exchange) = 0;

    // Called when a stock fill occurs (optional — for strategy-level bookkeeping)
    virtual void on_fill(const Fill& fill) { (void)fill; }

    // Called when an option fill occurs (optional)
    virtual void on_option_fill(const OptionFill& fill) { (void)fill; }

    // Called at end of simulation (optional — for cleanup/reporting)
    virtual void on_stop() {}

    // Strategy name for display and CLI selection
    virtual std::string name() const = 0;
};

} // namespace trading

#endif // QUANTPRICER_TRADING_STRATEGY_H

#ifndef QUANTPRICER_TRADING_MARKET_SNAPSHOT_H
#define QUANTPRICER_TRADING_MARKET_SNAPSHOT_H

// ============================================================================
// MarketSnapshot — All market data for a single date
//
// Strategies receive one MarketSnapshot per bar. It contains equity bars for
// every active symbol AND (optionally) option chains for symbols that have
// option data on that date.
//
// This is the SINGLE INPUT to strategy->on_bar(). Without it, every strategy
// would need to juggle separate bar maps and option chain maps independently.
// ============================================================================

#include "trading/market_data.h"   // Bar, OptionChain

#include <string>
#include <map>
#include <stdexcept>

namespace trading {

struct MarketSnapshot {
    std::string date;                                    // "YYYY-MM-DD"
    std::map<std::string, Bar> bars;                     // symbol -> equity bar
    std::map<std::string, OptionChain> option_chains;    // symbol -> option chain (may be empty)

    // Returns true if option chain data exists for the given symbol on this date
    bool has_options(const std::string& symbol) const {
        auto it = option_chains.find(symbol);
        return it != option_chains.end() && !it->second.empty();
    }

    // Shorthand for bars.at(symbol).close — throws if symbol not present
    double spot(const std::string& symbol) const {
        return bars.at(symbol).close;
    }

    // Number of symbols with bars on this date
    size_t num_symbols() const { return bars.size(); }

    // Number of symbols with option chain data on this date
    size_t num_option_symbols() const {
        size_t count = 0;
        for (auto& [sym, chain] : option_chains) {
            if (!chain.empty()) count++;
        }
        return count;
    }

    // Total option contracts across all symbols on this date
    size_t total_option_quotes() const {
        size_t total = 0;
        for (auto& [sym, chain] : option_chains) {
            total += chain.size();
        }
        return total;
    }
};

} // namespace trading

#endif // QUANTPRICER_TRADING_MARKET_SNAPSHOT_H

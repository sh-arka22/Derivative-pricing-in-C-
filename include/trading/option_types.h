#ifndef QUANTPRICER_TRADING_OPTION_TYPES_H
#define QUANTPRICER_TRADING_OPTION_TYPES_H

// ============================================================================
// Option Order & Fill Types + Helpers
//
// Options DON'T go through the OrderBook. The CSV bid/ask IS the market.
// Buy orders fill at the ask price. Sell orders fill at the bid price.
//
// This file defines:
//   OptionOrderRequest  — intent to trade an option contract
//   OptionFill          — confirmed execution
//   find_best_option()  — ATM selection closest to target DTE
//   fill_option_at_market() — simulate immediate fill at real bid/ask
// ============================================================================

#include "trading/market_data.h"        // OptionQuote, OptionChain
#include "trading/market_snapshot.h"    // MarketSnapshot

#include <string>
#include <vector>
#include <optional>
#include <cmath>
#include <limits>

namespace trading {

// ============================================================================
// OptionOrderRequest — Intent to trade an option contract
// ============================================================================

struct OptionOrderRequest {
    std::string symbol;         // underlying symbol, e.g. "AAPL"
    std::string expiration;     // "YYYY-MM-DD"
    double strike;
    bool is_call;               // true = call, false = put
    int contracts;              // number of contracts (each = 100 shares)
    bool is_buy;                // true = buy-to-open/close, false = sell
};

// ============================================================================
// OptionFill — Confirmed execution of an option order
// ============================================================================

struct OptionFill {
    std::string symbol;         // underlying
    std::string expiration;     // "YYYY-MM-DD"
    double strike;
    bool is_call;
    double price;               // per-share fill price
    int contracts;
    bool is_buy;
    std::string date;           // fill date
};

// ============================================================================
// find_best_option — Find ATM option closest to target DTE
// ============================================================================
// Searches the chain for options matching is_call, filters out quotes with
// bid <= 0, and selects the one closest to spot (ATM) among expirations
// closest to target_dte days out.
//
// Score = dte_distance * 1.0 + strike_distance_pct * 100.0
//   → Prioritizes DTE match first, then ATM-ness
//
// Returns empty optional if no valid option found.
// ============================================================================

inline std::optional<OptionQuote> find_best_option(
    const OptionChain& chain,
    double spot,
    int target_dte,
    bool is_call)
{
    std::optional<OptionQuote> best;
    double best_score = std::numeric_limits<double>::max();

    for (const auto& quote : chain.quotes) {
        // Filter: must match call/put and have a positive bid
        if (quote.is_call != is_call) continue;
        if (quote.bid <= 0.0) continue;

        // Compute DTE from the chain's observation date
        // (OptionQuote has no dte field — use days_to_expiry_from)
        int dte = quote.days_to_expiry_from(chain.date);
        if (dte <= 0) continue;  // expired or same-day — skip

        double dte_diff = std::abs(static_cast<double>(dte) - static_cast<double>(target_dte));
        double strike_diff = std::abs(quote.strike - spot) / spot;  // normalized by spot

        // Prioritize DTE match, then ATM-ness
        double score = dte_diff * 1.0 + strike_diff * 100.0;

        if (score < best_score) {
            best_score = score;
            best = quote;
        }
    }

    return best;
}

// ============================================================================
// fill_option_at_market — Simulate immediate fill at bid/ask
// ============================================================================
// Buy orders fill at the ask price, sell orders fill at the bid price.
// Returns empty optional if the option chain has no data for the requested
// strike/expiration or if the relevant price is zero.
// ============================================================================

inline std::optional<OptionFill> fill_option_at_market(
    const OptionOrderRequest& req,
    const MarketSnapshot& snapshot,
    const std::string& date)
{
    if (!snapshot.has_options(req.symbol)) return std::nullopt;

    const auto& chain = snapshot.option_chains.at(req.symbol);

    // Find matching quote in the chain
    for (const auto& quote : chain.quotes) {
        if (quote.is_call != req.is_call) continue;
        if (std::abs(quote.strike - req.strike) > 0.001) continue;
        if (quote.expiration != req.expiration) continue;

        // Determine fill price: buy at ask, sell at bid
        double fill_price = req.is_buy ? quote.ask : quote.bid;
        if (fill_price <= 0.0) return std::nullopt;

        OptionFill fill;
        fill.symbol     = req.symbol;
        fill.expiration = req.expiration;
        fill.strike     = req.strike;
        fill.is_call    = req.is_call;
        fill.price      = fill_price;
        fill.contracts  = req.contracts;
        fill.is_buy     = req.is_buy;
        fill.date       = date;
        return fill;
    }

    return std::nullopt;
}

} // namespace trading

#endif // QUANTPRICER_TRADING_OPTION_TYPES_H

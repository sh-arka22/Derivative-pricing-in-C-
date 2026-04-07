#ifndef QUANTPRICER_TRADING_SIMULATED_EXCHANGE_H
#define QUANTPRICER_TRADING_SIMULATED_EXCHANGE_H

// ============================================================================
// Simulated Exchange — Stock Order Execution
//
// Wraps orderbook::OrderBook to simulate a local exchange.
// Each bar:
//   1. seed_liquidity() — creates a FRESH OrderBook with N bid + N ask levels
//   2. submit_order()   — matches strategy orders against seeded liquidity
//   3. Fills are extracted from the book's trade history
//
// Options do NOT go through this exchange. Option orders fill directly at
// real bid/ask from chain data via fill_option_at_market() in option_types.h.
//
// IMPORTANT: A new OrderBook is created each bar because:
//   - OrderBook has no clear() or reset() method
//   - Stale liquidity from previous bars should not persist
//   - Real markets have completely different books each day
// ============================================================================

#include "trading/config.h"
#include "trading/market_data.h"
#include "orderbook/orderbook.h"

#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <stdexcept>

namespace trading {

// ============================================================================
// OrderRequest — What a strategy wants to do
// ============================================================================

struct OrderRequest {
    std::string symbol;
    orderbook::Side side;
    int quantity;               // always positive (side indicates direction)
    double limit_price;         // 0.0 = market order

    bool is_market() const { return limit_price <= 0.0; }
};

// ============================================================================
// Fill — What actually happened
// ============================================================================

struct Fill {
    std::string symbol;
    orderbook::Side side;
    double price;
    int quantity;
    std::string date;           // date of the bar
};

// ============================================================================
// SimulatedExchange
// ============================================================================

class SimulatedExchange {
public:
    explicit SimulatedExchange(const SimulationParams& params)
        : params_(params) {}

    // ================================================================
    // Seed fresh liquidity for a symbol around the bar price
    // ================================================================
    // Algorithm:
    //   half_spread = close * (spread_bps / 10000) / 2
    //   slippage_per_level = close * (slippage_bps / 10000)
    //   For i = 0..num_levels-1:
    //     offset = half_spread + i * slippage_per_level
    //     bid_price = close - offset
    //     ask_price = close + offset
    //     qty = base_qty * (num_levels - i) / num_levels  (thinning)
    //     place bid + ask at those prices
    // ================================================================

    void seed_liquidity(const std::string& symbol, const Bar& bar) {
        // Create a FRESH book (replaces any existing one for this symbol)
        books_.erase(symbol);
        books_.emplace(symbol, orderbook::OrderBook(symbol));

        auto& book = books_.at(symbol);
        double P = bar.close;

        double half_spread = P * (params_.spread_bps / 10000.0) / 2.0;
        double slippage_per_level = P * (params_.slippage_bps / 10000.0);

        for (int i = 0; i < params_.num_levels; i++) {
            double offset = half_spread + i * slippage_per_level;
            double bid_price = P - offset;
            double ask_price = P + offset;

            // Thinning: more liquidity at best level, less at deeper levels
            int qty = params_.base_qty_per_level * (params_.num_levels - i) / params_.num_levels;
            if (qty < 1) qty = 1;

            book.add_limit_order(orderbook::Side::Buy,  bid_price, static_cast<uint32_t>(qty));
            book.add_limit_order(orderbook::Side::Sell, ask_price, static_cast<uint32_t>(qty));
        }

        // Record the trade count AFTER seeding (seeding produces 0 trades
        // since bids < asks, but track it for safety)
        trade_offsets_[symbol] = book.trade_count();
    }

    // ================================================================
    // Submit an order and return fills
    // ================================================================
    // 1. Record current trade count
    // 2. Submit order to the book
    // 3. Extract new trades (since our submission) as Fill objects
    // ================================================================

    std::vector<Fill> submit_order(const OrderRequest& order, const std::string& date) {
        auto it = books_.find(order.symbol);
        if (it == books_.end()) {
            // No book seeded for this symbol — return empty fills
            return {};
        }

        auto& book = it->second;
        size_t trades_before = book.trade_count();

        // Submit to order book
        if (order.is_market()) {
            book.add_market_order(order.side, static_cast<uint32_t>(order.quantity));
        } else {
            book.add_limit_order(order.side, order.limit_price,
                                 static_cast<uint32_t>(order.quantity));
        }

        // Extract new fills
        std::vector<Fill> fills;
        const auto& all_trades = book.trades();
        for (size_t i = trades_before; i < all_trades.size(); i++) {
            Fill fill;
            fill.symbol = order.symbol;
            fill.side = order.side;
            fill.price = all_trades[i].price;
            fill.quantity = static_cast<int>(all_trades[i].quantity);
            fill.date = date;
            fills.push_back(fill);
        }

        return fills;
    }

    // ================================================================
    // Read-only access to a book (for strategy introspection)
    // ================================================================

    const orderbook::OrderBook& get_book(const std::string& symbol) const {
        auto it = books_.find(symbol);
        if (it == books_.end()) {
            throw std::runtime_error("No book for symbol: " + symbol);
        }
        return it->second;
    }

    bool has_book(const std::string& symbol) const {
        return books_.find(symbol) != books_.end();
    }

private:
    SimulationParams params_;
    std::map<std::string, orderbook::OrderBook> books_;
    std::map<std::string, size_t> trade_offsets_;   // tracks trade count after seeding
};

} // namespace trading

#endif // QUANTPRICER_TRADING_SIMULATED_EXCHANGE_H

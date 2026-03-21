#ifndef QUANTPRICER_ORDERBOOK_H
#define QUANTPRICER_ORDERBOOK_H

// ============================================================================
// Order Book & Matching Engine — Day 15
//
// ============================================================================
// MARKET MICROSTRUCTURE — How Exchanges Work
// ============================================================================
//
// An ORDER BOOK is the central data structure of an exchange.
// It maintains all resting (unexecuted) limit orders, organised by:
//   - Side:  BID (buy) or ASK (sell)
//   - Price: best (highest) bid, best (lowest) ask
//   - Time:  within a price level, FIFO ordering (first in, first out)
//
// The MATCHING ENGINE processes incoming orders against the book:
//   1. Limit Order: "Buy 100 shares at $50 or better"
//      → If there's a resting ask at $50 or below: immediate match (trade)
//      → Otherwise: rest in the book at $50
//
//   2. Market Order: "Buy 100 shares at any price"
//      → Match against the best available ask(s) immediately
//      → If book is empty on that side: reject or partial fill
//
// PRICE-TIME PRIORITY (most common matching algorithm):
//   - Price priority:  best price gets filled first
//     (highest bid, lowest ask)
//   - Time priority:   at the same price, earliest order fills first
//     (FIFO — first in, first out)
//
// ============================================================================
// DATA STRUCTURE DESIGN
// ============================================================================
//
// The classic design uses three layers:
//
//   Layer 1: ORDER — individual order with id, side, price, qty, timestamp
//   Layer 2: PRICE LEVEL — doubly-linked list of orders at the same price
//            (FIFO queue, O(1) front access, O(1) removal by iterator)
//   Layer 3: BOOK SIDE — sorted map of price → price_level
//            Bids: std::map<double, PriceLevel, std::greater<>> (highest first)
//            Asks: std::map<double, PriceLevel, std::less<>>    (lowest first)
//
// Complexity:
//   Add order:     O(log P) where P = number of distinct price levels
//   Cancel order:  O(1) if we have an iterator to the order (via hash map)
//   Best bid/ask:  O(1) — it's always map.begin()
//   Match:         O(1) per fill (walk the FIFO queue at best price)
//
// ============================================================================
// KEY INTERVIEW INSIGHTS
// ============================================================================
//
// 1. "Why std::map and not std::unordered_map for price levels?"
//    We need SORTED access (best price = begin()). unordered_map has O(1) lookup
//    but no ordering. std::map gives O(log P) insert AND sorted iteration.
//    In production: a skip list or a flat sorted array (for cache locality).
//
// 2. "Why std::list for orders within a price level?"
//    We need O(1) removal from the middle (cancel order) and O(1) front
//    access (match the oldest order). std::list provides both via iterators.
//    In production: an intrusive linked list avoids heap allocations.
//
// 3. "What's the hot path?"
//    The hot path is: receive message → parse → match → send fill.
//    Everything on this path must be O(1) or O(log P) with small P.
//    Heap allocations, cache misses, and system calls are the enemy.
//
// 4. "How do you handle order cancellation efficiently?"
//    Keep a hash map: order_id → {side, price, list_iterator}.
//    Cancel = O(1) hash lookup + O(1) list erase + O(1) check if level empty.
//
// 5. "What's the difference between price-time and pro-rata matching?"
//    Price-time (FIFO): queue position matters, rewards speed.
//    Pro-rata: fill proportional to order size. Used in some futures markets
//    (e.g. CME Eurodollar). Rewards SIZE over speed.
//
// 6. "How do real exchanges handle ticks?"
//    Prices are integers (price * 100 for cents, * 10000 for pips).
//    Floating-point comparison is dangerous. We use double here for clarity
//    but note that production systems use integer prices.
//
// 7. "What about hidden/iceberg orders?"
//    Hidden orders don't appear in the book but can still be matched.
//    Iceberg orders show a small "display quantity" but have a larger
//    hidden reserve. They lose time priority when the display refreshes.
// ============================================================================

#include <cstdint>
#include <map>
#include <list>
#include <unordered_map>
#include <vector>
#include <string>
#include <functional>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace orderbook {

// ============================================================================
// Core Types
// ============================================================================

enum class Side : uint8_t { Buy, Sell };
enum class OrderType : uint8_t { Limit, Market };

/// Unique order identifier (monotonically increasing)
using OrderId = uint64_t;

/// Timestamp in nanoseconds (for time priority)
using Timestamp = uint64_t;

/// Get current timestamp in nanoseconds
inline Timestamp now_ns() {
    return static_cast<Timestamp>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
}

// ============================================================================
// Order — A single order in the book
// ============================================================================
//
// Memory layout matters for cache performance:
//   - Keep frequently accessed fields (price, qty, side) together
//   - 64 bytes = one cache line on most architectures
//   - In production: alignas(64) and pack fields carefully
// ============================================================================

struct Order {
    OrderId    id;          // Unique identifier
    Side       side;        // Buy or Sell
    OrderType  type;        // Limit or Market
    double     price;       // Limit price (0 for market orders)
    uint32_t   quantity;    // Remaining quantity
    uint32_t   filled_qty;  // Cumulative filled quantity
    Timestamp  timestamp;   // Insertion time (for FIFO priority)

    Order() : id(0), side(Side::Buy), type(OrderType::Limit),
              price(0.0), quantity(0), filled_qty(0), timestamp(0) {}

    Order(OrderId id_, Side s, OrderType t, double p, uint32_t q)
        : id(id_), side(s), type(t), price(p), quantity(q),
          filled_qty(0), timestamp(now_ns()) {}

    bool is_filled() const { return quantity == 0; }
    uint32_t original_qty() const { return quantity + filled_qty; }
};

// ============================================================================
// Trade — Result of a match between two orders
// ============================================================================

struct Trade {
    OrderId    buy_order_id;
    OrderId    sell_order_id;
    double     price;       // Execution price (resting order's price)
    uint32_t   quantity;    // Fill quantity
    Timestamp  timestamp;
};

// ============================================================================
// Price Level — FIFO queue of orders at the same price
// ============================================================================
//
// Orders within a level are maintained in FIFO order (time priority).
// std::list provides:
//   - O(1) push_back (new order)
//   - O(1) pop_front (match oldest order)
//   - O(1) erase by iterator (cancel order)
// ============================================================================

class PriceLevel {
public:
    explicit PriceLevel(double price) : price_(price), total_qty_(0) {}

    /// Add an order to the back of the queue (newest = lowest priority)
    void add_order(const Order& order) {
        orders_.push_back(order);
        total_qty_ += order.quantity;
    }

    /// Get the oldest (highest priority) order
    Order& front() { return orders_.front(); }
    const Order& front() const { return orders_.front(); }

    /// Remove the front order (after full fill)
    void pop_front() {
        total_qty_ -= orders_.front().quantity;
        orders_.pop_front();
    }

    /// Reduce the front order's quantity (partial fill)
    void reduce_front(uint32_t qty) {
        orders_.front().quantity -= qty;
        orders_.front().filled_qty += qty;
        total_qty_ -= qty;
    }

    /// Remove an order by iterator (cancel)
    void remove(std::list<Order>::iterator it) {
        total_qty_ -= it->quantity;
        orders_.erase(it);
    }

    double price() const { return price_; }
    uint32_t total_quantity() const { return total_qty_; }
    size_t order_count() const { return orders_.size(); }
    bool empty() const { return orders_.empty(); }

    // Iterator access for walking the queue
    auto begin() { return orders_.begin(); }
    auto end() { return orders_.end(); }
    auto begin() const { return orders_.begin(); }
    auto end() const { return orders_.end(); }

private:
    double price_;
    uint32_t total_qty_;
    std::list<Order> orders_;  // FIFO queue
};

// ============================================================================
// Order Book — The Central Limit Order Book (CLOB)
// ============================================================================
//
// Two sorted sides:
//   Bids: std::map<double, PriceLevel, std::greater<double>>
//         → highest price first (best bid = begin())
//   Asks: std::map<double, PriceLevel, std::less<double>>
//         → lowest price first (best ask = begin())
//
// Order lookup: std::unordered_map<OrderId, location_info>
//   → O(1) cancel by order ID
//
// The matching engine is integrated: when a new order arrives,
// it first tries to match against the opposite side, then rests.
// ============================================================================

class OrderBook {
public:
    explicit OrderBook(const std::string& symbol = "AAPL")
        : symbol_(symbol), next_id_(1), trade_count_(0) {}

    // ================================================================
    // Order Submission
    // ================================================================

    /// Submit a limit order. Returns the order ID.
    /// Immediately matches against resting orders if possible.
    OrderId add_limit_order(Side side, double price, uint32_t qty) {
        OrderId id = next_id_++;
        Order order(id, side, OrderType::Limit, price, qty);

        // Try to match against opposite side first
        match(order);

        // If there's remaining quantity, rest in the book
        if (order.quantity > 0) {
            rest_order(order);
        }
        return id;
    }

    /// Submit a market order. Matches immediately, no resting.
    /// Returns the order ID. Unfilled quantity is lost (no resting).
    OrderId add_market_order(Side side, uint32_t qty) {
        OrderId id = next_id_++;
        Order order(id, side, OrderType::Market, 0.0, qty);
        match(order);
        return id;
    }

    /// Cancel a resting order by ID. Returns true if found and cancelled.
    bool cancel_order(OrderId id) {
        auto it = order_map_.find(id);
        if (it == order_map_.end()) return false;

        auto& loc = it->second;
        if (loc.side == Side::Buy) {
            auto bid_it = bids_.find(loc.price);
            if (bid_it != bids_.end()) {
                bid_it->second.remove(loc.list_it);
                if (bid_it->second.empty()) bids_.erase(bid_it);
            }
        } else {
            auto ask_it = asks_.find(loc.price);
            if (ask_it != asks_.end()) {
                ask_it->second.remove(loc.list_it);
                if (ask_it->second.empty()) asks_.erase(ask_it);
            }
        }
        order_map_.erase(it);
        return true;
    }

    // ================================================================
    // Book State Queries — O(1) best bid/ask
    // ================================================================

    bool has_bids() const { return !bids_.empty(); }
    bool has_asks() const { return !asks_.empty(); }

    /// Best bid price (highest resting buy). Throws if no bids.
    double best_bid() const {
        if (bids_.empty()) throw std::runtime_error("No bids");
        return bids_.begin()->first;
    }

    /// Best ask price (lowest resting sell). Throws if no asks.
    double best_ask() const {
        if (asks_.empty()) throw std::runtime_error("No asks");
        return asks_.begin()->first;
    }

    /// Bid-ask spread = best_ask - best_bid
    double spread() const {
        return best_ask() - best_bid();
    }

    /// Mid price = (best_bid + best_ask) / 2
    double mid_price() const {
        return (best_bid() + best_ask()) / 2.0;
    }

    /// Total quantity at best bid
    uint32_t bid_depth_at_best() const {
        if (bids_.empty()) return 0;
        return bids_.begin()->second.total_quantity();
    }

    /// Total quantity at best ask
    uint32_t ask_depth_at_best() const {
        if (asks_.empty()) return 0;
        return asks_.begin()->second.total_quantity();
    }

    /// Total bid depth across all levels
    uint32_t total_bid_depth() const {
        uint32_t d = 0;
        for (auto& [p, lvl] : bids_) d += lvl.total_quantity();
        return d;
    }

    /// Total ask depth across all levels
    uint32_t total_ask_depth() const {
        uint32_t d = 0;
        for (auto& [p, lvl] : asks_) d += lvl.total_quantity();
        return d;
    }

    /// Number of distinct bid price levels
    size_t bid_levels() const { return bids_.size(); }

    /// Number of distinct ask price levels
    size_t ask_levels() const { return asks_.size(); }

    /// Order book imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
    /// Range: [-1, +1]. Positive = more buy pressure.
    double imbalance() const {
        double bd = static_cast<double>(bid_depth_at_best());
        double ad = static_cast<double>(ask_depth_at_best());
        double total = bd + ad;
        return (total > 0) ? (bd - ad) / total : 0.0;
    }

    /// Volume-Weighted Average Price of the last N trades
    double vwap(size_t n = 0) const {
        if (trades_.empty()) return 0.0;
        size_t count = (n == 0 || n > trades_.size()) ? trades_.size() : n;
        double vol_sum = 0.0, pv_sum = 0.0;
        for (size_t i = trades_.size() - count; i < trades_.size(); ++i) {
            pv_sum += trades_[i].price * trades_[i].quantity;
            vol_sum += trades_[i].quantity;
        }
        return (vol_sum > 0) ? pv_sum / vol_sum : 0.0;
    }

    // ================================================================
    // Level 2 Data — Depth of Book
    // ================================================================

    struct L2Entry {
        double price;
        uint32_t quantity;
        size_t order_count;
    };

    /// Get top N bid levels
    std::vector<L2Entry> get_bids(size_t n = 5) const {
        std::vector<L2Entry> result;
        size_t count = 0;
        for (auto& [price, level] : bids_) {
            if (count++ >= n) break;
            result.push_back({price, level.total_quantity(), level.order_count()});
        }
        return result;
    }

    /// Get top N ask levels
    std::vector<L2Entry> get_asks(size_t n = 5) const {
        std::vector<L2Entry> result;
        size_t count = 0;
        for (auto& [price, level] : asks_) {
            if (count++ >= n) break;
            result.push_back({price, level.total_quantity(), level.order_count()});
        }
        return result;
    }

    // ================================================================
    // Trade History
    // ================================================================

    const std::vector<Trade>& trades() const { return trades_; }
    size_t trade_count() const { return trade_count_; }
    uint64_t total_volume() const {
        uint64_t vol = 0;
        for (auto& t : trades_) vol += t.quantity;
        return vol;
    }

    // ================================================================
    // Pretty Print
    // ================================================================

    std::string to_string(size_t depth = 5) const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        oss << "=== " << symbol_ << " Order Book ===\n";

        auto asks = get_asks(depth);
        // Print asks in reverse (highest first for visual clarity)
        for (auto it = asks.rbegin(); it != asks.rend(); ++it) {
            oss << "  ASK  " << std::setw(10) << it->price
                << "  x " << std::setw(6) << it->quantity
                << "  (" << it->order_count << " orders)\n";
        }

        if (has_bids() && has_asks()) {
            oss << "  ---- SPREAD: " << spread() << " ("
                << std::setprecision(1) << (spread() / mid_price() * 10000)
                << " bps) ----\n";
            oss << std::setprecision(2);
        }

        auto bids = get_bids(depth);
        for (auto& b : bids) {
            oss << "  BID  " << std::setw(10) << b.price
                << "  x " << std::setw(6) << b.quantity
                << "  (" << b.order_count << " orders)\n";
        }
        return oss.str();
    }

    const std::string& symbol() const { return symbol_; }

private:
    // ================================================================
    // Matching Engine — Core Logic
    // ================================================================
    //
    // PRICE-TIME PRIORITY:
    //   1. An incoming BUY matches against asks (lowest price first)
    //      if buy_price >= ask_price (willing to pay at least the ask)
    //   2. An incoming SELL matches against bids (highest price first)
    //      if sell_price <= bid_price (willing to accept at most the bid)
    //   3. Within a price level, the OLDEST order fills first (FIFO)
    //   4. The execution price is the RESTING order's price (maker's price)
    //      This is called "passive price improvement" for the aggressor.
    // ================================================================

    void match(Order& incoming) {
        if (incoming.side == Side::Buy) {
            match_buy(incoming);
        } else {
            match_sell(incoming);
        }
    }

    void match_buy(Order& buy_order) {
        // Walk the ask side: lowest ask first
        while (buy_order.quantity > 0 && !asks_.empty()) {
            auto& [ask_price, level] = *asks_.begin();

            // Price check: can we match?
            // Limit order: buy_price >= ask_price
            // Market order: always matches (price = 0 means "any price")
            if (buy_order.type == OrderType::Limit && buy_order.price < ask_price)
                break;  // No more matches possible (asks are sorted ascending)

            // Match against orders in this level (FIFO)
            while (buy_order.quantity > 0 && !level.empty()) {
                Order& resting = level.front();
                uint32_t fill_qty = std::min(buy_order.quantity, resting.quantity);

                // Record the trade at the RESTING order's price (maker price)
                trades_.push_back({buy_order.id, resting.id, ask_price,
                                   fill_qty, now_ns()});
                ++trade_count_;

                buy_order.quantity -= fill_qty;
                buy_order.filled_qty += fill_qty;

                if (fill_qty == resting.quantity) {
                    // Full fill of resting order — remove from book
                    order_map_.erase(resting.id);
                    level.pop_front();
                } else {
                    // Partial fill of resting order
                    level.reduce_front(fill_qty);
                }
            }

            // If the level is now empty, remove it from the map
            if (level.empty()) {
                asks_.erase(asks_.begin());
            }
        }
    }

    void match_sell(Order& sell_order) {
        // Walk the bid side: highest bid first
        while (sell_order.quantity > 0 && !bids_.empty()) {
            auto& [bid_price, level] = *bids_.begin();

            // Price check
            if (sell_order.type == OrderType::Limit && sell_order.price > bid_price)
                break;

            while (sell_order.quantity > 0 && !level.empty()) {
                Order& resting = level.front();
                uint32_t fill_qty = std::min(sell_order.quantity, resting.quantity);

                trades_.push_back({resting.id, sell_order.id, bid_price,
                                   fill_qty, now_ns()});
                ++trade_count_;

                sell_order.quantity -= fill_qty;
                sell_order.filled_qty += fill_qty;

                if (fill_qty == resting.quantity) {
                    order_map_.erase(resting.id);
                    level.pop_front();
                } else {
                    level.reduce_front(fill_qty);
                }
            }

            if (level.empty()) {
                bids_.erase(bids_.begin());
            }
        }
    }

    /// Place a non-filled (or partially filled) order into the resting book
    void rest_order(Order& order) {
        if (order.side == Side::Buy) {
            auto [it, _] = bids_.try_emplace(order.price, PriceLevel(order.price));
            it->second.add_order(order);
            // Store location for O(1) cancel
            auto list_it = std::prev(it->second.end());
            order_map_[order.id] = {Side::Buy, order.price, list_it};
        } else {
            auto [it, _] = asks_.try_emplace(order.price, PriceLevel(order.price));
            it->second.add_order(order);
            auto list_it = std::prev(it->second.end());
            order_map_[order.id] = {Side::Sell, order.price, list_it};
        }
    }

    // ================================================================
    // Data Members
    // ================================================================

    std::string symbol_;
    OrderId next_id_;
    size_t trade_count_;

    // Bids: highest price first (std::greater)
    std::map<double, PriceLevel, std::greater<double>> bids_;
    // Asks: lowest price first (std::less — default)
    std::map<double, PriceLevel> asks_;

    // Order location map for O(1) cancel
    struct OrderLocation {
        Side side;
        double price;
        std::list<Order>::iterator list_it;
    };
    std::unordered_map<OrderId, OrderLocation> order_map_;

    // Trade history
    std::vector<Trade> trades_;
};

} // namespace orderbook

#endif // QUANTPRICER_ORDERBOOK_H

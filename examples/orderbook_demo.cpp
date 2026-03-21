// ============================================================================
// Day 15 Example: Order Book & Matching Engine
//
// Demonstrates:
// 1. Building an order book from limit orders
// 2. Price-time priority matching
// 3. Market order execution (walking the book)
// 4. Order cancellation
// 5. L2 book depth and analytics (spread, imbalance, VWAP)
// 6. Trade flow simulation
// ============================================================================

#include "orderbook/orderbook.h"
#include <iostream>
#include <iomanip>

using namespace orderbook;

int main() {
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "======================================================================\n"
              << "  Day 15: Order Book & Matching Engine\n"
              << "======================================================================\n\n";

    OrderBook book("AAPL");

    // ================================================================
    // 1. Build the Book — Limit Orders
    // ================================================================
    std::cout << "=== 1. Building the Order Book ===\n\n";

    // Place resting bid orders (buy side)
    book.add_limit_order(Side::Buy, 149.50, 500);
    book.add_limit_order(Side::Buy, 149.50, 300);  // Same price, later time
    book.add_limit_order(Side::Buy, 149.00, 1000);
    book.add_limit_order(Side::Buy, 148.50, 200);
    book.add_limit_order(Side::Buy, 148.00, 800);

    // Place resting ask orders (sell side)
    book.add_limit_order(Side::Sell, 150.00, 400);
    book.add_limit_order(Side::Sell, 150.00, 600);
    book.add_limit_order(Side::Sell, 150.50, 300);
    book.add_limit_order(Side::Sell, 151.00, 500);
    book.add_limit_order(Side::Sell, 151.50, 1000);

    std::cout << book.to_string(5) << "\n";

    std::cout << "Analytics:\n"
              << "  Best Bid:      " << book.best_bid() << "\n"
              << "  Best Ask:      " << book.best_ask() << "\n"
              << "  Spread:        " << book.spread() << " ("
              << std::setprecision(1) << (book.spread() / book.mid_price() * 10000)
              << " bps)\n" << std::setprecision(2)
              << "  Mid Price:     " << book.mid_price() << "\n"
              << "  Bid Depth(L1): " << book.bid_depth_at_best() << "\n"
              << "  Ask Depth(L1): " << book.ask_depth_at_best() << "\n"
              << "  Imbalance:     " << std::setprecision(3) << book.imbalance()
              << " (positive = buy pressure)\n\n";
    std::cout << std::setprecision(2);

    // ================================================================
    // 2. Aggressive Limit Order — Crosses the Spread
    // ================================================================
    std::cout << "=== 2. Aggressive Buy Limit @ 150.00 x 500 ===\n\n";
    std::cout << "This order crosses the spread (bid 149.50 -> buys at 150.00)\n";
    std::cout << "Matches against the two resting asks at 150.00:\n"
              << "  Fill 1: 400 @ 150.00 (from resting ask #6, FIFO first)\n"
              << "  Fill 2: 100 @ 150.00 (partial fill of resting ask #7)\n\n";

    book.add_limit_order(Side::Buy, 150.00, 500);

    std::cout << book.to_string(5) << "\n";
    std::cout << "Trades so far: " << book.trade_count() << "\n";
    std::cout << "Total volume:  " << book.total_volume() << "\n\n";

    // ================================================================
    // 3. Market Order — Walks the Book
    // ================================================================
    std::cout << "=== 3. Market Sell x 1200 (walks the book) ===\n\n";
    std::cout << "A market sell matches against bids from best to worst:\n"
              << "  Fills at 149.50 (800 qty), then 149.00 (400 of 1000)\n\n";

    book.add_market_order(Side::Sell, 1200);

    std::cout << book.to_string(5) << "\n";
    std::cout << "Trades so far: " << book.trade_count() << "\n";
    std::cout << "VWAP (all):    " << book.vwap() << "\n\n";

    // ================================================================
    // 4. Order Cancellation
    // ================================================================
    std::cout << "=== 4. Order Cancellation ===\n\n";

    // Add and then cancel
    auto id1 = book.add_limit_order(Side::Sell, 152.00, 999);
    std::cout << "Added ask @ 152.00 x 999 (id=" << id1 << ")\n";
    std::cout << "Ask levels before cancel: " << book.ask_levels() << "\n";

    bool cancelled = book.cancel_order(id1);
    std::cout << "Cancelled id=" << id1 << ": " << (cancelled ? "YES" : "NO") << "\n";
    std::cout << "Ask levels after cancel:  " << book.ask_levels() << "\n";

    // Try to cancel a non-existent order
    bool cancelled2 = book.cancel_order(99999);
    std::cout << "Cancel non-existent id=99999: " << (cancelled2 ? "YES" : "NO") << "\n\n";

    // ================================================================
    // 5. Trade History
    // ================================================================
    std::cout << "=== 5. Trade History ===\n\n";

    std::cout << std::setw(8) << "BuyID"
              << std::setw(8) << "SellID"
              << std::setw(10) << "Price"
              << std::setw(10) << "Qty" << "\n";
    std::cout << std::string(36, '-') << "\n";

    for (auto& t : book.trades()) {
        std::cout << std::setw(8) << t.buy_order_id
                  << std::setw(8) << t.sell_order_id
                  << std::setw(10) << t.price
                  << std::setw(10) << t.quantity << "\n";
    }
    std::cout << "\nVWAP: " << book.vwap() << "\n";
    std::cout << "Total trades: " << book.trade_count() << "\n";
    std::cout << "Total volume: " << book.total_volume() << "\n\n";

    // ================================================================
    // 6. Order Flow Simulation — Market Making Scenario
    // ================================================================
    std::cout << "=== 6. Market Making Simulation ===\n\n";

    OrderBook mm_book("SPY");

    // Market maker posts two-sided quotes
    std::cout << "Market maker posts bid/ask around mid=400.00\n\n";

    for (int i = 0; i < 10; ++i) {
        double mid = 400.00;
        double half_spread = 0.02 + 0.01 * (i % 3);
        uint32_t size = 100 + (i * 50);

        mm_book.add_limit_order(Side::Buy,  mid - half_spread, size);
        mm_book.add_limit_order(Side::Sell, mid + half_spread, size);
    }

    std::cout << mm_book.to_string(5) << "\n";

    // Simulate aggressor flow
    std::cout << "Aggressor buys 300 (market order)...\n";
    mm_book.add_market_order(Side::Buy, 300);
    std::cout << "After market buy:\n";
    std::cout << "  Best Ask: " << mm_book.best_ask()
              << "  (moved up from aggressor eating liquidity)\n";
    std::cout << "  Trades:   " << mm_book.trade_count() << "\n";
    std::cout << "  VWAP:     " << mm_book.vwap() << "\n\n";

    std::cout << mm_book.to_string(5) << "\n";

    // ================================================================
    // 7. Complexity Summary
    // ================================================================
    std::cout << "=== 7. Complexity Summary ===\n\n"
              << "  Operation          | Time Complexity\n"
              << "  -------------------|----------------\n"
              << "  Add limit order    | O(log P)  — map insertion\n"
              << "  Cancel order       | O(1)      — hash map + list erase\n"
              << "  Best bid/ask       | O(1)      — map.begin()\n"
              << "  Match (per fill)   | O(1)      — list front access\n"
              << "  Market order       | O(K log P) — K fills across levels\n"
              << "  Spread/mid         | O(1)\n"
              << "  L2 depth (N lvls)  | O(N)\n\n"
              << "  P = number of distinct price levels\n"
              << "  K = number of fills (partial/full)\n\n"
              << "  Production optimisations (not implemented here):\n"
              << "    - Integer prices (avoid floating-point comparison)\n"
              << "    - Intrusive linked lists (avoid heap allocation)\n"
              << "    - Lock-free SPSC queues (for multi-threaded gateway)\n"
              << "    - Memory pools / arena allocators\n"
              << "    - Cache-line aligned Order structs (alignas(64))\n";

    return 0;
}

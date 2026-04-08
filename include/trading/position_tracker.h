#ifndef QUANTPRICER_TRADING_POSITION_TRACKER_H
#define QUANTPRICER_TRADING_POSITION_TRACKER_H

// ============================================================================
// Position Tracker — Day 4
//
// The accounting engine: tracks all stock + option positions, cash, P&L,
// and equity history. Connects exchange fills to risk/strategy layers.
//
// Architecture:
//   Exchange fills → on_fill() / on_option_fill() → PositionTracker
//   MarketData     → on_price() / update_option_marks() → mark-to-market
//   End of day     → on_day_end() → equity_history_ → daily_return_series()
//                                                     → risk::compute_metrics()
//
// P&L Algorithm (4 cases):
//   Case 1: Flat → open (avg_cost = fill price)
//   Case 2: Same direction → weighted average cost
//   Case 3: Reduce/close → realize P&L on closed portion
//   Case 4: Reverse through zero → close old + open remainder (two-step)
//
// Cash: BUY → cash -= price * qty, SELL → cash += price * qty
// Equity = cash + Σ(stock.qty * mkt_price) + Σ(opt.contracts * 100 * mark)
// ============================================================================

#include "trading/simulated_exchange.h"  // Fill, OrderRequest
#include "trading/option_types.h"        // OptionFill
#include "trading/market_data.h"         // OptionChain, OptionQuote

#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace trading {

// ============================================================================
// Position — Holdings for one stock symbol
// ============================================================================

struct Position {
    std::string symbol;
    int quantity = 0;            // signed: +long, -short
    double avg_cost = 0.0;      // weighted average entry price
    double realized_pnl = 0.0;  // cumulative closed P&L
    double market_price = 0.0;  // latest mark-to-market price

    double unrealized_pnl() const {
        return quantity * (market_price - avg_cost);
    }

    double notional() const {
        return std::abs(quantity) * market_price;
    }

    bool is_flat() const { return quantity == 0; }
    bool is_long() const { return quantity > 0; }
    bool is_short() const { return quantity < 0; }
};

// ============================================================================
// OptionPosition — Holdings for one option contract series
// ============================================================================
// Key format: "AAPL|2025-06-20|200.0|C"
// Each contract = 100 shares (the standard equity options multiplier).
// Greeks are refreshed from chain data via update_option_marks().
// ============================================================================

struct OptionPosition {
    std::string symbol;         // underlying
    std::string expiration;     // "YYYY-MM-DD"
    double strike;
    bool is_call;
    int contracts = 0;          // signed: +long, -short
    double avg_cost = 0.0;      // per-share entry price (not per contract)
    double mark = 0.0;          // current mark from chain
    double realized_pnl = 0.0;

    // Greeks from latest chain data
    double delta = 0.0, gamma = 0.0, theta = 0.0, vega = 0.0;

    double unrealized_pnl() const {
        return contracts * 100.0 * (mark - avg_cost);
    }

    double notional() const {
        return std::abs(contracts) * 100.0 * mark;
    }

    double notional_delta() const {
        return contracts * 100.0 * delta;
    }

    bool is_flat() const { return contracts == 0; }

    // Composite key for map lookup
    std::string key() const {
        std::ostringstream oss;
        oss << symbol << "|" << expiration << "|" << strike << "|"
            << (is_call ? "C" : "P");
        return oss.str();
    }

    // Build key from components (static helper)
    static std::string make_key(const std::string& sym, const std::string& exp,
                                double strike, bool is_call) {
        std::ostringstream oss;
        oss << sym << "|" << exp << "|" << strike << "|"
            << (is_call ? "C" : "P");
        return oss.str();
    }
};

// ============================================================================
// GreeksSummary — Aggregate portfolio Greeks
// ============================================================================

struct GreeksSummary {
    double delta = 0.0;
    double gamma = 0.0;
    double theta = 0.0;
    double vega  = 0.0;
};

// ============================================================================
// PositionTracker
// ============================================================================
// Central portfolio state: stock + option positions, cash, equity curve.
//
// Usage flow in main loop:
//   1. exchange.submit_order() → fills → tracker.on_fill(fill)
//   2. fill_option_at_market() → opt_fill → tracker.on_option_fill(opt_fill)
//   3. tracker.on_price(sym, close) for each bar
//   4. tracker.update_option_marks(chain) for each chain
//   5. tracker.on_day_end(date) at end of each bar
//   6. At session end: risk::compute_metrics(tracker.daily_return_series())
// ============================================================================

class PositionTracker {
public:
    explicit PositionTracker(double initial_capital)
        : initial_capital_(initial_capital), cash_(initial_capital) {}

    // ================================================================
    // Stock fills — core P&L logic (4 cases)
    // ================================================================

    void on_fill(const Fill& fill) {
        auto& pos = positions_[fill.symbol];
        if (pos.symbol.empty()) {
            pos.symbol = fill.symbol;
        }

        int fill_signed = (fill.side == orderbook::Side::Buy)
                              ? +fill.quantity : -fill.quantity;

        if (pos.quantity == 0) {
            // Case 1: Opening a new position from flat
            pos.quantity = fill_signed;
            pos.avg_cost = fill.price;

        } else if ((pos.quantity > 0 && fill_signed > 0) ||
                   (pos.quantity < 0 && fill_signed < 0)) {
            // Case 2: Adding to existing position (same direction)
            // Weighted average: (old_cost * |old_qty| + new_cost * fill_qty) / |new_qty|
            double old_cost = pos.avg_cost * std::abs(pos.quantity);
            double new_cost = fill.price * fill.quantity;
            pos.quantity += fill_signed;
            pos.avg_cost = (old_cost + new_cost) / std::abs(pos.quantity);

        } else {
            // Case 3/4: Reducing or reversing position
            int close_qty = std::min(fill.quantity, std::abs(pos.quantity));

            // Realize P&L on the closed portion
            if (pos.quantity > 0) {
                // Was long, now selling — profit if sell > avg_cost
                pos.realized_pnl += close_qty * (fill.price - pos.avg_cost);
            } else {
                // Was short, now buying — profit if buy < avg_cost
                pos.realized_pnl += close_qty * (pos.avg_cost - fill.price);
            }

            pos.quantity += fill_signed;

            if (pos.quantity == 0) {
                // Fully closed
                pos.avg_cost = 0.0;
            } else if (std::abs(fill_signed) > close_qty) {
                // Case 4: Reversed through zero — remainder at new price
                pos.avg_cost = fill.price;
            }
            // else: partially reduced, avg_cost stays the same
        }

        // Cash update: BUY costs money, SELL receives money
        if (fill.side == orderbook::Side::Buy) {
            cash_ -= fill.price * fill.quantity;
        } else {
            cash_ += fill.price * fill.quantity;
        }

        trade_count_++;
    }

    // ================================================================
    // Option fills — same 4-case algorithm with 100x multiplier
    // ================================================================

    void on_option_fill(const OptionFill& fill) {
        std::string k = OptionPosition::make_key(
            fill.symbol, fill.expiration, fill.strike, fill.is_call);

        auto& pos = option_positions_[k];
        if (pos.symbol.empty()) {
            pos.symbol     = fill.symbol;
            pos.expiration = fill.expiration;
            pos.strike     = fill.strike;
            pos.is_call    = fill.is_call;
        }

        int fill_signed = fill.is_buy ? +fill.contracts : -fill.contracts;

        if (pos.contracts == 0) {
            // Case 1: Opening from flat
            pos.contracts = fill_signed;
            pos.avg_cost  = fill.price;

        } else if ((pos.contracts > 0 && fill_signed > 0) ||
                   (pos.contracts < 0 && fill_signed < 0)) {
            // Case 2: Adding same direction
            double old_cost = pos.avg_cost * std::abs(pos.contracts);
            double new_cost = fill.price * fill.contracts;
            pos.contracts += fill_signed;
            pos.avg_cost = (old_cost + new_cost) / std::abs(pos.contracts);

        } else {
            // Case 3/4: Reducing or reversing
            int close_qty = std::min(fill.contracts, std::abs(pos.contracts));

            if (pos.contracts > 0) {
                // Was long, selling to close
                pos.realized_pnl += close_qty * 100.0 * (fill.price - pos.avg_cost);
            } else {
                // Was short, buying to close
                pos.realized_pnl += close_qty * 100.0 * (pos.avg_cost - fill.price);
            }

            pos.contracts += fill_signed;

            if (pos.contracts == 0) {
                pos.avg_cost = 0.0;
            } else if (std::abs(fill_signed) > close_qty) {
                pos.avg_cost = fill.price;
            }
        }

        // Cash: per-share price * contracts * 100 shares/contract
        if (fill.is_buy) {
            cash_ -= fill.price * fill.contracts * 100.0;
        } else {
            cash_ += fill.price * fill.contracts * 100.0;
        }

        trade_count_++;
    }

    // ================================================================
    // Mark-to-market updates
    // ================================================================

    void on_price(const std::string& symbol, double price) {
        positions_[symbol].market_price = price;
        if (positions_[symbol].symbol.empty()) {
            positions_[symbol].symbol = symbol;
        }
    }

    // Update option marks + Greeks from the latest chain data.
    // Matches each held OptionPosition against quotes in the chain.
    void update_option_marks(const OptionChain& chain) {
        for (auto& [key, pos] : option_positions_) {
            if (pos.symbol != chain.symbol) continue;
            if (pos.contracts == 0) continue;

            for (const auto& q : chain.quotes) {
                if (q.is_call != pos.is_call) continue;
                if (std::abs(q.strike - pos.strike) > 0.001) continue;
                if (q.expiration != pos.expiration) continue;

                // Found matching quote — update mark + Greeks
                pos.mark  = q.mid();
                pos.delta = q.delta;
                pos.gamma = q.gamma;
                pos.theta = q.theta;
                pos.vega  = q.vega;
                break;
            }
        }
    }

    // ================================================================
    // End of day — snapshot equity for return series
    // ================================================================

    void on_day_end(const std::string& date) {
        equity_history_.push_back(total_equity());
        last_date_ = date;
    }

    // ================================================================
    // Stock position getters
    // ================================================================

    const Position& get_position(const std::string& symbol) const {
        static const Position empty_pos{};
        auto it = positions_.find(symbol);
        return (it != positions_.end()) ? it->second : empty_pos;
    }

    bool has_position(const std::string& symbol) const {
        auto it = positions_.find(symbol);
        return it != positions_.end() && it->second.quantity != 0;
    }

    std::map<std::string, Position>& positions() { return positions_; }
    const std::map<std::string, Position>& positions() const { return positions_; }

    // ================================================================
    // Option position getters
    // ================================================================

    const OptionPosition& get_option_position(const std::string& key) const {
        static const OptionPosition empty_pos{};
        auto it = option_positions_.find(key);
        return (it != option_positions_.end()) ? it->second : empty_pos;
    }

    bool has_option_position(const std::string& key) const {
        auto it = option_positions_.find(key);
        return it != option_positions_.end() && it->second.contracts != 0;
    }

    std::map<std::string, OptionPosition>& option_positions() { return option_positions_; }
    const std::map<std::string, OptionPosition>& option_positions() const { return option_positions_; }

    // ================================================================
    // Net delta for a symbol: stock qty + option notional deltas
    // ================================================================
    // This is what delta-hedging strategies query to know their net
    // exposure. A perfectly hedged book has net_delta ≈ 0.
    // ================================================================

    double net_delta(const std::string& symbol) const {
        double delta = 0.0;

        // Stock contribution
        auto sit = positions_.find(symbol);
        if (sit != positions_.end()) {
            delta += sit->second.quantity;
        }

        // Option contribution
        for (const auto& [key, pos] : option_positions_) {
            if (pos.symbol == symbol && pos.contracts != 0) {
                delta += pos.notional_delta();
            }
        }

        return delta;
    }

    // ================================================================
    // Portfolio-level Greeks (aggregate across all option positions)
    // ================================================================

    GreeksSummary portfolio_greeks() const {
        GreeksSummary g;
        for (const auto& [key, pos] : option_positions_) {
            if (pos.contracts == 0) continue;
            double multiplied = pos.contracts * 100.0;
            g.delta += multiplied * pos.delta;
            g.gamma += multiplied * pos.gamma;
            g.theta += multiplied * pos.theta;
            g.vega  += multiplied * pos.vega;
        }
        return g;
    }

    // ================================================================
    // Portfolio-level P&L and equity
    // ================================================================

    double cash() const { return cash_; }
    double initial_capital() const { return initial_capital_; }

    double total_equity() const {
        double equity = cash_;
        // Stock positions
        for (const auto& [sym, pos] : positions_) {
            equity += pos.quantity * pos.market_price;
        }
        // Option positions
        for (const auto& [key, pos] : option_positions_) {
            equity += pos.contracts * 100.0 * pos.mark;
        }
        return equity;
    }

    double total_realized_pnl() const {
        double total = 0;
        for (const auto& [sym, pos] : positions_) {
            total += pos.realized_pnl;
        }
        for (const auto& [key, pos] : option_positions_) {
            total += pos.realized_pnl;
        }
        return total;
    }

    double total_unrealized_pnl() const {
        double total = 0;
        for (const auto& [sym, pos] : positions_) {
            total += pos.unrealized_pnl();
        }
        for (const auto& [key, pos] : option_positions_) {
            total += pos.unrealized_pnl();
        }
        return total;
    }

    double total_pnl() const {
        return total_realized_pnl() + total_unrealized_pnl();
    }

    double portfolio_notional() const {
        double total = 0;
        for (const auto& [sym, pos] : positions_) {
            total += pos.notional();
        }
        for (const auto& [key, pos] : option_positions_) {
            total += pos.notional();
        }
        return total;
    }

    int total_trades() const { return trade_count_; }

    // ================================================================
    // Return series for risk::compute_metrics()
    // ================================================================
    // Daily percentage returns from equity snapshots.
    // Requires >= 2 on_day_end() calls to produce any returns.
    // ================================================================

    std::vector<double> daily_return_series() const {
        std::vector<double> returns;
        for (size_t i = 1; i < equity_history_.size(); i++) {
            if (std::abs(equity_history_[i - 1]) > 1e-10) {
                returns.push_back(
                    (equity_history_[i] - equity_history_[i - 1]) / equity_history_[i - 1]);
            } else {
                returns.push_back(0.0);
            }
        }
        return returns;
    }

    const std::vector<double>& equity_history() const { return equity_history_; }

    // ================================================================
    // Debug print — shows all non-flat positions
    // ================================================================

    void print_positions() const {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "--- Stock Positions ---\n";
        for (const auto& [sym, pos] : positions_) {
            if (pos.quantity == 0 && pos.realized_pnl == 0) continue;
            std::cout << "  " << sym
                      << "  qty=" << pos.quantity
                      << "  avg=" << pos.avg_cost
                      << "  mkt=" << pos.market_price
                      << "  unreal=" << pos.unrealized_pnl()
                      << "  real=" << pos.realized_pnl << "\n";
        }

        // Option positions
        bool any_opts = false;
        for (const auto& [key, pos] : option_positions_) {
            if (pos.contracts == 0 && pos.realized_pnl == 0) continue;
            if (!any_opts) {
                std::cout << "--- Option Positions ---\n";
                any_opts = true;
            }
            std::cout << "  " << pos.symbol
                      << " " << pos.strike << (pos.is_call ? "C" : "P")
                      << " " << pos.expiration
                      << "  cts=" << pos.contracts
                      << "  avg=" << pos.avg_cost
                      << "  mark=" << pos.mark
                      << "  unreal=" << pos.unrealized_pnl()
                      << "  real=" << pos.realized_pnl
                      << "  d=" << pos.delta
                      << "\n";
        }

        std::cout << "--- Summary ---\n";
        std::cout << "  Cash: $" << cash_
                  << "  Equity: $" << total_equity()
                  << "  P&L: $" << total_pnl()
                  << "  Trades: " << trade_count_ << "\n";

        // Net delta per underlying (if any options held)
        if (any_opts) {
            std::cout << "  Net Deltas:";
            std::set<std::string> underlyings;
            for (const auto& [key, pos] : option_positions_) {
                if (pos.contracts != 0) underlyings.insert(pos.symbol);
            }
            for (const auto& sym : underlyings) {
                std::cout << "  " << sym << "=" << net_delta(sym);
            }
            std::cout << "\n";
        }
    }

private:
    double initial_capital_;
    double cash_;
    int trade_count_ = 0;
    std::map<std::string, Position> positions_;
    std::map<std::string, OptionPosition> option_positions_;
    std::vector<double> equity_history_;
    std::string last_date_;
};

} // namespace trading

#endif // QUANTPRICER_TRADING_POSITION_TRACKER_H

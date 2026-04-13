#ifndef QUANTPRICER_TRADING_LOGGING_H
#define QUANTPRICER_TRADING_LOGGING_H

// ============================================================================
// Trade Logger — Terminal + CSV output
//
// Two output channels:
//   1. Terminal (stdout) — formatted, verbose-gated
//   2. CSV file (logs/trades_YYYYMMDD_HHMMSS.csv) — every fill
//
// Terminal format:
//   [2025-01-02] BAR   AAPL  O=185.20 H=186.10 L=184.50 C=185.80
//   [2025-01-02] SIG   AAPL  BUY 100 @ MKT
//   [2025-01-02] FILL  AAPL  BUY 100 @ 185.85  pos=+100 avg=185.85
//   [2025-01-02] RISK  VaR=$2341.00  notional=$18585.00  P&L=$0.00
//   [2025-01-02] REJ   AAPL  SELL 600 -- order size exceeds limit
//
// CSV format:
//   Date,Symbol,Side,Quantity,Price,Position,AvgCost,RealizedPnL,Cash,Equity
// ============================================================================

#include "trading/config.h"
#include "trading/market_data.h"
#include "trading/simulated_exchange.h"
#include "trading/position_tracker.h"
#include "trading/risk_guard.h"
#include "risk/risk.h"

#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <sys/stat.h>

namespace trading {

class TradeLogger {
public:
    TradeLogger(const std::string& log_dir, bool verbose)
        : log_dir_(log_dir), verbose_(verbose)
    {
        ensure_log_dir();
        open_csv();
    }

    ~TradeLogger() {
        if (csv_file_.is_open()) {
            csv_file_.close();
        }
    }

    // ================================================================
    // Log header at start of simulation
    // ================================================================

    void log_header(const Config& config) {
        std::cout << "\n";
        config.print_summary();
        std::cout << "\n";
    }

    // ================================================================
    // Log bars (terminal only, if verbose)
    // ================================================================

    void log_bar(const std::map<std::string, Bar>& bars,
                 const PositionTracker& tracker,
                 const RiskGuard& risk_guard)
    {
        if (!verbose_) return;

        for (auto& [sym, bar] : bars) {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "[" << bar.date << "] BAR   " << std::setw(5) << sym
                      << "  O=" << bar.open
                      << " H=" << bar.high
                      << " L=" << bar.low
                      << " C=" << bar.close
                      << "  vol=" << format_volume(bar.volume) << "\n";
        }

        // Risk line (once per bar set)
        auto status = risk_guard.monitor(tracker);
        if (!bars.empty()) {
            std::string date = bars.begin()->second.date;
            std::cout << "[" << date << "] RISK  "
                      << "VaR=$" << format_number(status.portfolio_var)
                      << "  notional=$" << format_number(status.portfolio_notional)
                      << "  P&L=$" << format_number(tracker.total_pnl()) << "\n";
        }
    }

    // ================================================================
    // Log a signal (order request from strategy, before risk check)
    // ================================================================

    void log_signal(const OrderRequest& order, const std::string& date) {
        if (!verbose_) return;
        std::cout << "[" << date << "] SIG   " << std::setw(5) << order.symbol << "  "
                  << side_str(order.side) << " " << order.quantity;
        if (order.is_market()) {
            std::cout << " @ MKT";
        } else {
            std::cout << " @ LMT " << std::fixed << std::setprecision(2) << order.limit_price;
        }
        std::cout << "\n";
    }

    // ================================================================
    // Log a fill (terminal + CSV)
    // ================================================================

    void log_fill(const Fill& fill, const PositionTracker& tracker) {
        fill_count_++;
        const auto& pos = tracker.get_position(fill.symbol);

        // Terminal
        if (verbose_) {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "[" << fill.date << "] FILL  " << std::setw(5) << fill.symbol << "  "
                      << side_str(fill.side) << " " << fill.quantity
                      << " @ " << fill.price
                      << "  pos=" << (pos.quantity >= 0 ? "+" : "") << pos.quantity
                      << " avg=" << pos.avg_cost << "\n";
        }

        // CSV
        if (csv_file_.is_open()) {
            csv_file_ << std::fixed << std::setprecision(2);
            csv_file_ << fill.date << ","
                      << fill.symbol << ","
                      << side_str(fill.side) << ","
                      << fill.quantity << ","
                      << fill.price << ","
                      << pos.quantity << ","
                      << pos.avg_cost << ","
                      << pos.realized_pnl << ","
                      << tracker.cash() << ","
                      << tracker.total_equity() << "\n";
            csv_file_.flush();
        }
    }

    // ================================================================
    // Log risk rejection
    // ================================================================

    void log_risk_rejection(const OrderRequest& order, const std::string& reason,
                            const std::string& date) {
        rejection_count_++;
        if (verbose_) {
            std::cout << "[" << date << "] REJ   " << std::setw(5) << order.symbol << "  "
                      << side_str(order.side) << " " << order.quantity
                      << " -- " << reason << "\n";
        }
    }

    // ================================================================
    // Log kill switch event
    // ================================================================

    void log_kill_switch(const RiskStatus& status, const std::string& date) {
        std::cout << "\n[" << date << "] *** " << status.message << " ***\n";
        std::cout << "Simulation terminated by risk guard.\n\n";
    }

    // ================================================================
    // Session summary (always printed, not gated by verbose)
    // ================================================================

    void log_session_summary(const PositionTracker& tracker,
                             const risk::PerformanceMetrics& metrics,
                             const std::string& strategy_name,
                             size_t bars_processed)
    {
        double total_pnl = tracker.total_pnl();
        double pnl_pct = (tracker.initial_capital() > 0)
            ? (total_pnl / tracker.initial_capital()) * 100.0 : 0.0;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n";
        std::cout << std::string(47, '=') << "\n";
        std::cout << "           SESSION SUMMARY\n";
        std::cout << std::string(47, '=') << "\n";
        std::cout << "  Strategy:        " << strategy_name << "\n";
        std::cout << "  Bars Processed:  " << bars_processed << "\n";
        std::cout << "  Trades:          " << tracker.total_trades() << "\n";
        std::cout << "  Rejections:      " << rejection_count_ << "\n";
        std::cout << std::string(47, '-') << "\n";
        std::cout << "  Initial Capital: $" << format_number(tracker.initial_capital()) << "\n";
        std::cout << "  Final Equity:    $" << format_number(tracker.total_equity()) << "\n";
        std::cout << "  Total P&L:       $" << format_number(total_pnl)
                  << " (" << pnl_pct << "%)\n";
        std::cout << "  Realized P&L:    $" << format_number(tracker.total_realized_pnl()) << "\n";
        std::cout << "  Unrealized P&L:  $" << format_number(tracker.total_unrealized_pnl()) << "\n";
        std::cout << std::string(47, '-') << "\n";
        std::cout << "  Sharpe Ratio:    " << std::setprecision(4) << metrics.sharpe << "\n";
        std::cout << "  Sortino Ratio:   " << metrics.sortino << "\n";
        std::cout << "  Calmar Ratio:    " << metrics.calmar << "\n";
        std::cout << "  Max Drawdown:    " << std::setprecision(2) << (metrics.max_drawdown * 100.0) << "%\n";
        std::cout << "  Volatility:      " << (metrics.volatility * 100.0) << "%\n";
        std::cout << "  Skewness:        " << std::setprecision(4) << metrics.skewness << "\n";
        std::cout << "  Kurtosis:        " << metrics.kurtosis << "\n";
        std::cout << std::string(47, '=') << "\n";

        // Print open positions
        bool has_open = false;
        for (auto& [sym, pos] : tracker.positions()) {
            if (pos.quantity != 0) {
                if (!has_open) {
                    std::cout << "  Open Positions:\n";
                    has_open = true;
                }
                std::cout << "    " << sym << ": " << pos.quantity
                          << " @ avg " << std::setprecision(2) << pos.avg_cost
                          << " (mkt " << pos.market_price
                          << ", unreal=$" << format_number(pos.unrealized_pnl()) << ")\n";
            }
        }
        if (has_open) {
            std::cout << std::string(47, '=') << "\n";
        }

        std::cout << "\n";

        // Append summary to CSV
        if (csv_file_.is_open()) {
            csv_file_ << "\n# Summary: " << strategy_name
                      << " | P&L=$" << total_pnl
                      << " | Sharpe=" << metrics.sharpe
                      << " | MaxDD=" << metrics.max_drawdown << "\n";
        }
    }

    const std::string& csv_path() const { return csv_path_; }
    int fill_count() const { return fill_count_; }
    int rejection_count() const { return rejection_count_; }

private:
    std::string log_dir_;
    std::string csv_path_;
    bool verbose_;
    std::ofstream csv_file_;
    int fill_count_ = 0;
    int rejection_count_ = 0;

    void ensure_log_dir() {
        mkdir(log_dir_.c_str(), 0755);
    }

    void open_csv() {
        csv_path_ = log_dir_ + "/trades_" + file_timestamp() + ".csv";
        csv_file_.open(csv_path_);
        if (csv_file_.is_open()) {
            csv_file_ << "Date,Symbol,Side,Quantity,Price,Position,AvgCost,RealizedPnL,Cash,Equity\n";
        }
    }

    static std::string file_timestamp() {
        time_t now = time(nullptr);
        struct tm* t = localtime(&now);
        char buf[32];
        strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", t);
        return std::string(buf);
    }

    static std::string side_str(orderbook::Side side) {
        return (side == orderbook::Side::Buy) ? "BUY" : "SELL";
    }

    static std::string format_volume(uint64_t vol) {
        if (vol >= 1000000000) return format_number(static_cast<double>(vol) / 1e9) + "B";
        if (vol >= 1000000)    return format_number(static_cast<double>(vol) / 1e6) + "M";
        if (vol >= 1000)       return format_number(static_cast<double>(vol) / 1e3) + "K";
        return std::to_string(vol);
    }

    static std::string format_number(double val) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.2f", val);
        return std::string(buf);
    }
};

} // namespace trading

#endif // QUANTPRICER_TRADING_LOGGING_H

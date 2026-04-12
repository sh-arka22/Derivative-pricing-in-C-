#ifndef QUANTPRICER_TRADING_RISK_GUARD_H
#define QUANTPRICER_TRADING_RISK_GUARD_H

// ============================================================================
// Risk Guard — Pre-trade checks + portfolio monitoring
//
// Two responsibilities:
//   1. Pre-trade checks: validate each OrderRequest before execution
//   2. Portfolio monitoring: compute VaR, check kill switch
//
// check_order() rules (evaluated in order, first rejection wins):
//   1. Order quantity > max_order_size → REJECT
//   2. Post-trade position notional > max_position_notional → REJECT
//   3. Post-trade portfolio notional > max_portfolio_notional → REJECT
//   4. Total P&L < -max_loss → REJECT (kill switch)
//
// monitor() computes:
//   - Portfolio VaR using risk::parametric_var()
//   - Kill switch = (total_pnl < -max_loss) OR (VaR > max_var_99)
// ============================================================================

#include "trading/config.h"
#include "trading/simulated_exchange.h"
#include "trading/position_tracker.h"
#include "risk/risk.h"

#include <string>
#include <vector>
#include <cmath>
#include <utility>

namespace trading {

// ============================================================================
// Risk Check Result — per-order verdict
// ============================================================================

struct RiskCheckResult {
    bool approved;
    std::string reason;

    static RiskCheckResult approve() { return {true, ""}; }
    static RiskCheckResult reject(const std::string& r) { return {false, r}; }
};

// ============================================================================
// Risk Status — portfolio-level health snapshot
// ============================================================================

struct RiskStatus {
    double portfolio_var = 0.0;
    double total_pnl = 0.0;
    double portfolio_notional = 0.0;
    bool kill_switch = false;
    std::string message;
};

// ============================================================================
// RiskGuard
// ============================================================================

class RiskGuard {
public:
    explicit RiskGuard(const RiskLimits& limits)
        : limits_(limits) {}

    // ================================================================
    // Check a single order against limits
    // ================================================================

    RiskCheckResult check_order(
        const OrderRequest& order,
        const PositionTracker& tracker) const
    {
        // Rule 1: Max order size
        if (order.quantity > static_cast<int>(limits_.max_order_size)) {
            return RiskCheckResult::reject(
                "order size " + std::to_string(order.quantity) +
                " exceeds limit " + std::to_string(static_cast<int>(limits_.max_order_size)));
        }

        // Rule 2: Position notional limit
        const auto& pos = tracker.get_position(order.symbol);
        int fill_signed = (order.side == orderbook::Side::Buy)
                              ? +order.quantity : -order.quantity;
        int projected_qty = pos.quantity + fill_signed;
        double projected_notional = std::abs(projected_qty) * pos.market_price;

        // Use order limit_price as fallback if market_price is 0
        if (pos.market_price < 1e-10 && order.limit_price > 0) {
            projected_notional = std::abs(projected_qty) * order.limit_price;
        }

        if (projected_notional > limits_.max_position_notional) {
            return RiskCheckResult::reject(
                "position notional $" + format_number(projected_notional) +
                " exceeds limit $" + format_number(limits_.max_position_notional));
        }

        // Rule 3: Portfolio notional limit
        double current_portfolio_notional = tracker.portfolio_notional();
        double current_position_notional = pos.notional();
        double new_portfolio_notional = current_portfolio_notional
            - current_position_notional + projected_notional;

        if (new_portfolio_notional > limits_.max_portfolio_notional) {
            return RiskCheckResult::reject(
                "portfolio notional $" + format_number(new_portfolio_notional) +
                " exceeds limit $" + format_number(limits_.max_portfolio_notional));
        }

        // Rule 4: Loss limit (kill switch)
        if (tracker.total_pnl() < -limits_.max_loss) {
            return RiskCheckResult::reject(
                "loss limit breached: P&L=$" + format_number(tracker.total_pnl()) +
                " < -$" + format_number(limits_.max_loss));
        }

        return RiskCheckResult::approve();
    }

    // ================================================================
    // Filter a batch of orders — returns only approved ones
    // ================================================================

    std::vector<OrderRequest> check_orders(
        const std::vector<OrderRequest>& orders,
        const PositionTracker& tracker,
        std::vector<std::pair<OrderRequest, std::string>>* rejections = nullptr) const
    {
        std::vector<OrderRequest> approved;
        for (auto& order : orders) {
            auto result = check_order(order, tracker);
            if (result.approved) {
                approved.push_back(order);
            } else if (rejections) {
                rejections->push_back({order, result.reason});
            }
        }
        return approved;
    }

    // ================================================================
    // Monitor portfolio health
    // ================================================================

    RiskStatus monitor(const PositionTracker& tracker) const {
        RiskStatus status;
        status.total_pnl = tracker.total_pnl();
        status.portfolio_notional = tracker.portfolio_notional();

        // Estimate daily portfolio volatility
        // Simple approach: assume 2% daily vol scaled by notional
        double daily_vol = status.portfolio_notional * 0.02;

        if (daily_vol > 1e-10) {
            auto var_result = risk::parametric_var(0.0, daily_vol, 0.99);
            status.portfolio_var = var_result.var;
        }

        // Kill switch conditions
        bool loss_breached = (status.total_pnl < -limits_.max_loss);
        bool var_breached = (status.portfolio_var > limits_.max_var_99);

        status.kill_switch = loss_breached || var_breached;

        if (loss_breached) {
            status.message = "KILL SWITCH: loss limit breached (P&L=$" +
                             format_number(status.total_pnl) + ")";
        } else if (var_breached) {
            status.message = "KILL SWITCH: VaR limit breached (VaR=$" +
                             format_number(status.portfolio_var) + ")";
        }

        return status;
    }

    const RiskLimits& limits() const { return limits_; }

private:
    RiskLimits limits_;

    static std::string format_number(double val) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.2f", val);
        return std::string(buf);
    }
};

} // namespace trading

#endif // QUANTPRICER_TRADING_RISK_GUARD_H

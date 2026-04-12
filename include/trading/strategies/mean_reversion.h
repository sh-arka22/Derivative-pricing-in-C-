#ifndef QUANTPRICER_TRADING_STRATEGIES_MEAN_REVERSION_H
#define QUANTPRICER_TRADING_STRATEGIES_MEAN_REVERSION_H

// ============================================================================
// Mean Reversion Strategy
// ============================================================================
// Logic:
//   1. Maintain a rolling window of `lookback` close prices per symbol
//   2. Compute z-score = (current_price - mean) / stddev
//   3. Signal rules:
//      - z < -entry_z AND not already long  → BUY (undervalued)
//      - z > +entry_z AND not already short → SELL (overvalued)
//      - |z| < exit_z AND position != 0     → FLATTEN (mean reverted)
//
// Parameters (from Config.strategy_params):
//   lookback       = 20 bars (rolling window)
//   entry_z        = 2.0 (standard deviations from mean)
//   exit_z         = 0.5 (close to mean)
//   position_size  = 100 shares per signal
// ============================================================================

#include "trading/strategy.h"

#include <deque>
#include <cmath>
#include <numeric>

namespace trading {

class MeanReversionStrategy : public Strategy {
public:
    void init(const Config& config) override {
        lookback_      = config.strategy_params.mr_lookback;
        entry_z_       = config.strategy_params.mr_entry_z;
        exit_z_        = config.strategy_params.mr_exit_z;
        position_size_ = config.strategy_params.mr_position_size;
    }

    // Equity-only strategy — returns empty option_orders
    StrategyOrders on_bar(
        const MarketSnapshot& snapshot,
        const PositionTracker& tracker,
        const SimulatedExchange& exchange) override
    {
        (void)exchange;
        std::vector<OrderRequest> orders;

        for (auto& [symbol, bar] : snapshot.bars) {
            // Update rolling price history
            auto& history = price_history_[symbol];
            history.push_back(bar.close);

            // Wait for enough data
            if (static_cast<int>(history.size()) < lookback_) continue;

            // Trim to lookback window
            while (static_cast<int>(history.size()) > lookback_) {
                history.pop_front();
            }

            // Compute z-score
            double z = compute_zscore(history);

            // Current position
            int current_qty = tracker.get_position(symbol).quantity;

            if (z < -entry_z_ && current_qty <= 0) {
                // Undervalued — go long
                // If short, cover + go long in one order
                int qty = position_size_;
                if (current_qty < 0) {
                    qty += std::abs(current_qty);
                }
                orders.push_back({symbol, orderbook::Side::Buy, qty, 0.0});

            } else if (z > entry_z_ && current_qty >= 0) {
                // Overvalued — go short
                int qty = position_size_;
                if (current_qty > 0) {
                    qty += current_qty;
                }
                orders.push_back({symbol, orderbook::Side::Sell, qty, 0.0});

            } else if (std::abs(z) < exit_z_ && current_qty != 0) {
                // Mean reverted — flatten
                if (current_qty > 0) {
                    orders.push_back({symbol, orderbook::Side::Sell, current_qty, 0.0});
                } else {
                    orders.push_back({symbol, orderbook::Side::Buy, std::abs(current_qty), 0.0});
                }
            }
        }

        return StrategyOrders{orders, {}};
    }

    std::string name() const override { return "mean_reversion"; }

private:
    int lookback_       = 20;
    double entry_z_     = 2.0;
    double exit_z_      = 0.5;
    int position_size_  = 100;

    std::map<std::string, std::deque<double>> price_history_;

    // z = (current_price - mean) / stddev
    // Returns 0 if stddev is near zero (all prices identical)
    double compute_zscore(const std::deque<double>& prices) const {
        if (prices.empty()) return 0.0;

        double n = static_cast<double>(prices.size());
        double sum = std::accumulate(prices.begin(), prices.end(), 0.0);
        double mean = sum / n;

        double sq_sum = 0.0;
        for (double p : prices) {
            sq_sum += (p - mean) * (p - mean);
        }
        double stddev = std::sqrt(sq_sum / n);

        if (stddev < 1e-10) return 0.0;

        return (prices.back() - mean) / stddev;
    }
};

} // namespace trading

#endif // QUANTPRICER_TRADING_STRATEGIES_MEAN_REVERSION_H

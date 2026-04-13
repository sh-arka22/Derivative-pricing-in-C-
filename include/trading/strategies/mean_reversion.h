#ifndef QUANTPRICER_TRADING_STRATEGIES_MEAN_REVERSION_H
#define QUANTPRICER_TRADING_STRATEGIES_MEAN_REVERSION_H

// ============================================================================
// Mean Reversion Strategy — Production-Quality Z-Score + OU Calibration
// ============================================================================
//
// Three layers of statistical machinery:
//
//   Layer 1: Ornstein-Uhlenbeck calibration via OLS
//     Regress delta_x(t) = a + b * x(t-1) + epsilon
//     Recover: kappa = -b, theta = -a/b, sigma = std(residuals)
//     Half-life = ln(2) / kappa
//
//   Layer 2: Hurst exponent (variance-ratio method)
//     For each lag in [2, max_lag]:
//       tau(lag) = std(x[i+lag] - x[i] for all valid i)
//     H = slope of linear regression: log(tau) ~ H * log(lag)
//     H < 0.5 = mean-reverting, H = 0.5 = random walk, H > 0.5 = trending
//
//   Layer 3: Production risk controls
//     - Eligibility filter: Hurst < max_hurst AND half_life in [min, max]
//     - Adaptive lookback: clamp(round(half_life), 5, 40)
//     - Max holding period: force flatten at mult * half_life bars
//     - Cooldown: wait N bars after closing before re-entering
//     - Min volatility: skip dead markets
//
// Signal logic (same core as basic z-score):
//   z < -entry_z AND not long  → BUY  (undervalued)
//   z > +entry_z AND not short → SELL (overvalued)
//   |z| < exit_z AND position  → FLATTEN (mean reverted)
//
// References:
//   - Ernie Chan, "Algorithmic Trading" — half-life via OLS
//   - Leung & Li (2015) — optimal mean reversion trading
//   - letianzj.github.io/mean-reversion.html — ADF + Hurst + half-life
//   - Hudson & Thames arbitragelab — OU model calibration
// ============================================================================

#include "trading/strategy.h"

#include <deque>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace trading {

// ============================================================================
// OLS 1D — Simple linear regression y = a + b*x
// ============================================================================
// Used for OU calibration (delta_x on x_lag) and Hurst (log-log fit).
// Sum-based closed-form — no matrix library needed.
// ============================================================================

struct OLSResult {
    double a;           // intercept
    double b;           // slope
    double r_squared;   // goodness of fit
};

inline OLSResult ols_1d(const double* x, const double* y, int n) {
    if (n < 3) return {0, 0, 0};

    double sx = 0, sy = 0, sxy = 0, sxx = 0;
    for (int i = 0; i < n; i++) {
        sx  += x[i];
        sy  += y[i];
        sxy += x[i] * y[i];
        sxx += x[i] * x[i];
    }

    double denom = n * sxx - sx * sx;
    if (std::abs(denom) < 1e-15) return {0, 0, 0};

    double b = (n * sxy - sx * sy) / denom;
    double a = (sy - b * sx) / n;

    // R-squared
    double mean_y = sy / n;
    double ss_tot = 0, ss_res = 0;
    for (int i = 0; i < n; i++) {
        ss_tot += (y[i] - mean_y) * (y[i] - mean_y);
        double pred = a + b * x[i];
        ss_res += (y[i] - pred) * (y[i] - pred);
    }
    double r2 = (ss_tot > 1e-15) ? 1.0 - ss_res / ss_tot : 0.0;

    return {a, b, r2};
}

// ============================================================================
// OUParams — Ornstein-Uhlenbeck process parameters
// ============================================================================
// dX = kappa * (theta - X) * dt + sigma * dW
//
// kappa:     mean-reversion speed (larger = faster reversion)
// theta:     long-run equilibrium level
// sigma:     diffusion coefficient (volatility of the process)
// half_life: time for deviation to decay by half = ln(2) / kappa
// ============================================================================

struct OUParams {
    double kappa      = 0;
    double theta      = 0;
    double sigma      = 0;
    double half_life  = 0;
    double r_squared  = 0;
    bool   valid      = false;  // false if b >= 0 (no mean reversion)
};

// ============================================================================
// calibrate_ou — Estimate OU parameters from a price series via OLS
// ============================================================================
// Discretization: x(t) - x(t-1) = a + b * x(t-1) + epsilon
//   kappa = -b        (dt = 1 for daily)
//   theta = -a / b
//   sigma = std(residuals)
//   half_life = ln(2) / kappa
//
// Returns valid=false if b >= 0 (series is not mean-reverting).
// Minimum 15 data points required for a stable estimate.
// ============================================================================

inline OUParams calibrate_ou(const std::deque<double>& prices) {
    int n = static_cast<int>(prices.size());
    if (n < 15) return {};

    // Build x_lag and delta_x vectors
    std::vector<double> x_lag(n - 1), delta_x(n - 1);
    for (int i = 0; i < n - 1; i++) {
        x_lag[i]   = prices[i];
        delta_x[i] = prices[i + 1] - prices[i];
    }

    auto ols = ols_1d(x_lag.data(), delta_x.data(), n - 1);

    // b must be negative for mean reversion
    if (ols.b >= 0) return {0, 0, 0, 0, ols.r_squared, false};

    double kappa = -ols.b;          // dt = 1 for daily bars
    double theta = -ols.a / ols.b;

    // Residual standard deviation = sigma estimate
    double sum_sq = 0;
    for (int i = 0; i < n - 1; i++) {
        double pred = ols.a + ols.b * x_lag[i];
        double resid = delta_x[i] - pred;
        sum_sq += resid * resid;
    }
    double sigma = std::sqrt(sum_sq / (n - 1));

    double half_life = std::log(2.0) / kappa;

    return {kappa, theta, sigma, half_life, ols.r_squared, true};
}

// ============================================================================
// estimate_hurst — Hurst exponent via variance-ratio method
// ============================================================================
// For lag in [2, max_lag]:
//   tau(lag) = std(x[i+lag] - x[i])   for all valid i
//
// Fit: log(tau) = H * log(lag) + const
// H is the slope of this log-log regression.
//
// H < 0.5 → mean-reverting
// H = 0.5 → random walk (geometric Brownian motion)
// H > 0.5 → trending / momentum
//
// Returns 0.5 (random walk) if insufficient data.
// ============================================================================

inline double estimate_hurst(const std::deque<double>& prices, int max_lag = 20) {
    int n = static_cast<int>(prices.size());
    if (n < max_lag + 5) return 0.5;

    std::vector<double> log_lags, log_tau;

    for (int lag = 2; lag <= max_lag; lag++) {
        int count = n - lag;
        if (count < 5) break;

        // Compute std(prices[i+lag] - prices[i])
        double sum = 0, sum_sq = 0;
        for (int i = 0; i < count; i++) {
            double diff = prices[i + lag] - prices[i];
            sum    += diff;
            sum_sq += diff * diff;
        }
        double mean = sum / count;
        double var  = sum_sq / count - mean * mean;
        if (var < 1e-15) continue;

        double tau = std::sqrt(var);
        log_lags.push_back(std::log(static_cast<double>(lag)));
        log_tau.push_back(std::log(tau));
    }

    if (static_cast<int>(log_lags.size()) < 3) return 0.5;

    // Slope of log-log regression = Hurst exponent
    auto ols = ols_1d(log_lags.data(), log_tau.data(),
                      static_cast<int>(log_lags.size()));
    return ols.b;
}

// ============================================================================
// SymbolState — Per-symbol tracking for the enhanced strategy
// ============================================================================

struct SymbolState {
    std::deque<double> price_history;
    OUParams ou;
    double hurst            = 0.5;
    bool   eligible         = false;
    int    holding_bars     = 0;     // bars since last entry
    int    cooldown_remaining = 0;   // bars until re-entry allowed
    int    bars_since_calib = 0;     // bars since last OU calibration
    int    effective_lookback = 20;  // current z-score window
};

// ============================================================================
// MeanReversionStrategy
// ============================================================================

class MeanReversionStrategy : public Strategy {
public:
    void init(const Config& config) override {
        lookback_          = config.strategy_params.mr_lookback;
        entry_z_           = config.strategy_params.mr_entry_z;
        exit_z_            = config.strategy_params.mr_exit_z;
        position_size_     = config.strategy_params.mr_position_size;
        max_hurst_         = config.strategy_params.mr_max_hurst;
        min_half_life_     = config.strategy_params.mr_min_half_life;
        max_half_life_     = config.strategy_params.mr_max_half_life;
        min_daily_vol_     = config.strategy_params.mr_min_daily_vol;
        max_holding_mult_  = config.strategy_params.mr_max_holding_mult;
        cooldown_bars_     = config.strategy_params.mr_cooldown_bars;
        recalib_interval_  = config.strategy_params.mr_recalib_interval;
        adaptive_lookback_ = config.strategy_params.mr_adaptive_lookback;
    }

    StrategyOrders on_bar(
        const MarketSnapshot& snapshot,
        const PositionTracker& tracker,
        const SimulatedExchange& exchange) override
    {
        (void)exchange;
        std::vector<OrderRequest> orders;

        for (auto& [symbol, bar] : snapshot.bars) {
            auto& state = states_[symbol];

            // ---- 1. Update price history ----
            state.price_history.push_back(bar.close);

            // Cap history at a reasonable maximum
            while (static_cast<int>(state.price_history.size()) > 120) {
                state.price_history.pop_front();
            }

            int n = static_cast<int>(state.price_history.size());

            // ---- 2. Recalibrate OU + Hurst periodically ----
            state.bars_since_calib++;
            if (state.bars_since_calib >= recalib_interval_ && n >= 15) {
                state.ou    = calibrate_ou(state.price_history);
                state.hurst = estimate_hurst(state.price_history,
                                             std::min(20, n / 2));

                // Eligibility: mean-reverting AND tradeable half-life
                state.eligible = state.ou.valid
                    && state.hurst < max_hurst_
                    && state.ou.half_life >= min_half_life_
                    && state.ou.half_life <= max_half_life_;

                // Adaptive lookback: use half-life as z-score window
                if (adaptive_lookback_ && state.ou.valid) {
                    int hl = static_cast<int>(std::round(state.ou.half_life));
                    state.effective_lookback = std::max(5, std::min(hl, 40));
                } else {
                    state.effective_lookback = lookback_;
                }

                state.bars_since_calib = 0;
            }

            // ---- 3. Manage cooldown and holding counters ----
            int current_qty = tracker.get_position(symbol).quantity;

            if (current_qty != 0) {
                state.holding_bars++;
            } else {
                if (state.holding_bars > 0) {
                    // Just went flat — start cooldown
                    state.cooldown_remaining = cooldown_bars_;
                }
                state.holding_bars = 0;
            }
            if (state.cooldown_remaining > 0) {
                state.cooldown_remaining--;
            }

            // ---- 4. Wait for enough data ----
            if (n < state.effective_lookback) continue;

            // ---- 5. Compute z-score with effective lookback ----
            double z = compute_zscore(state.price_history, state.effective_lookback);

            // ---- 6. Min volatility check ----
            double vol = compute_volatility(state.price_history, state.effective_lookback);
            if (bar.close > 1e-10 && vol / bar.close < min_daily_vol_) {
                continue;  // dead market — skip
            }

            // ---- 7. Max holding period: force flatten ----
            if (current_qty != 0 && state.ou.valid) {
                int max_hold = static_cast<int>(
                    max_holding_mult_ * state.ou.half_life);
                max_hold = std::max(max_hold, 5);  // at least 5 bars
                if (state.holding_bars > max_hold) {
                    if (current_qty > 0) {
                        orders.push_back({symbol, orderbook::Side::Sell,
                                          current_qty, 0.0});
                    } else {
                        orders.push_back({symbol, orderbook::Side::Buy,
                                          std::abs(current_qty), 0.0});
                    }
                    continue;
                }
            }

            // ---- 8. Entry signals (only if eligible + no cooldown) ----
            if (state.eligible && state.cooldown_remaining == 0) {
                if (z < -entry_z_ && current_qty <= 0) {
                    int qty = position_size_;
                    if (current_qty < 0) {
                        qty += std::abs(current_qty);
                    }
                    orders.push_back({symbol, orderbook::Side::Buy, qty, 0.0});
                    if (current_qty == 0) state.holding_bars = 0;

                } else if (z > entry_z_ && current_qty >= 0) {
                    int qty = position_size_;
                    if (current_qty > 0) {
                        qty += current_qty;
                    }
                    orders.push_back({symbol, orderbook::Side::Sell, qty, 0.0});
                    if (current_qty == 0) state.holding_bars = 0;
                }
            }

            // ---- 9. Exit signals (always allowed, even if ineligible) ----
            if (std::abs(z) < exit_z_ && current_qty != 0) {
                if (current_qty > 0) {
                    orders.push_back({symbol, orderbook::Side::Sell,
                                      current_qty, 0.0});
                } else {
                    orders.push_back({symbol, orderbook::Side::Buy,
                                      std::abs(current_qty), 0.0});
                }
            }
        }

        return StrategyOrders{orders, {}};
    }

    std::string name() const override { return "mean_reversion"; }

    // ================================================================
    // Diagnostics — print OU calibration state for all tracked symbols
    // ================================================================

    void print_calibration() const {
        std::cout << std::fixed;
        std::cout << "\n--- OU Calibration & Hurst Exponent ---\n";
        std::cout << std::setw(6) << "Sym"
                  << std::setw(9) << "kappa"
                  << std::setw(10) << "theta"
                  << std::setw(9) << "sigma"
                  << std::setw(8) << "t_half"
                  << std::setw(6) << "R2"
                  << std::setw(7) << "Hurst"
                  << std::setw(5) << "LB"
                  << std::setw(9) << "Status"
                  << "\n";

        for (auto& [sym, state] : states_) {
            std::cout << std::setw(6) << sym;
            if (state.ou.valid) {
                std::cout << std::setprecision(4)
                          << std::setw(9) << state.ou.kappa
                          << std::setprecision(2)
                          << std::setw(10) << state.ou.theta
                          << std::setprecision(4)
                          << std::setw(9) << state.ou.sigma
                          << std::setprecision(1)
                          << std::setw(8) << state.ou.half_life
                          << std::setprecision(3)
                          << std::setw(6) << state.ou.r_squared
                          << std::setprecision(3)
                          << std::setw(7) << state.hurst
                          << std::setw(5) << state.effective_lookback
                          << std::setw(9)
                          << (state.eligible ? "ELIG" : "SKIP");
            } else {
                std::cout << std::setw(9) << "-"
                          << std::setw(10) << "-"
                          << std::setw(9) << "-"
                          << std::setw(8) << "-"
                          << std::setw(6) << "-"
                          << std::setprecision(3)
                          << std::setw(7) << state.hurst
                          << std::setw(5) << state.effective_lookback
                          << std::setw(9) << "NO_MR";
            }
            std::cout << "\n";
        }
        std::cout << std::setprecision(2);
    }

    const std::map<std::string, SymbolState>& states() const { return states_; }

private:
    // Config params
    int    lookback_          = 20;
    double entry_z_           = 2.0;
    double exit_z_            = 0.5;
    int    position_size_     = 100;
    double max_hurst_         = 0.5;
    double min_half_life_     = 2.0;
    double max_half_life_     = 42.0;
    double min_daily_vol_     = 0.005;
    int    max_holding_mult_  = 3;
    int    cooldown_bars_     = 3;
    int    recalib_interval_  = 5;
    bool   adaptive_lookback_ = true;

    std::map<std::string, SymbolState> states_;

    // ================================================================
    // Z-score with configurable lookback window
    // ================================================================

    static double compute_zscore(const std::deque<double>& prices, int lookback) {
        int n = static_cast<int>(prices.size());
        if (n < lookback || lookback < 2) return 0.0;

        // Use the last `lookback` prices
        double sum = 0;
        int start = n - lookback;
        for (int i = start; i < n; i++) {
            sum += prices[i];
        }
        double mean = sum / lookback;

        double sq_sum = 0;
        for (int i = start; i < n; i++) {
            sq_sum += (prices[i] - mean) * (prices[i] - mean);
        }
        double stddev = std::sqrt(sq_sum / lookback);

        if (stddev < 1e-10) return 0.0;
        return (prices.back() - mean) / stddev;
    }

    // ================================================================
    // Daily volatility (stddev of 1-day returns) over lookback window
    // ================================================================

    static double compute_volatility(const std::deque<double>& prices, int lookback) {
        int n = static_cast<int>(prices.size());
        if (n < lookback + 1 || lookback < 2) return 0.0;

        int start = n - lookback;
        double sum = 0, sum_sq = 0;
        int count = 0;
        for (int i = start; i < n; i++) {
            if (prices[i - 1] > 1e-10) {
                double ret = (prices[i] - prices[i - 1]) / prices[i - 1];
                sum    += ret;
                sum_sq += ret * ret;
                count++;
            }
        }
        if (count < 2) return 0.0;

        double mean = sum / count;
        double var  = sum_sq / count - mean * mean;
        return std::sqrt(std::max(var, 0.0));
    }
};

} // namespace trading

#endif // QUANTPRICER_TRADING_STRATEGIES_MEAN_REVERSION_H

#ifndef QUANTPRICER_TRADING_CONFIG_H
#define QUANTPRICER_TRADING_CONFIG_H

// ============================================================================
// Paper Trading Configuration — Day 1 (Revised: Self-Simulated)
//
// Loads all runtime configuration from a JSON file:
//   - Data file paths (CSV per symbol)
//   - Simulation parameters (spread, liquidity, slippage)
//   - Risk limits (per-position, portfolio, kill switch)
//   - Strategy selection
//
// No broker account, no API keys, no network dependencies.
// All execution is simulated locally using orderbook::OrderBook.
// ============================================================================

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <iomanip>

namespace trading {

// ============================================================================
// Simulation Parameters — controls the simulated exchange behavior
// ============================================================================

struct SimulationParams {
    double spread_bps       = 10.0;     // Bid-ask spread in basis points
    int    num_levels        = 5;        // Number of bid/ask levels to seed
    int    base_qty_per_level = 100;     // Base quantity at best level (thins outward)
    double slippage_bps     = 2.0;      // Additional slippage per level
    double initial_capital  = 100000.0;  // Starting cash
};

inline void from_json(const nlohmann::json& j, SimulationParams& s) {
    if (j.contains("spread_bps"))          j.at("spread_bps").get_to(s.spread_bps);
    if (j.contains("num_levels"))          j.at("num_levels").get_to(s.num_levels);
    if (j.contains("base_qty_per_level"))  j.at("base_qty_per_level").get_to(s.base_qty_per_level);
    if (j.contains("slippage_bps"))        j.at("slippage_bps").get_to(s.slippage_bps);
    if (j.contains("initial_capital"))     j.at("initial_capital").get_to(s.initial_capital);
}

// ============================================================================
// Risk Limits — enforced by RiskGuard
// ============================================================================

struct RiskLimits {
    double max_position_notional  = 50000.0;    // Max $ per symbol
    double max_portfolio_notional = 200000.0;    // Max $ total
    double max_loss               = 5000.0;      // Kill switch: max total loss
    double max_order_size         = 500;          // Max shares per order
    double max_var_99             = 10000.0;      // 99% VaR limit
    int    max_open_orders        = 20;           // Max concurrent open orders
};

inline void from_json(const nlohmann::json& j, RiskLimits& r) {
    if (j.contains("max_position_notional"))  j.at("max_position_notional").get_to(r.max_position_notional);
    if (j.contains("max_portfolio_notional")) j.at("max_portfolio_notional").get_to(r.max_portfolio_notional);
    if (j.contains("max_loss"))               j.at("max_loss").get_to(r.max_loss);
    if (j.contains("max_order_size"))         j.at("max_order_size").get_to(r.max_order_size);
    if (j.contains("max_var_99"))             j.at("max_var_99").get_to(r.max_var_99);
    if (j.contains("max_open_orders"))        j.at("max_open_orders").get_to(r.max_open_orders);
}

// ============================================================================
// Strategy Parameters — per-strategy tunable knobs
// ============================================================================
// Each strategy reads its own section. Unknown keys are silently ignored.
// All fields have sensible defaults so the JSON section is optional.
// ============================================================================

struct StrategyParams {
    // Mean Reversion
    int    mr_lookback       = 20;
    double mr_entry_z        = 2.0;
    double mr_exit_z         = 0.5;
    int    mr_position_size  = 100;

    // Delta Hedge
    double dh_risk_free      = 0.05;
    double dh_vol            = 0.20;
    double dh_time_to_expiry = 1.0;
    int    dh_contracts      = 10;
    int    dh_rebalance_threshold = 5;

    // Momentum
    int    mom_fast_period   = 10;
    int    mom_slow_period   = 50;
    int    mom_position_size = 100;

    // Pairs Trading
    int    pt_lookback       = 60;
    double pt_entry_z        = 2.0;
    double pt_exit_z         = 0.5;
    int    pt_position_size  = 100;

    // Gamma Scalp
    double gs_vol            = 0.20;
    double gs_time_to_expiry = 0.25;
    int    gs_contracts      = 5;
    int    gs_rebalance_threshold = 10;
};

inline void from_json(const nlohmann::json& j, StrategyParams& sp) {
    // Mean Reversion
    if (j.contains("mr_lookback"))       j.at("mr_lookback").get_to(sp.mr_lookback);
    if (j.contains("mr_entry_z"))        j.at("mr_entry_z").get_to(sp.mr_entry_z);
    if (j.contains("mr_exit_z"))         j.at("mr_exit_z").get_to(sp.mr_exit_z);
    if (j.contains("mr_position_size"))  j.at("mr_position_size").get_to(sp.mr_position_size);

    // Delta Hedge
    if (j.contains("dh_risk_free"))      j.at("dh_risk_free").get_to(sp.dh_risk_free);
    if (j.contains("dh_vol"))            j.at("dh_vol").get_to(sp.dh_vol);
    if (j.contains("dh_time_to_expiry")) j.at("dh_time_to_expiry").get_to(sp.dh_time_to_expiry);
    if (j.contains("dh_contracts"))      j.at("dh_contracts").get_to(sp.dh_contracts);
    if (j.contains("dh_rebalance_threshold")) j.at("dh_rebalance_threshold").get_to(sp.dh_rebalance_threshold);

    // Momentum
    if (j.contains("mom_fast_period"))   j.at("mom_fast_period").get_to(sp.mom_fast_period);
    if (j.contains("mom_slow_period"))   j.at("mom_slow_period").get_to(sp.mom_slow_period);
    if (j.contains("mom_position_size")) j.at("mom_position_size").get_to(sp.mom_position_size);

    // Pairs Trading
    if (j.contains("pt_lookback"))       j.at("pt_lookback").get_to(sp.pt_lookback);
    if (j.contains("pt_entry_z"))        j.at("pt_entry_z").get_to(sp.pt_entry_z);
    if (j.contains("pt_exit_z"))         j.at("pt_exit_z").get_to(sp.pt_exit_z);
    if (j.contains("pt_position_size"))  j.at("pt_position_size").get_to(sp.pt_position_size);

    // Gamma Scalp
    if (j.contains("gs_vol"))            j.at("gs_vol").get_to(sp.gs_vol);
    if (j.contains("gs_time_to_expiry")) j.at("gs_time_to_expiry").get_to(sp.gs_time_to_expiry);
    if (j.contains("gs_contracts"))      j.at("gs_contracts").get_to(sp.gs_contracts);
    if (j.contains("gs_rebalance_threshold")) j.at("gs_rebalance_threshold").get_to(sp.gs_rebalance_threshold);
}

// ============================================================================
// Main Config
// ============================================================================

struct Config {
    // Data: symbol -> CSV file path
    std::map<std::string, std::string> data_files;

    // Simulation
    SimulationParams sim;

    // Risk
    RiskLimits risk_limits;

    // Strategy
    std::string strategy = "mean_reversion";
    StrategyParams strategy_params;

    // Logging
    std::string log_dir = "logs";
    bool verbose = true;

    // Convenience: list of symbols (derived from data_files keys)
    std::vector<std::string> symbols() const {
        std::vector<std::string> syms;
        for (auto& [sym, _] : data_files)
            syms.push_back(sym);
        return syms;
    }

    /// Load config from a JSON file
    static Config from_json_file(const std::string& path) {
        std::ifstream ifs(path);
        if (!ifs.is_open()) {
            throw std::runtime_error("Cannot open config file: " + path);
        }

        nlohmann::json j;
        try {
            ifs >> j;
        } catch (const nlohmann::json::parse_error& e) {
            throw std::runtime_error("JSON parse error in " + path + ": " + e.what());
        }

        Config cfg;

        // Required: data_files mapping
        if (!j.contains("data_files") || !j["data_files"].is_object()) {
            throw std::runtime_error("Config must contain 'data_files' object mapping symbol -> CSV path");
        }
        for (auto& [sym, path_val] : j["data_files"].items()) {
            cfg.data_files[sym] = path_val.get<std::string>();
        }

        // Optional overrides
        if (j.contains("simulation"))   cfg.sim = j.at("simulation").get<SimulationParams>();
        if (j.contains("risk_limits"))  cfg.risk_limits = j.at("risk_limits").get<RiskLimits>();
        if (j.contains("strategy"))        j.at("strategy").get_to(cfg.strategy);
        if (j.contains("strategy_params")) cfg.strategy_params = j.at("strategy_params").get<StrategyParams>();
        if (j.contains("log_dir"))         j.at("log_dir").get_to(cfg.log_dir);
        if (j.contains("verbose"))         j.at("verbose").get_to(cfg.verbose);

        return cfg;
    }

    /// Print config summary
    void print_summary() const {
        std::cout << "=== Paper Trading Config (Self-Simulated) ===" << std::endl;
        std::cout << "  Data Files:" << std::endl;
        for (auto& [sym, path] : data_files)
            std::cout << "    " << sym << " -> " << path << std::endl;
        std::cout << "  Strategy:   " << strategy << std::endl;
        std::cout << "  Simulation:" << std::endl;
        std::cout << "    Spread:     " << sim.spread_bps << " bps" << std::endl;
        std::cout << "    Levels:     " << sim.num_levels << std::endl;
        std::cout << "    Qty/Level:  " << sim.base_qty_per_level << std::endl;
        std::cout << "    Slippage:   " << sim.slippage_bps << " bps" << std::endl;
        std::cout << "    Capital:    $" << std::fixed << std::setprecision(0) << sim.initial_capital << std::endl;
        std::cout << "  Risk:" << std::endl;
        std::cout << "    Max Position:  $" << risk_limits.max_position_notional << std::endl;
        std::cout << "    Max Portfolio: $" << risk_limits.max_portfolio_notional << std::endl;
        std::cout << "    Max Loss:      $" << risk_limits.max_loss << std::endl;
        std::cout << "    Max VaR(99%):  $" << risk_limits.max_var_99 << std::endl;
        std::cout << std::setprecision(2) << "  Strategy Params:" << std::endl;
        if (strategy == "mean_reversion") {
            std::cout << "    lookback=" << strategy_params.mr_lookback
                      << " entry_z=" << strategy_params.mr_entry_z
                      << " exit_z=" << strategy_params.mr_exit_z
                      << " size=" << strategy_params.mr_position_size << std::endl;
        } else if (strategy == "delta_hedge") {
            std::cout << "    vol=" << strategy_params.dh_vol
                      << " T=" << strategy_params.dh_time_to_expiry
                      << " contracts=" << strategy_params.dh_contracts
                      << " rebal=" << strategy_params.dh_rebalance_threshold << std::endl;
        } else if (strategy == "momentum") {
            std::cout << "    fast=" << strategy_params.mom_fast_period
                      << " slow=" << strategy_params.mom_slow_period
                      << " size=" << strategy_params.mom_position_size << std::endl;
        } else if (strategy == "pairs_trading") {
            std::cout << "    lookback=" << strategy_params.pt_lookback
                      << " entry_z=" << strategy_params.pt_entry_z
                      << " exit_z=" << strategy_params.pt_exit_z
                      << " size=" << strategy_params.pt_position_size << std::endl;
        } else if (strategy == "gamma_scalp") {
            std::cout << "    vol=" << strategy_params.gs_vol
                      << " T=" << strategy_params.gs_time_to_expiry
                      << " contracts=" << strategy_params.gs_contracts
                      << " rebal=" << strategy_params.gs_rebalance_threshold << std::endl;
        }
        std::cout << "==============================================" << std::endl;
    }
};

} // namespace trading

#endif // QUANTPRICER_TRADING_CONFIG_H

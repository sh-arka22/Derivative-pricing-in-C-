# Paper Trading System — Complete Implementation Specification

> **Purpose:** This document is the single source of truth for building the self-simulated paper trading system on top of quantpricer. It contains exact struct definitions, method signatures, algorithms, and integration points so that a Claude 200k model can implement each day's work without hallucination or context overflow.
>
> **How to use:** Read this file + the relevant `docs/specs/dayNN.md` file for the day you're implementing. Each day spec contains the complete code blueprint with every line of logic specified.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Existing Module API Reference](#2-existing-module-api-reference)
3. [New Files to Create](#3-new-files-to-create)
4. [Data Flow & Event Loop](#4-data-flow--event-loop)
5. [Day-by-Day Build Plan](#5-day-by-day-build-plan)
6. [Struct & Class Specifications](#6-struct--class-specifications)
7. [Algorithm Details](#7-algorithm-details)
8. [Configuration Schema](#8-configuration-schema)
9. [Testing & Verification](#9-testing--verification)
10. [Design Decisions & Constraints](#10-design-decisions--constraints)
11. [Common Pitfalls & Edge Cases](#11-common-pitfalls--edge-cases)
12. [Per-Day Spec File Index](#12-per-day-spec-file-index)

---

## 1. Architecture Overview

```
                    ┌─────────────────────────────────────────────┐
                    │              paper_trader.cpp                │
                    │           (main event loop)                  │
                    └──────┬──────┬──────┬──────┬──────┬──────────┘
                           │      │      │      │      │
                    ┌──────▼──┐ ┌─▼────┐ │  ┌───▼──┐ ┌▼─────────┐
                    │ Market  │ │Simul.│ │  │ Risk │ │  Trade   │
                    │  Data   │ │Exch. │ │  │Guard │ │  Logger  │
                    │ Replay  │ │      │ │  │      │ │          │
                    └─────────┘ └──────┘ │  └──────┘ └──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │     Strategy         │
                              │  (pluggable, picks   │
                              │   one of 3 impls)    │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  Position Tracker    │
                              │  (holdings, P&L)     │
                              └─────────────────────┘
```

**Key principle:** Single-threaded, deterministic. Same CSV + same config = identical output every run.

**Data model:** Two data sources merged into a single `MarketSnapshot` per date:
- **OHLCV bars** (`data/*.csv`) — daily stock prices, used by all strategies
- **Option chains** (`data/options/*_options.csv`) — real bid/ask, IV, Greeks from philippdubach/options-data. Used by options strategies (delta hedge, gamma scalp, market maker). Equity-only strategies ignore this.

**Execution model:**
1. `MarketDataReplay` loads OHLCV CSVs; `OptionDataReplay` loads option chain CSVs
2. For each date: build `MarketSnapshot` merging bars + option chains
3. Seed `SimulatedExchange` with fresh liquidity from bars
4. Pass `MarketSnapshot` to the active `Strategy` → it returns stock and option orders
5. `RiskGuard` filters each order (reject if limits breached)
6. Stock orders go to `SimulatedExchange` → returns `vector<Fill>`
7. Option orders fill at real bid/ask from chain data (buy at ask, sell at bid)
8. `PositionTracker` updates stock + option holdings and P&L from fills
9. `TradeLogger` logs everything to terminal + CSV
10. After all bars: compute session summary via `risk::compute_metrics()`

---

## 2. Existing Module API Reference

> These are the EXACT signatures from the existing codebase. Do NOT modify these files.

### 2.1 OrderBook (`include/orderbook/orderbook.h`)

```cpp
namespace orderbook {
  enum class Side : uint8_t { Buy, Sell };
  enum class OrderType : uint8_t { Limit, Market };
  using OrderId = uint64_t;

  struct Order {
    OrderId id; Side side; OrderType type;
    double price; uint32_t quantity, filled_qty; Timestamp timestamp;
    bool is_filled() const;
  };

  struct Trade {
    OrderId buy_order_id, sell_order_id;
    double price; uint32_t quantity; Timestamp timestamp;
  };

  class OrderBook {
  public:
    explicit OrderBook(const std::string& symbol = "AAPL");
    OrderId add_limit_order(Side side, double price, uint32_t qty);
    OrderId add_market_order(Side side, uint32_t qty);
    bool cancel_order(OrderId id);
    bool has_bids() const;  bool has_asks() const;
    double best_bid() const;  double best_ask() const;  // throw if empty
    double spread() const;  double mid_price() const;
    uint32_t bid_depth_at_best() const;  uint32_t ask_depth_at_best() const;
    const std::vector<Trade>& trades() const;
    size_t trade_count() const;
    std::string to_string(size_t depth = 5) const;
    const std::string& symbol() const;
  };
  // NOTE: No clear() or reset() method exists. Create a new OrderBook each bar.
}
```

### 2.2 Risk (`include/risk/risk.h`)

```cpp
namespace risk {
  struct VaRResult { double var, cvar, confidence; size_t num_scenarios; };
  struct PerformanceMetrics {
    double mean_return, volatility, sharpe, sortino, calmar, max_drawdown, skewness, kurtosis;
  };

  VaRResult parametric_var(double mu, double sigma, double alpha);
  PerformanceMetrics compute_metrics(
    const std::vector<double>& returns, double rf = 0.0, double periods_per_year = 252.0);
  VaRResult scenario_var(std::vector<double> pnl, double alpha);
}
```

### 2.3 Black-Scholes (`include/greeks/black_scholes.h`)

```cpp
namespace bs {
  double N(double x);  double n(double x);
  double d1(double S, double K, double r, double T, double sigma);
  double call_price(double S, double K, double r, double T, double sigma);
  double put_price(double S, double K, double r, double T, double sigma);
  double delta_call(double S, double K, double r, double T, double sigma);
  double delta_put(double S, double K, double r, double T, double sigma);
  double gamma(double S, double K, double r, double T, double sigma);
  double vega(double S, double K, double r, double T, double sigma);
  double theta_call(double S, double K, double r, double T, double sigma);
  double rho_call(double S, double K, double r, double T, double sigma);
}
```

### 2.4 Greeks Engine (`include/greeks/greeks_engine.h`)

```cpp
struct GreeksResult { double delta, gamma, vega, theta, rho; };
GreeksResult analytic_greeks_call(double S, double K, double r, double T, double sigma);
GreeksResult analytic_greeks_put(double S, double K, double r, double T, double sigma);
```

### 2.5 Implied Vol (`include/vol/implied_vol.h`)

```cpp
struct ImpliedVolResult { double sigma; int iterations; double residual; bool converged; };
ImpliedVolResult implied_vol_newton(
  double market_price, double S, double K, double r, double T,
  double sigma_init = 0.3, double tol = 1e-8, int max_iter = 100);
```

### 2.6 RNG (`include/rng/rng.h`)

```cpp
class MersenneTwisterRNG : public RandomNumberGenerator { /* std::mt19937_64 */ };
std::vector<double> generate_normals(RandomNumberGenerator& rng, size_t n);
```

### 2.7 Config (`include/trading/config.h`) — ALREADY EXISTS

```cpp
namespace trading {
  struct SimulationParams {
    double spread_bps = 10.0; int num_levels = 5; int base_qty_per_level = 100;
    double slippage_bps = 2.0; double initial_capital = 100000.0;
  };
  struct RiskLimits {
    double max_position_notional = 50000.0; double max_portfolio_notional = 200000.0;
    double max_loss = 5000.0; double max_order_size = 500;
    double max_var_99 = 10000.0; int max_open_orders = 20;
  };
  struct Config {
    std::map<std::string, std::string> data_files;
    SimulationParams sim; RiskLimits risk_limits;
    std::string strategy = "mean_reversion";
    std::string log_dir = "logs"; bool verbose = true;
    std::vector<std::string> symbols() const;
    static Config from_json_file(const std::string& path);
    void print_summary() const;
  };
}
```

---

## 3. New Files to Create

### Complete file list (in build order):

| Day | File | Namespace | Purpose |
|-----|------|-----------|---------|
| 1 | `tools/download_data.py` | — | Downloads real OHLCV from Yahoo Finance |
| 1 | `tools/download_options_data.py` | — | Downloads real option chains from philippdubach/options-data (Parquet → CSV) |
| 2 | `include/trading/market_data.h` | `trading` | Bar, OptionQuote, OptionChain, MarketDataReplay, OptionDataReplay |
| 2 | `include/trading/market_snapshot.h` | `trading` | MarketSnapshot — unified per-date view (bars + option chains) |
| 2 | `include/trading/option_types.h` | `trading` | OptionOrderRequest, OptionFill, find_best_option(), fill_option_at_market() |
| 2 | `tools/generate_sample_data.cpp` | — | GBM-based synthetic OHLCV CSV generator (fallback if offline) |
| 3 | `include/trading/simulated_exchange.h` | `trading` | Wraps OrderBook, seeds liquidity, returns stock fills |
| 4 | `include/trading/position_tracker.h` | `trading` | Stock + option holdings, avg cost, realized/unrealized P&L, portfolio Greeks |
| 5 | `include/trading/strategy.h` | `trading` | Abstract Strategy base class (receives MarketSnapshot) |
| 5 | `include/trading/strategies/mean_reversion.h` | `trading` | Z-score mean reversion strategy (OHLCV only) |
| 6 | `include/trading/risk_guard.h` | `trading` | Pre-trade + real-time risk checks |
| 7 | `include/trading/logging.h` | `trading` | CSV trade log + terminal output + session summary |
| 8 | `src/paper_trader.cpp` | — | **REWRITE** — full event loop with MarketSnapshot |
| 9 | `include/trading/strategies/delta_hedge.h` | `trading` | Delta hedge using **real option chain data** (market delta, real bid/ask) |
| 9 | `include/trading/strategies/momentum.h` | `trading` | Dual MA crossover strategy (OHLCV only) |
| 10 | — | — | Polish, SIGINT handler, smoke test (mid-project checkpoint) |
| 11 | `include/trading/strategies/pairs_trading.h` | `trading` | Statistical arbitrage pairs strategy (OHLCV only) |
| 12 | `include/trading/strategies/gamma_scalp.h` | `trading` | Gamma scalping using **real option chain data** (market Greeks, real straddle prices) |
| 13 | `include/trading/backtester.h` | `trading` | Multi-strategy comparison framework |
| 13 | `tools/run_backtest.cpp` | — | Backtest runner executable |
| 14 | `include/trading/report_generator.h` | `trading` | Equity curve, trade analysis, rolling metrics CSV |
| 15 | `include/trading/strategies/market_maker.h` | `trading` | Capstone: **options market maker** — quotes real options, vol surface, delta hedge, P&L by Greek |

### Data files:

| File | Source | Purpose |
|------|--------|---------|
| `data/AAPL.csv` ... `data/META.csv` | Yahoo Finance via `download_data.py` | Daily OHLCV for 8 symbols |
| `data/options/AAPL_options.csv` ... `data/options/META_options.csv` | philippdubach/options-data via `download_options_data.py` | Daily option chain snapshots (bid, ask, IV, Greeks, volume, OI) |

### Header pattern (all new headers follow this):

```cpp
#ifndef QUANTPRICER_TRADING_XXXX_H
#define QUANTPRICER_TRADING_XXXX_H

#include <...>

namespace trading {
  // All code inline (header-only, matches project pattern)
} // namespace trading

#endif
```

---

## 4. Data Flow & Event Loop

### Main loop pseudo-code (paper_trader.cpp, Day 8):

```
parse CLI args (config_path, --strategy name)
config = Config::from_json_file(config_path)
override config.strategy if --strategy given

replay = MarketDataReplay(config.data_files)
option_replay = OptionDataReplay(config.option_data_files)    // NEW: real option chains
exchange = SimulatedExchange(config.sim)
tracker = PositionTracker(config.sim.initial_capital)
strategy = create_strategy(config.strategy, config)
risk_guard = RiskGuard(config.risk_limits)
logger = TradeLogger(config.log_dir, config.verbose)

strategy->init(config)
logger.log_header(config)

while (replay.has_next()) {
    auto bars = replay.next()
    std::string date = bars.begin()->second.date

    // 1. Build unified MarketSnapshot (OHLCV + option chains for this date)
    MarketSnapshot snapshot;
    snapshot.date = date;
    snapshot.bars = bars;
    if (!config.option_data_files.empty()) {
        snapshot.option_chains = option_replay.get_chains(date);  // may be empty
    }

    // 2. Update marks (stock + option positions)
    for (auto& [sym, bar] : bars) {
        tracker.on_price(sym, bar.close)
        exchange.seed_liquidity(sym, bar)
    }
    for (auto& [sym, chain] : snapshot.option_chains) {
        tracker.update_option_marks(chain)    // mark option positions to market
    }

    // 3. Risk monitoring (pre-strategy)
    auto risk_status = risk_guard.monitor(tracker)
    if (risk_status.kill_switch) {
        logger.log_kill_switch(risk_status)
        break
    }

    // 4. Strategy generates signals (receives full snapshot)
    auto [stock_orders, option_orders] = strategy->on_bar(snapshot, tracker, exchange)

    // 5. Risk guard filters stock orders
    auto approved = risk_guard.check_orders(stock_orders, tracker)

    // 6. Execute stock orders via SimulatedExchange
    for (auto& order : approved) {
        auto fills = exchange.submit_order(order, date)
        for (auto& fill : fills) {
            tracker.on_fill(fill)
            strategy->on_fill(fill)
            logger.log_fill(fill, tracker)
        }
    }

    // 7. Execute option orders at real bid/ask from chain data
    for (auto& opt_order : option_orders) {
        auto opt_fill = fill_option_at_market(opt_order, snapshot, date)
        if (opt_fill.has_value()) {
            tracker.on_option_fill(opt_fill.value())
            strategy->on_option_fill(opt_fill.value())
            logger.log_option_fill(opt_fill.value(), tracker)
        }
    }

    // 8. Log bar summary
    logger.log_bar(snapshot, tracker, risk_guard)
    tracker.on_day_end(date)
}

// Session summary
strategy->on_stop()
auto daily_returns = tracker.daily_return_series()
auto metrics = risk::compute_metrics(daily_returns)
logger.log_session_summary(tracker, metrics)
```

---

## 5. Day-by-Day Build Plan

### Day 1: Setup + Config + Data ✅ DONE
- `include/trading/config.h` — complete (SimulationParams, RiskLimits, StrategyParams, Config)
- `config/paper_trading.json` — complete (all strategy params, risk limits, sim params, option_data_files)
- `src/paper_trader.cpp` — stub that verifies all engine deps compile
- `tools/download_data.py` — downloads real OHLCV from Yahoo Finance via yfinance
- `tools/download_options_data.py` — downloads real option chains from philippdubach/options-data (Parquet → CSV)
- Real option chain data for 8 symbols in `data/options/` (~304 MB total, 2025 data, max 90 DTE)
- CMakeLists.txt, Makefile, directories, .gitignore — all set up

### Day 2: Market Data + MarketSnapshot + Option Types
- **Files:** `include/trading/market_data.h` ✅ (already created: Bar, OptionQuote, OptionChain, MarketDataReplay, OptionDataReplay)
- **New files:** `include/trading/market_snapshot.h`, `include/trading/option_types.h`
- **Also:** `tools/generate_sample_data.cpp` (synthetic fallback)
- **Spec:** `docs/specs/day02_market_data.md`
- **Key change from original:** MarketSnapshot bundles bars + option chains. OptionOrderRequest/OptionFill for option execution. find_best_option() helper for ATM selection.
- **Verification:** Load OHLCV + option chains, iterate in sync, print snapshot summary showing "SPY: O=589 H=591 L=587 C=590, options: 3400 contracts, 17 expirations"

### Day 3: Simulated Exchange (+ option fill mechanics)
- **Files:** `include/trading/simulated_exchange.h`
- **Spec:** `docs/specs/day03_simulated_exchange.md`
- **Key change from original:** Add `fill_option_at_market()` — fills option orders at real bid/ask from chain data (buy at ask, sell at bid). No order book needed for options; the CSV bid/ask IS the market.
- **Verification:** Seed stock book + submit stock orders as before. Also: fill option order, verify buy fills at ask price, sell fills at bid price.

### Day 4: Position Tracker (+ option positions)
- **Files:** `include/trading/position_tracker.h`
- **Spec:** `docs/specs/day04_position_tracker.md`
- **Key change from original:** Add `OptionPosition` struct (symbol, expiration, strike, is_call, contracts, avg_cost, mark, Greeks). Add `on_option_fill()`, `update_option_marks(chain)`, `net_delta(symbol)`, `portfolio_greeks()`.
- **Verification:** Stock buy/sell cycle as before. Also: option buy at ask → mark to market next day → sell at bid → verify realized P&L includes spread cost.

### Day 5: Strategy Base + Mean Reversion
- **Files:** `include/trading/strategy.h`, `include/trading/strategies/mean_reversion.h`
- **Spec:** `docs/specs/day05_strategy_mean_reversion.md`
- **Key change from original:** `on_bar()` signature takes `const MarketSnapshot&` instead of `const std::map<std::string, Bar>&`. Returns `StrategyOrders` struct containing both `vector<OrderRequest>` (stock) and `vector<OptionOrderRequest>` (options). Equity-only strategies return empty option orders.
- **Verification:** Mean reversion unchanged in behavior — accesses `snapshot.bars`, ignores `snapshot.option_chains`.

### Day 6: Risk Guard
- **Files:** `include/trading/risk_guard.h`
- **Spec:** `docs/specs/day06_risk_guard.md`
- **No change from original.** Risk checks apply to stock orders. Option risk managed via portfolio Greeks limits (net delta, net gamma thresholds) — added as optional checks.
- **Verification:** Same as before + verify option position notional counted in portfolio notional.

### Day 7: Logging
- **Files:** `include/trading/logging.h`
- **Spec:** `docs/specs/day07_logging.md`
- **Key change from original:** Add `log_option_fill()` for option trades. Add Greeks summary in bar log when option positions exist.
- **Verification:** CSV includes option fills with strike, expiry, type, fill price, IV.

### Day 8: Main Loop Integration
- **Files:** `src/paper_trader.cpp` (rewrite)
- **Spec:** `docs/specs/day08_main_loop.md`
- **Key change from original:** Construct `OptionDataReplay`, build `MarketSnapshot` per date, pass to strategy, handle both stock and option fills, update option marks daily.
- **Verification:** Full run: CSV + option chains → MarketSnapshot → strategy → stock fills + option fills → P&L → summary. Run with mean_reversion (OHLCV only) to verify backward compatibility.

### Day 9: Delta Hedge + Momentum Strategies
- **Files:** `include/trading/strategies/delta_hedge.h`, `include/trading/strategies/momentum.h`
- **Spec:** `docs/specs/day09_strategies.md`
- **MAJOR CHANGE — Delta Hedge now uses REAL option chain data:**
  - On first bar: `find_best_option(chain, spot, dh_target_dte, call)` → buy contracts at real ask price
  - Each bar: look up same (exp, strike) in chain → read `OptionQuote.delta` directly (NOT bs::delta_call with hardcoded vol)
  - Hedge with stock using market delta
  - Track option P&L from real mark changes, stock P&L from tracker
  - Roll option when DTE < 5 (sell at bid, buy new at ask — captures real roll cost)
- **Momentum unchanged** — OHLCV only, no options.
- **Config change:** `dh_vol` and `dh_time_to_expiry` replaced by `dh_target_dte = 30`
- **Verification:** Run delta hedge on AAPL. Output must show: "Selected AAPL $245C exp 2025-02-21 at ask=$5.30, market delta=0.55". Each day: "delta=0.52, target=-520 shares, current=-480, rebalance: SELL 40"

### Day 10: Polish + Mid-Project Checkpoint
- **Files:** Minor updates to paper_trader.cpp, CMakeLists.txt
- **Spec:** `docs/specs/day10_polish.md`
- **Verification:** SIGINT handler, all 4 strategies pass (mean_rev, momentum, delta_hedge, pairs), risk guard stress test. Delta hedge P&L uses real option marks.

### Day 11: Pairs Trading Strategy
- **Files:** `include/trading/strategies/pairs_trading.h`
- **Spec:** `docs/specs/day11_pairs_trading.md`
- **No change from original.** OHLCV-only strategy.
- **Verification:** Simultaneous long/short on two symbols, spread z-score signals.

### Day 12: Gamma Scalping Strategy
- **Files:** `include/trading/strategies/gamma_scalp.h`
- **Spec:** `docs/specs/day12_gamma_scalping.md`
- **MAJOR CHANGE — uses REAL option chain data:**
  - On first bar: buy ATM call + ATM put from chain at `gs_target_dte` → real straddle at ask prices
  - Each bar: read delta, gamma, theta directly from `OptionQuote` for both legs
  - Straddle delta = call.delta + put.delta → hedge with stock
  - P&L attribution from REAL Greeks: gamma P&L = 0.5 × Γ × ΔS², theta P&L = Θ × dt, actual = mark change, residual = vega + higher order
  - Report realized vol vs implied vol comparison at on_stop()
  - Roll when DTE < 5
- **Config change:** `gs_vol` and `gs_time_to_expiry` replaced by `gs_target_dte = 30`
- **Verification:** Run on AAPL. Output must show: "Straddle: $245C + $245P, entry=$6.55 (real ask), gamma=0.10, theta=-0.15". Daily: gamma P&L, theta cost, net. Final: "realized vol=0.28 vs implied vol=0.22 → gamma profit"

### Day 13: Backtesting Framework
- **Files:** `include/trading/backtester.h`, `tools/run_backtest.cpp`
- **Spec:** `docs/specs/day13_backtesting.md`
- **Verification:** All 6 strategies compared side-by-side, ranked by Sharpe. Options strategies show additional metrics: total spread cost, gamma/theta ratio.

### Day 14: Performance Report Generator
- **Files:** `include/trading/report_generator.h`
- **Spec:** `docs/specs/day14_report_generator.md`
- **Key addition:** For options strategies, report includes: P&L by Greek (delta, gamma, theta, vega, spread), IV vs realized vol chart data, roll costs.
- **Verification:** Equity curve CSV, trade analysis, rolling metrics, Greek P&L attribution.

### Day 15: Options Market Maker Strategy (Capstone)
- **Files:** `include/trading/strategies/market_maker.h`
- **Spec:** `docs/specs/day15_market_maker.md`
- **COMPLETE REWRITE — now an OPTIONS market maker, not a stock market maker:**
  - Each bar: read full option chain, select `mm_quote_expiries` × `mm_strikes_around_atm` options to quote
  - For each option: `theo = quote.mark`, `my_bid = theo - edge`, `my_ask = theo + edge`
  - Fill logic: capture edge when quote is within market spread
  - After fills: compute net delta across all option positions → delta hedge with stock
  - **P&L attribution by Greek** (the interview talking point):
    - Spread capture = sum of (fill price - theo) across all fills
    - Delta P&L = net_delta × ΔS
    - Gamma P&L = 0.5 × net_gamma × ΔS²
    - Theta P&L = net_theta × dt
    - Vega P&L = net_vega × ΔIV
  - Manages inventory across strikes/expirations
  - Vol surface from market IVs drives pricing
- **Config:** `mm_quote_expiries=2, mm_strikes_around_atm=3, mm_edge_bps=5.0, mm_max_option_inventory=20, mm_delta_hedge_threshold=50.0`
- **This is the capstone** — ties EVERY module together: BS pricing, Greeks, implied vol, vol surface, risk, orderbook, real option chain data, P&L attribution
- **Verification:** Run on SPY. Output must show: quotes placed per bar, fills with real spread, net Greeks, delta hedges, daily P&L breakdown by Greek component. Final: "Spread income: $X, Delta P&L: $Y, Gamma P&L: $Z, Theta P&L: $W"

---

## 6. Struct & Class Specifications

### 6.1 Bar (market_data.h)

```cpp
struct Bar {
    std::string symbol;
    std::string date;       // "YYYY-MM-DD" format
    double open;
    double high;
    double low;
    double close;
    uint64_t volume;

    double mid() const { return (high + low) / 2.0; }
    double range() const { return high - low; }
    double body() const { return close - open; }  // positive = green bar
};
```

### 6.2 MarketDataReplay (market_data.h)

```cpp
class MarketDataReplay {
public:
    // Constructor: loads all CSVs immediately
    // data_files: map of symbol → CSV file path
    explicit MarketDataReplay(const std::map<std::string, std::string>& data_files);

    // Returns true if there are more bars to iterate
    bool has_next() const;

    // Returns the next set of bars (one per symbol, aligned by date)
    // Symbols that have no bar for this date are omitted from the map
    std::map<std::string, Bar> next();

    // Total number of unique dates across all symbols
    size_t total_bars() const;

    // Current bar index (0-based)
    size_t current_index() const;

private:
    // All bars loaded per symbol, sorted by date
    std::map<std::string, std::vector<Bar>> all_bars_;

    // Unified sorted date list (union of all symbols' dates)
    std::vector<std::string> dates_;

    // Current position in dates_
    size_t current_idx_ = 0;
};
```

**CSV format expected (Yahoo Finance style):**
```
Date,Open,High,Low,Close,Adj Close,Volume
2025-01-02,185.20,186.10,184.50,185.80,185.80,45000000
```

**Parsing rules:**
- Skip first line (header)
- Split by comma
- Date = field[0], Open = field[1], High = field[2], Low = field[3], Close = field[4], Volume = field[6]
- Skip field[5] (Adj Close) — use raw Close
- Ignore lines that fail to parse (log warning if verbose)

### 6.3 SimulatedExchange (simulated_exchange.h)

```cpp
// Order request from strategy → exchange
struct OrderRequest {
    std::string symbol;
    orderbook::Side side;
    int quantity;                    // signed: positive = buy intent qty
    double limit_price;             // 0.0 = market order
    bool is_market() const { return limit_price <= 0.0; }
};

// Fill report from exchange → position tracker
struct Fill {
    std::string symbol;
    orderbook::Side side;
    double price;
    int quantity;
    std::string date;               // date of the bar that generated this fill
};

class SimulatedExchange {
public:
    explicit SimulatedExchange(const trading::SimulationParams& params);

    // Seed fresh liquidity for a symbol around the bar's price
    // Creates a NEW OrderBook, places N bid + N ask levels
    void seed_liquidity(const std::string& symbol, const Bar& bar);

    // Submit an order. Returns fills (may be empty if no liquidity / market order unfilled)
    std::vector<Fill> submit_order(const OrderRequest& order, const std::string& date);

    // Get the current book for a symbol (for strategy introspection)
    const orderbook::OrderBook& get_book(const std::string& symbol) const;

    // Check if a symbol has a seeded book
    bool has_book(const std::string& symbol) const;

private:
    SimulationParams params_;
    std::map<std::string, orderbook::OrderBook> books_;
};
```

**Liquidity seeding algorithm (CRITICAL — must match exactly):**

```
Given: bar with close price P, params with spread_bps, num_levels, base_qty_per_level, slippage_bps

half_spread = P * (spread_bps / 10000.0) / 2.0
slippage_per_level = P * (slippage_bps / 10000.0)

For level i = 0 to num_levels-1:
    offset = half_spread + i * slippage_per_level
    bid_price = P - offset
    ask_price = P + offset

    // Thinning: more liquidity at best, less at deeper levels
    qty = base_qty_per_level * (num_levels - i) / num_levels
    if (qty < 1) qty = 1

    book.add_limit_order(Side::Buy,  bid_price, qty)
    book.add_limit_order(Side::Sell, ask_price, qty)
```

**Order submission logic:**

```
Given: OrderRequest {symbol, side, quantity, limit_price}

If is_market():
    book.add_market_order(side, quantity)
Else:
    book.add_limit_order(side, limit_price, quantity)

// Extract fills from book.trades() (only the NEW trades since last call)
// Convert each orderbook::Trade to a Fill with the correct side, price, qty
```

**Important:** Track `trades_before` count before submitting, then extract `book.trades()[trades_before .. end]` as new fills.

### 6.4 PositionTracker (position_tracker.h)

```cpp
struct Position {
    std::string symbol;
    int quantity = 0;               // signed: +long, -short
    double avg_cost = 0.0;          // weighted average entry price
    double realized_pnl = 0.0;     // cumulative realized P&L for this symbol
    double market_price = 0.0;     // latest mark

    double unrealized_pnl() const {
        return quantity * (market_price - avg_cost);
    }
    double notional() const {
        return std::abs(quantity) * market_price;
    }
};

class PositionTracker {
public:
    explicit PositionTracker(double initial_capital);

    // Process a fill — updates position qty, avg cost, realized P&L
    void on_fill(const Fill& fill);

    // Update market price for mark-to-market
    void on_price(const std::string& symbol, double price);

    // Mark end of day — snapshots daily P&L for return series
    void on_day_end(const std::string& date);

    // Getters
    const Position& get_position(const std::string& symbol) const;
    bool has_position(const std::string& symbol) const;
    std::map<std::string, Position>& positions();
    const std::map<std::string, Position>& positions() const;

    double cash() const;
    double initial_capital() const;
    double total_equity() const;          // cash + sum(unrealized P&L)
    double total_realized_pnl() const;    // sum across all symbols
    double total_unrealized_pnl() const;  // sum across all symbols
    double total_pnl() const;             // realized + unrealized
    double portfolio_notional() const;    // sum of abs(qty * price)
    int total_trades() const;

    // Return series for risk::compute_metrics()
    std::vector<double> daily_return_series() const;

private:
    double initial_capital_;
    double cash_;
    int trade_count_ = 0;
    std::map<std::string, Position> positions_;

    // Daily equity snapshots for return computation
    std::vector<double> equity_history_;
    std::string last_date_;
};
```

**on_fill algorithm (CRITICAL — average cost accounting):**

```
Given: Fill {symbol, side, price, quantity}
Let pos = positions_[symbol]

fill_qty_signed = (side == Buy) ? +quantity : -quantity

If position is flat (qty == 0):
    pos.qty = fill_qty_signed
    pos.avg_cost = price

Else if adding to position (same direction):
    // Weighted average cost
    total_cost = pos.avg_cost * abs(pos.qty) + price * quantity
    pos.qty += fill_qty_signed
    pos.avg_cost = total_cost / abs(pos.qty)

Else (reducing/closing/reversing position):
    close_qty = min(quantity, abs(pos.qty))
    // Realize P&L on the closed portion
    if pos.qty > 0:  // was long, selling
        pos.realized_pnl += close_qty * (price - pos.avg_cost)
    else:             // was short, buying
        pos.realized_pnl += close_qty * (pos.avg_cost - price)

    remaining = quantity - close_qty
    pos.qty += fill_qty_signed

    if pos.qty == 0:
        pos.avg_cost = 0.0
    else if remaining > 0:
        // Reversed through zero — new position at fill price
        pos.avg_cost = price

Update cash:
    if side == Buy:  cash_ -= price * quantity
    if side == Sell: cash_ += price * quantity
trade_count_++
```

**on_day_end algorithm:**
```
equity = cash_ + sum(pos.qty * pos.market_price for all positions)
equity_history_.push_back(equity)
last_date_ = date
```

**daily_return_series algorithm:**
```
returns = []
for i = 1 to equity_history_.size()-1:
    returns.push_back((equity_history_[i] - equity_history_[i-1]) / equity_history_[i-1])
return returns
```

### 6.5 Strategy (strategy.h)

```cpp
// Return type for on_bar — strategies can generate stock AND option orders
struct StrategyOrders {
    std::vector<OrderRequest> stock_orders;
    std::vector<OptionOrderRequest> option_orders;  // from option_types.h
};

class Strategy {
public:
    virtual ~Strategy() = default;

    // Called once before the main loop
    virtual void init(const Config& config) = 0;

    // Called each bar — receives unified MarketSnapshot (OHLCV + option chains)
    // Equity strategies use snapshot.bars only; options strategies also use snapshot.option_chains
    virtual StrategyOrders on_bar(
        const MarketSnapshot& snapshot,
        const PositionTracker& tracker,
        const SimulatedExchange& exchange) = 0;

    // Called when a stock fill occurs (for strategy bookkeeping)
    virtual void on_fill(const Fill& fill) {}

    // Called when an option fill occurs
    virtual void on_option_fill(const OptionFill& fill) {}

    // Called at end of simulation
    virtual void on_stop() {}

    // Strategy name for display
    virtual std::string name() const = 0;
};

// Factory function — returns the selected strategy
inline std::unique_ptr<Strategy> create_strategy(const std::string& name, const Config& config);
```

### 6.6 MeanReversionStrategy (strategies/mean_reversion.h)

```cpp
class MeanReversionStrategy : public Strategy {
public:
    void init(const Config& config) override;

    std::vector<OrderRequest> on_bar(
        const std::map<std::string, Bar>& bars,
        const PositionTracker& tracker,
        const SimulatedExchange& exchange) override;

    std::string name() const override { return "mean_reversion"; }

private:
    int lookback_ = 20;        // rolling window size
    double entry_z_ = 2.0;     // enter when |z| > entry_z
    double exit_z_ = 0.5;      // exit when |z| < exit_z
    int position_size_ = 100;  // shares per signal

    // Per-symbol rolling price history
    std::map<std::string, std::deque<double>> price_history_;

    // Compute z-score from rolling window
    double compute_zscore(const std::deque<double>& prices) const;
};
```

**Z-score algorithm:**
```
Given: deque of last `lookback` close prices

mean = sum(prices) / n
variance = sum((p - mean)^2 for p in prices) / n
stddev = sqrt(variance)
z = (current_price - mean) / stddev    (return 0 if stddev < 1e-10)
```

**Signal logic:**
```
For each symbol with a bar:
    Append close to price_history_[symbol]
    If history.size() < lookback: skip (not enough data)
    Trim history to lookback (pop_front if needed)
    z = compute_zscore(history)

    current_qty = tracker.get_position(symbol).quantity

    if z < -entry_z_ and current_qty <= 0:
        BUY position_size_ at market  (price is undervalued)
    elif z > entry_z_ and current_qty >= 0:
        SELL position_size_ at market  (price is overvalued)
    elif abs(z) < exit_z_ and current_qty != 0:
        Close position: sell if long, buy if short
```

### 6.7 DeltaHedgeStrategy (strategies/delta_hedge.h) — REAL OPTION DATA

```cpp
class DeltaHedgeStrategy : public Strategy {
public:
    void init(const Config& config) override;

    StrategyOrders on_bar(
        const MarketSnapshot& snapshot,
        const PositionTracker& tracker,
        const SimulatedExchange& exchange) override;

    void on_option_fill(const OptionFill& fill) override;
    void on_stop() override;
    std::string name() const override { return "delta_hedge"; }

private:
    // Option selection params (from config)
    int target_dte_ = 30;              // target days to expiry
    int contracts_ = 10;               // option contracts to hold
    int rebalance_threshold_ = 5;      // shares diff before rehedging
    double risk_free_ = 0.05;

    // Tracked option position (selected from real chain data)
    std::string tracked_expiration_;    // "YYYY-MM-DD"
    double tracked_strike_ = 0.0;
    double entry_mark_ = 0.0;          // what we paid (ask price)
    double current_mark_ = 0.0;        // current market mark
    bool position_open_ = false;

    // P&L tracking
    double option_pnl_ = 0.0;         // realized option P&L (from rolls)
    double total_spread_cost_ = 0.0;   // cumulative bid-ask cost
    int rolls_ = 0;
};
```

**Delta hedge algorithm (REAL DATA):**
```
For the FIRST symbol only (primary underlying):
    spot = snapshot.bars[symbol].close
    chain = snapshot.option_chains[symbol]  // real option chain for today

    If chain is empty: skip (no option data for this date)

    If no position open:
        // SELECT: find ATM call closest to target_dte
        best = find_best_option(chain, spot, target_dte_, is_call=true)
        if best == nullptr: skip

        // BUY: generate OptionOrderRequest (buy contracts at ask)
        option_orders.push_back({symbol, best->expiration, best->strike,
                                 is_call=true, contracts_, is_buy=true})
        tracked_expiration_ = best->expiration
        tracked_strike_ = best->strike
        entry_mark_ = best->ask
        position_open_ = true
        RETURN (no hedge on entry bar)

    // LOOK UP current option in today's chain
    quote = find_quote(chain, tracked_expiration_, tracked_strike_, is_call=true)
    if quote == nullptr:
        // Option no longer in chain (expired or delisted) → force roll
        // Generate sell order, then select new option next bar
        position_open_ = false
        RETURN

    // READ MARKET DELTA (not computed from BS!)
    delta = quote->delta       // THIS is the key change
    current_mark_ = quote->mark

    // ROLL CHECK: if DTE < 5, close and reopen
    dte = quote->days_to_expiry_from(snapshot.date)
    if dte < 5:
        // SELL current option at bid
        option_orders.push_back({symbol, tracked_expiration_, tracked_strike_,
                                 is_call=true, contracts_, is_buy=false})
        position_open_ = false
        rolls_++
        RETURN (will select new option next bar)

    // HEDGE: same logic as before, but delta comes from MARKET
    target_shares = -round(delta * contracts_ * 100)
    current_shares = tracker.get_position(symbol).quantity

    diff = target_shares - current_shares
    if abs(diff) >= rebalance_threshold_:
        if diff > 0: stock_orders.push_back(BUY abs(diff) at market)
        else:        stock_orders.push_back(SELL abs(diff) at market)

on_stop():
    Print: "Option P&L: $X, Stock Hedge P&L: $Y, Total Spread Cost: $Z, Rolls: N"
```

### 6.8 MomentumStrategy (strategies/momentum.h)

```cpp
class MomentumStrategy : public Strategy {
public:
    void init(const Config& config) override;

    std::vector<OrderRequest> on_bar(
        const std::map<std::string, Bar>& bars,
        const PositionTracker& tracker,
        const SimulatedExchange& exchange) override;

    std::string name() const override { return "momentum"; }

private:
    int fast_period_ = 10;
    int slow_period_ = 50;
    int position_size_ = 100;

    // Per-symbol price history
    std::map<std::string, std::deque<double>> price_history_;

    // Per-symbol previous signal state (to detect crossovers)
    std::map<std::string, int> prev_signal_;  // +1=bullish, -1=bearish, 0=none

    double compute_ma(const std::deque<double>& prices, int period) const;
};
```

**Momentum algorithm:**
```
For each symbol:
    Append close to price_history_[symbol]
    If history.size() < slow_period_: skip

    fast_ma = average of last fast_period_ prices
    slow_ma = average of last slow_period_ prices

    signal = (fast_ma > slow_ma) ? +1 : -1
    prev = prev_signal_[symbol]  (default 0)

    if signal == +1 and prev != +1:  // golden cross
        // Go long: close short if any, then buy
        current_qty = tracker.get_position(symbol).quantity
        if current_qty < 0: BUY abs(current_qty) at market  // close short
        if current_qty <= 0: BUY position_size_ at market    // go long

    elif signal == -1 and prev != -1:  // death cross
        current_qty = tracker.get_position(symbol).quantity
        if current_qty > 0: SELL current_qty at market       // close long
        if current_qty >= 0: SELL position_size_ at market   // go short

    prev_signal_[symbol] = signal
```

### 6.9 RiskGuard (risk_guard.h)

```cpp
struct RiskCheckResult {
    bool approved;
    std::string reason;  // empty if approved
};

struct RiskStatus {
    double portfolio_var;
    double total_pnl;
    double portfolio_notional;
    bool kill_switch;       // true if max_loss breached
    std::string message;
};

class RiskGuard {
public:
    explicit RiskGuard(const RiskLimits& limits);

    // Check a single order against risk limits
    RiskCheckResult check_order(const OrderRequest& order, const PositionTracker& tracker) const;

    // Filter a vector of orders — returns only approved ones
    std::vector<OrderRequest> check_orders(
        const std::vector<OrderRequest>& orders, const PositionTracker& tracker) const;

    // Monitor portfolio state — check kill switch
    RiskStatus monitor(const PositionTracker& tracker) const;

private:
    RiskLimits limits_;
};
```

**check_order rules (evaluated in order, first rejection wins):**

```
1. Order size check:
   if order.quantity > limits_.max_order_size → REJECT "order size exceeds limit"

2. Position notional check (post-trade):
   projected_qty = current_qty + fill_qty_signed
   projected_notional = abs(projected_qty) * current_market_price
   if projected_notional > limits_.max_position_notional → REJECT "position notional limit"

3. Portfolio notional check (post-trade):
   projected_portfolio_notional = current_portfolio_notional
       - current_position_notional + projected_position_notional
   if projected_portfolio_notional > limits_.max_portfolio_notional → REJECT "portfolio notional limit"

4. P&L kill switch:
   if tracker.total_pnl() < -limits_.max_loss → REJECT "loss limit breached"

Otherwise → APPROVE
```

**monitor logic:**
```
var_result = risk::parametric_var(0.0, daily_vol_estimate, 0.99)
  where daily_vol_estimate = portfolio_notional * 0.02 / sqrt(252)  [simple approximation]

kill_switch = (tracker.total_pnl() < -limits_.max_loss)
           OR (var_result.var > limits_.max_var_99)
```

### 6.10 TradeLogger (logging.h)

```cpp
class TradeLogger {
public:
    TradeLogger(const std::string& log_dir, bool verbose);

    // Log config at start
    void log_header(const Config& config);

    // Log a bar (terminal only, if verbose)
    void log_bar(const std::map<std::string, Bar>& bars,
                 const PositionTracker& tracker,
                 const RiskGuard& risk_guard);

    // Log a fill (terminal + CSV)
    void log_fill(const Fill& fill, const PositionTracker& tracker);

    // Log risk events
    void log_risk_rejection(const OrderRequest& order, const std::string& reason);
    void log_kill_switch(const RiskStatus& status);

    // End-of-session summary
    void log_session_summary(const PositionTracker& tracker,
                             const risk::PerformanceMetrics& metrics);

private:
    std::string log_dir_;
    bool verbose_;
    std::ofstream csv_file_;   // trades CSV
    int fill_count_ = 0;
    int rejection_count_ = 0;

    void ensure_log_dir();     // create log_dir_ if needed
    std::string timestamp();   // current date/time string for filenames
};
```

**CSV format (trades log):**
```
Date,Symbol,Side,Quantity,Price,Position,AvgCost,RealizedPnL,Cash,Equity
2025-01-02,AAPL,BUY,100,185.75,100,185.75,0.00,81425.00,100000.00
```

**Terminal output format:**
```
[2025-01-02] BAR   AAPL  O=185.20 H=186.10 L=184.50 C=185.80  vol=45M
[2025-01-02] SIG   AAPL  z=-2.31 → BUY 100 @ MKT
[2025-01-02] FILL  AAPL  BUY  100 @ 185.85  pos=+100 avg=185.85
[2025-01-02] RISK  VaR=$2,341  notional=$18,585  P&L=$0.00
[2025-01-02] REJ   AAPL  SELL 600 — order size exceeds limit (max=500)
```

**Session summary format:**
```
═══════════════════════════════════════════
           SESSION SUMMARY
═══════════════════════════════════════════
  Strategy:        mean_reversion
  Bars Processed:  252
  Trades:          87
  Rejections:      3
───────────────────────────────────────────
  Initial Capital: $100,000.00
  Final Equity:    $99,857.63
  Total P&L:       -$142.37 (-0.14%)
  Realized P&L:    -$89.12
  Unrealized P&L:  -$53.25
───────────────────────────────────────────
  Sharpe Ratio:    -0.18
  Sortino Ratio:   -0.24
  Calmar Ratio:    -0.09
  Max Drawdown:    1.2%
  Volatility:      12.3%
  Skewness:        -0.31
  Kurtosis:        3.87
═══════════════════════════════════════════
```

---

## 7. Algorithm Details

### 7.1 Market Data Acquisition

**Primary (real data):** `tools/download_data.py` downloads from Yahoo Finance via yfinance.
```bash
# Download default symbols (AAPL, SPY, TSLA), 1 year of daily data
python3 tools/download_data.py

# Custom symbols or date range
python3 tools/download_data.py --symbols AAPL MSFT NVDA --period 2y
python3 tools/download_data.py --start 2024-01-01 --end 2025-01-01

# Or via Makefile
make download-data
```

Output: `data/AAPL.csv`, `data/SPY.csv`, `data/TSLA.csv` in Yahoo Finance CSV format.
Requires: `pip install yfinance` (already installed on this system).

**Fallback (synthetic data):** `tools/generate_sample_data.cpp` generates GBM-based OHLCV.
Use this when offline or when you need deterministic reproducible data.
```bash
make generate-data
```

### 7.2 Sample Data Generator (generate_sample_data.cpp) — Fallback

Generates synthetic OHLCV CSV files using GBM from `rng.h`.

```
Parameters per symbol:
  S0 = initial price
  mu = annual drift (e.g. 0.08)
  sigma = annual vol (e.g. 0.25)
  n_days = 252

Algorithm:
  dt = 1.0 / 252.0
  rng = MersenneTwisterRNG(seed)

  S = S0
  for day in 1..n_days:
      Z = generate_normals(rng, 1)[0]
      S_next = S * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

      // Simulate intraday OHLC from the daily move
      open = S
      close = S_next
      // High/Low: add random intraday volatility
      Z2 = abs(generate_normals(rng, 1)[0])
      intraday_range = S * sigma * sqrt(dt) * (0.5 + 0.5 * Z2)
      if close >= open:
          high = max(open, close) + intraday_range * 0.3
          low  = min(open, close) - intraday_range * 0.7
      else:
          high = max(open, close) + intraday_range * 0.7
          low  = min(open, close) - intraday_range * 0.3

      // Ensure high >= max(open,close) and low <= min(open,close)
      high = max(high, max(open, close))
      low  = min(low,  min(open, close))
      low  = max(low, 0.01)  // price can't go negative

      volume = base_volume * (0.8 + 0.4 * uniform_random)

      Write CSV line: date, open, high, low, close, close, volume
      S = S_next

Symbols to generate:
  AAPL: S0=185, mu=0.10, sigma=0.25, seed=42
  SPY:  S0=480, mu=0.08, sigma=0.15, seed=123
  TSLA: S0=250, mu=0.15, sigma=0.50, seed=456

Date generation:
  Start from 2025-01-02, skip weekends (Sat/Sun)
  Use simple date arithmetic (no library needed)
```

### 7.3 Position Sizing

All strategies use fixed-size orders (configurable via `position_size_` member). This keeps the system simple and testable. Variable sizing (Kelly criterion, etc.) is a future enhancement.

### 7.4 P&L Attribution

```
Total P&L = Realized P&L + Unrealized P&L
Realized P&L = sum of (exit_price - avg_cost) * closed_qty for all closed trades
Unrealized P&L = sum of (market_price - avg_cost) * current_qty for all open positions
Cash = initial_capital - sum(buy_value) + sum(sell_value) for all fills
Equity = Cash + sum(qty * market_price) for all positions
```

---

## 8. Configuration Schema

### 8.1 paper_trading.json

```json
{
    "data_files": {
        "AAPL": "data/AAPL.csv",
        "SPY": "data/SPY.csv",
        "TSLA": "data/TSLA.csv",
        "MSFT": "data/MSFT.csv",
        "NVDA": "data/NVDA.csv",
        "AMZN": "data/AMZN.csv",
        "GOOG": "data/GOOG.csv",
        "META": "data/META.csv"
    },
    "option_data_files": {
        "AAPL": "data/options/AAPL_options.csv",
        "SPY":  "data/options/SPY_options.csv",
        "TSLA": "data/options/TSLA_options.csv",
        "MSFT": "data/options/MSFT_options.csv",
        "NVDA": "data/options/NVDA_options.csv",
        "AMZN": "data/options/AMZN_options.csv",
        "GOOG": "data/options/GOOG_options.csv",
        "META": "data/options/META_options.csv"
    },
    "strategy": "mean_reversion",
    "strategy_params": {
        "mr_lookback": 20,
        "mr_entry_z": 2.0,
        "mr_exit_z": 0.5,
        "mr_position_size": 100,

        "dh_risk_free": 0.05,
        "dh_target_dte": 30,
        "dh_contracts": 10,
        "dh_rebalance_threshold": 5,

        "mom_fast_period": 10,
        "mom_slow_period": 50,
        "mom_position_size": 100,

        "pt_lookback": 60,
        "pt_entry_z": 2.0,
        "pt_exit_z": 0.5,
        "pt_position_size": 100,

        "gs_target_dte": 30,
        "gs_contracts": 5,
        "gs_rebalance_threshold": 10,

        "mm_quote_expiries": 2,
        "mm_strikes_around_atm": 3,
        "mm_edge_bps": 5.0,
        "mm_max_option_inventory": 20,
        "mm_delta_hedge_threshold": 50.0
    },
    "simulation": {
        "spread_bps": 10.0,
        "num_levels": 5,
        "base_qty_per_level": 100,
        "slippage_bps": 2.0,
        "initial_capital": 100000.0
    },
    "risk_limits": {
        "max_position_notional": 50000,
        "max_portfolio_notional": 200000,
        "max_loss": 5000,
        "max_order_size": 500,
        "max_var_99": 10000,
        "max_open_orders": 20
    },
    "log_dir": "logs",
    "verbose": true
}
```

### 8.2 CLI Interface

```
./build/paper_trader [config_path] [--strategy name]

Default config_path: "config/paper_trading.json"
--strategy overrides the JSON "strategy" field

Valid strategy names: "mean_reversion", "delta_hedge", "momentum", "pairs_trading", "gamma_scalp", "market_maker"
--report generates detailed CSV analytics (equity curve, trade analysis, rolling metrics)
```

---

## 9. Testing & Verification

### Per-day verification (run after each day's code):

| Day | Command | Expected Output |
|-----|---------|-----------------|
| 2 | `make paper-trade` | Prints bars + option chain summary ("SPY: 252 bars, 3400 options/day") |
| 3 | `make paper-trade` | Seeds stock books, fills stock + option orders at real bid/ask |
| 4 | `make paper-trade` | Stock + option P&L tracking, avg cost, mark-to-market |
| 5 | `make paper-trade` | Mean reversion signals (OHLCV only, ignores option chains) |
| 6 | `make paper-trade` | Oversized order rejected, kill switch, option notional in portfolio check |
| 7 | `make paper-trade` | CSV with stock + option fills, Greeks summary in bar log |
| 8 | `make paper-trade` | Full run with MarketSnapshot, all strategies work |
| 9 | `--strategy delta_hedge` | Selects real ATM call from chain, reads market delta, hedges, shows option P&L |
| 10 | Ctrl+C during run | Clean shutdown with partial summary |
| 11 | `--strategy pairs_trading` | Simultaneous long/short on AAPL+SPY (OHLCV only) |
| 12 | `--strategy gamma_scalp` | Real straddle from chain, market Greeks, gamma/theta attribution, realized vs implied vol |
| 13 | `./build/run_backtest` | 6 strategies compared, ranked by Sharpe |
| 14 | `--strategy delta_hedge --report` | P&L by Greek, IV vs realized vol, roll costs |
| 15 | `--strategy market_maker` | Options quotes, fills at real spread, net Greeks, delta hedge, P&L by Greek component |

### Smoke test script (Day 15 — final):

```bash
#!/bin/bash
set -e
echo "=== Smoke Test ==="

# Generate sample data
./build/generate_sample_data

# Run each strategy
for strat in mean_reversion delta_hedge momentum pairs_trading gamma_scalp market_maker; do
    echo "--- Testing $strat ---"
    ./build/paper_trader config/paper_trading.json --strategy $strat
    echo ""
done

# Verify log files exist
ls -la logs/trades_*.csv
echo "=== All tests passed ==="
```

---

## 10. Design Decisions & Constraints

### DO:
1. **Header-only** — all new files in `include/trading/` are header-only with `inline` functions (matches project pattern)
2. **namespace trading** — all new code lives in `namespace trading`
3. **Reuse existing modules** — use `orderbook::OrderBook`, `risk::compute_metrics`, `bs::delta_call`, etc. directly
4. **C++17** — use structured bindings, `std::optional`, `if constexpr`, `std::filesystem` where helpful
5. **Deterministic** — same input = same output, no random seeds that change per run (use config-driven seeds)
6. **New OrderBook per bar** — construct fresh `orderbook::OrderBook(symbol)` each bar since there's no `clear()` method

### DON'T:
1. **No Boost** — no Boost.Beast, Boost.Asio, or any Boost libraries
2. **No network** — no HTTP, no WebSocket, no API keys, no broker accounts
3. **No threading** — no mutexes, no atomics, no thread pools
4. **No UI** — terminal output only, no Streamlit, no web server
5. **No new external deps** — only nlohmann/json (already installed) and standard library
6. **Don't modify existing headers** — the 17 existing pricing/risk/orderbook headers are frozen
7. **Don't use `std::endl`** — use `"\n"` for performance (except in final summary where flush matters)

### Build system updates needed:
- `CMakeLists.txt`: Add `generate_sample_data` executable target
- `Makefile`: Add `generate-data` target
- Both already have `paper_trader` target (no change needed for that)

---

## 11. Common Pitfalls & Edge Cases

### 11.1 OrderBook has no clear()
**Problem:** You can't reset a book between bars.
**Solution:** Create `books_[symbol] = orderbook::OrderBook(symbol)` each bar. This constructs a fresh book. Use map's `operator[]` assignment or `emplace`.

### 11.2 OrderBook::best_bid() throws if empty
**Problem:** Calling `best_bid()` or `best_ask()` on an empty book throws `std::runtime_error`.
**Solution:** Always call `has_bids()` / `has_asks()` first, or use try-catch in the exchange.

### 11.3 Extracting fills from OrderBook trades
**Problem:** `book.trades()` returns ALL trades since book creation, not just the latest.
**Solution:** Record `book.trade_count()` before submitting, then iterate from that index after.

### 11.4 Position reversal through zero
**Problem:** If long 100 and sell 150, you're now short 50. The avg_cost must be the fill price for the new short, not a blend.
**Solution:** Split into two parts: close 100 (realize P&L), open short 50 (new avg_cost = fill price). See on_fill algorithm in §6.4.

### 11.5 Cash can go negative (margin)
**Problem:** Buying $50k of stock with $20k cash.
**Solution:** Allow negative cash — this is a paper trading sim, not a cash-management system. The risk guard limits notional exposure, not cash balance.

### 11.6 Empty bars in multi-symbol replay
**Problem:** Not all symbols have data for every date (e.g., different listing dates).
**Solution:** `MarketDataReplay::next()` returns only symbols that have a bar for that date. Strategy must handle missing symbols gracefully.

### 11.7 Division by zero in z-score
**Problem:** If all prices in the window are identical, stddev = 0.
**Solution:** Return z = 0 if stddev < 1e-10.

### 11.8 Delta hedge time decay
**Problem:** T_remaining can go to zero or negative.
**Solution:** Stop trading when T_remaining < 0.001. Return empty orders.

### 11.9 Volume formatting
**Problem:** Volume can be very large (45,000,000).
**Solution:** Format as "45M" or "45,000,000" in terminal output using a helper function.

### 11.10 CSV parsing edge cases
**Problem:** Trailing newlines, Windows \r\n line endings, missing fields.
**Solution:** Trim whitespace from each field. Skip lines with fewer than 7 fields. Handle both \n and \r\n.

### 11.11 Pairs trading requires 2+ symbols
**Problem:** PairsTradingStrategy throws if config has < 2 symbols.
**Solution:** Check `config.symbols().size() >= 2` in `init()`. Throw a clear error message.

### 11.12 Hedge ratio can be negative or very large
**Problem:** If two symbols are negatively correlated or barely correlated, `cov(A,B)/var(B)` can be wild.
**Solution:** Clamp hedge ratio to reasonable range (e.g. 0.1 to 10.0) or return 1.0 as default. The current spec returns 1.0 when var(B) < 1e-10.

### 11.13 Gamma scalp straddle expires mid-simulation
**Problem:** The 3-month straddle expires at bar ~63, but data has 252 bars. After expiry the strategy stops trading.
**Solution:** This is correct behavior — after expiry, return empty orders. The remaining bars are idle. The session summary reflects the partial activity.

### 11.14 Backtester creates separate components per run
**Problem:** Each `run_backtest()` call creates its own MarketDataReplay, which re-reads CSV files.
**Solution:** This is acceptable — CSV loading is fast and keeps runs isolated. For performance, the data could be cached, but that's premature optimization.

### 11.15 ReportGenerator round-trip detection is simplified
**Problem:** The `on_fill()` tracking assumes simple open→close round trips. Partial closes, scaling in/out, and reversals produce imprecise trade records.
**Solution:** This is a simplification. The position_tracker handles P&L correctly regardless — the trade records in the report are best-effort for analysis. Document this limitation.

---

## 12. Per-Day Spec File Index

Each spec file contains the COMPLETE code blueprint for that day — every include, every function body, every edge case. The model implementing a day should read ONLY this master plan + the relevant day spec.

| Spec File | Day | Content |
|-----------|-----|---------|
| `docs/specs/day02_market_data.md` | 2 | Bar, CSV parser, MarketDataReplay, generate_sample_data.cpp |
| `docs/specs/day03_simulated_exchange.md` | 3 | SimulatedExchange, liquidity seeding, fill extraction |
| `docs/specs/day04_position_tracker.md` | 4 | PositionTracker, avg cost, P&L, equity snapshots |
| `docs/specs/day05_strategy_mean_reversion.md` | 5 | Strategy base class, MeanReversionStrategy, factory |
| `docs/specs/day06_risk_guard.md` | 6 | RiskGuard, pre-trade checks, kill switch |
| `docs/specs/day07_logging.md` | 7 | TradeLogger, CSV output, terminal formatting, session summary |
| `docs/specs/day08_main_loop.md` | 8 | Full paper_trader.cpp rewrite, CLI parsing, event loop |
| `docs/specs/day09_strategies.md` | 9 | DeltaHedgeStrategy, MomentumStrategy |
| `docs/specs/day10_polish.md` | 10 | SIGINT handler, smoke tests, CMakeLists updates, mid-project checkpoint |
| `docs/specs/day11_pairs_trading.md` | 11 | PairsTradingStrategy — spread z-score, hedge ratio, market-neutral |
| `docs/specs/day12_gamma_scalping.md` | 12 | GammaScalpStrategy — long straddle, delta hedge, gamma/theta attribution |
| `docs/specs/day13_backtesting.md` | 13 | Backtester framework, run_backtest.cpp, comparison report |
| `docs/specs/day14_report_generator.md` | 14 | ReportGenerator — equity curve CSV, trade analysis, rolling metrics |
| `docs/specs/day15_market_maker.md` | 15 | MarketMakerStrategy — capstone: limit orders, spread capture, P&L attribution |

---

## 6.11 PairsTradingStrategy (strategies/pairs_trading.h)

```cpp
class PairsTradingStrategy : public Strategy {
    // Trades the spread between first two symbols
    // spread = price_A - hedge_ratio * price_B
    // hedge_ratio = cov(A,B) / var(B)  (rolling OLS)
    // Signal: z-score of spread over 60-bar window
    // z < -2: BUY A + SELL B  |  z > +2: SELL A + BUY B  |  |z| < 0.5: flatten

    int lookback_ = 60;
    double entry_z_ = 2.0, exit_z_ = 0.5;
    int position_size_ = 100;  // leg A shares; leg B = round(hedge_ratio * position_size_)
};
```

## 6.12 GammaScalpStrategy (strategies/gamma_scalp.h)

```cpp
class GammaScalpStrategy : public Strategy {
    // Simulates long straddle (ATM call + put) on first symbol
    // Delta-hedges by trading shares, profits from realized vol > implied vol
    // Straddle delta = (delta_call + delta_put) * contracts * 100
    // Uses bs::delta_call, bs::gamma, bs::theta_call from greeks/black_scholes.h
    // Prints gamma/theta P&L attribution in on_stop()

    double strike_;  // set to first bar's close
    double sigma_ = 0.20;
    double time_to_expiry_ = 0.25;  // 3 months
    int contracts_ = 5;
    int rebalance_threshold_ = 10;
};
```

## 6.13 Backtester (backtester.h)

```cpp
struct BacktestResult {
    std::string strategy_name, label;
    double total_pnl, pnl_pct, final_equity, realized_pnl, unrealized_pnl;
    double sharpe, sortino, calmar, max_drawdown, volatility;
    int total_trades, rejections;
    size_t bars_processed;
};

// Runs a strategy through the full pipeline silently, returns BacktestResult
BacktestResult run_backtest(const Config& config, std::unique_ptr<Strategy> strategy,
                            const std::string& label = "");

// Prints side-by-side comparison table + Sharpe ranking
void print_comparison(const std::vector<BacktestResult>& results);
```

## 6.14 ReportGenerator (report_generator.h)

```cpp
struct TradeRecord {
    std::string symbol, entry_date, exit_date, direction;
    int quantity;
    double entry_price, exit_price, pnl, pnl_pct;
    int holding_bars;
};

struct TradeAnalysis {
    int total_trades, winning_trades, losing_trades;
    double win_rate, avg_win, avg_loss, largest_win, largest_loss;
    double profit_factor, expectancy;
};

class ReportGenerator {
    void on_fill(const Fill& fill, int position_after, double avg_cost);
    void on_day_end(const std::string& date, double equity, double initial_capital);
    void generate(const std::string& strategy_name);
    // Writes: equity_curve_*.csv, trades_detail_*.csv, rolling_metrics_*.csv
    // Prints: TradeAnalysis summary (win rate, profit factor, expectancy)
};
```

## 6.15 MarketMakerStrategy (strategies/market_maker.h)

```cpp
class MarketMakerStrategy : public Strategy {
    // Capstone strategy — ties ALL modules together
    // Places LIMIT orders (bid below fair, ask above fair) — first strategy to use limit orders
    // Fair price = EMA of close prices
    // Spread width = base_spread * (realized_vol / base_vol) — wider in high vol (vega-driven)
    // Inventory skew = shift quotes to flatten position (delta management)
    // P&L attribution: spread_income (theta-like) vs inventory_cost (gamma-like)
    // Uses: bs:: for vol concepts, risk:: for VaR, orderbook for limit execution

    int ema_period_ = 20;
    double base_spread_bps_ = 15.0;
    int max_inventory_ = 200;
    int quote_size_ = 50;
    double inventory_skew_factor_ = 0.3;
    // Prints attribution in on_stop(): spread income, inventory cost, net
};
```

---

## Appendix A: Include Dependency Graph for New Files

```
trading/config.h          (exists, depends on: nlohmann/json)
trading/market_data.h     (depends on: nothing)
trading/simulated_exchange.h (depends on: market_data.h, config.h, orderbook/orderbook.h)
trading/position_tracker.h   (depends on: simulated_exchange.h [for Fill struct])
trading/strategy.h           (depends on: market_data.h, config.h, simulated_exchange.h, position_tracker.h)
trading/strategies/mean_reversion.h (depends on: strategy.h)
trading/strategies/delta_hedge.h    (depends on: strategy.h, greeks/black_scholes.h)
trading/strategies/momentum.h       (depends on: strategy.h)
trading/strategies/pairs_trading.h  (depends on: strategy.h)
trading/strategies/gamma_scalp.h    (depends on: strategy.h, greeks/black_scholes.h, greeks/greeks_engine.h)
trading/strategies/market_maker.h   (depends on: strategy.h, greeks/black_scholes.h)
trading/risk_guard.h         (depends on: config.h, simulated_exchange.h, position_tracker.h, risk/risk.h)
trading/logging.h            (depends on: market_data.h, config.h, simulated_exchange.h,
                                          position_tracker.h, risk_guard.h, risk/risk.h)
trading/backtester.h         (depends on: ALL trading headers + strategy.h + risk/risk.h)
trading/report_generator.h   (depends on: position_tracker.h, simulated_exchange.h, risk/risk.h)
```

## Appendix B: Compilation Order

```
Day 2:  market_data.h compiles standalone (no deps beyond STL)
Day 3:  simulated_exchange.h needs orderbook.h + market_data.h + config.h
Day 4:  position_tracker.h needs simulated_exchange.h (for Fill)
Day 5:  strategy.h needs all above; mean_reversion.h needs strategy.h
Day 6:  risk_guard.h needs config.h + position_tracker.h + risk.h
Day 7:  logging.h needs everything
Day 8:  paper_trader.cpp includes everything
Day 9:  new strategy headers, same deps as Day 5
Day 10: no new headers (polish checkpoint)
Day 11: pairs_trading.h needs strategy.h (same pattern as Day 5)
Day 12: gamma_scalp.h needs strategy.h + black_scholes.h + greeks_engine.h
Day 13: backtester.h needs ALL trading headers + risk.h
Day 14: report_generator.h needs position_tracker.h + simulated_exchange.h
Day 15: market_maker.h needs strategy.h + black_scholes.h (capstone)
```

## Appendix C: Quick Reference — Existing Function Signatures Used

```cpp
// OrderBook (orderbook/orderbook.h)
orderbook::OrderBook(const std::string& symbol)
OrderId add_limit_order(Side side, double price, uint32_t qty)
OrderId add_market_order(Side side, uint32_t qty)
const std::vector<Trade>& trades() const
size_t trade_count() const
double best_bid() const   // throws if empty
double best_ask() const   // throws if empty
double mid_price() const
bool has_bids() const
bool has_asks() const

// Risk (risk/risk.h)
risk::VaRResult parametric_var(double mu, double sigma, double alpha)
risk::PerformanceMetrics compute_metrics(const std::vector<double>& returns, double rf, double periods)
risk::VaRResult scenario_var(std::vector<double> pnl, double alpha)

// Black-Scholes (greeks/black_scholes.h)
bs::call_price(double S, double K, double r, double T, double sigma)
bs::delta_call(double S, double K, double r, double T, double sigma)
bs::gamma(double S, double K, double r, double T, double sigma)
bs::vega(double S, double K, double r, double T, double sigma)

// Greeks Engine (greeks/greeks_engine.h)
GreeksResult analytic_greeks_call(double S, double K, double r, double T, double sigma)

// Implied Vol (vol/implied_vol.h)
ImpliedVolResult implied_vol_newton(double price, double S, double K, double r, double T, ...)

// Black-Scholes additional (used by gamma_scalp)
bs::theta_call(double S, double K, double r, double T, double sigma)
bs::delta_put(double S, double K, double r, double T, double sigma)  // = delta_call - 1

// Greeks Engine (greeks/greeks_engine.h)
GreeksResult analytic_greeks_call(double S, double K, double r, double T, double sigma)
// GreeksResult has: delta, gamma, vega, theta, rho

// RNG (rng/rng.h)
MersenneTwisterRNG rng(seed)
std::vector<double> generate_normals(RandomNumberGenerator& rng, size_t n)
```

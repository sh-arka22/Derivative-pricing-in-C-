#ifndef QUANTPRICER_TRADING_MARKET_DATA_H
#define QUANTPRICER_TRADING_MARKET_DATA_H

// ============================================================================
// Market Data — OHLCV bars + Option chain snapshots
//
// Bar / MarketDataReplay:    Daily OHLCV from Yahoo Finance CSVs
// OptionQuote / OptionChain: Daily option chain snapshots from CSV
//
// Data source: philippdubach/options-data (Parquet → CSV via download script)
// CSV format:  date,symbol,expiration,strike,type,bid,ask,mark,
//              implied_volatility,delta,gamma,theta,vega,volume,open_interest
// ============================================================================

#include <string>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cmath>

namespace trading {

// ============================================================================
// String Utilities
// ============================================================================

inline std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

inline std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(trim(field));
    }
    return fields;
}

// ============================================================================
// Bar — One OHLCV bar for a single symbol on a single date
// ============================================================================

struct Bar {
    std::string symbol;
    std::string date;       // "YYYY-MM-DD"
    double open;
    double high;
    double low;
    double close;
    uint64_t volume;

    double mid() const { return (high + low) / 2.0; }
    double range() const { return high - low; }
    double body() const { return close - open; }
};

// ============================================================================
// CSV Parser — Reads Yahoo Finance format CSV (OHLCV)
// ============================================================================

inline std::vector<Bar> load_csv(const std::string& filepath, const std::string& symbol) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + filepath);
    }

    std::vector<Bar> bars;
    std::string line;

    // Skip header
    if (!std::getline(ifs, line)) return bars;

    while (std::getline(ifs, line)) {
        line = trim(line);
        if (line.empty()) continue;

        auto fields = split_csv_line(line);
        if (fields.size() < 7) continue;

        try {
            Bar bar;
            bar.symbol = symbol;
            bar.date   = fields[0];
            bar.open   = std::stod(fields[1]);
            bar.high   = std::stod(fields[2]);
            bar.low    = std::stod(fields[3]);
            bar.close  = std::stod(fields[4]);
            bar.volume = static_cast<uint64_t>(std::stod(fields[6]));
            bars.push_back(bar);
        } catch (const std::exception&) {
            continue;
        }
    }

    return bars;
}

// ============================================================================
// MarketDataReplay — Multi-symbol bar-by-bar iteration
// ============================================================================

class MarketDataReplay {
public:
    explicit MarketDataReplay(const std::map<std::string, std::string>& data_files) {
        for (auto& [symbol, filepath] : data_files) {
            auto bars = load_csv(filepath, symbol);
            for (auto& bar : bars) {
                bar_index_[symbol][bar.date] = bar;
                date_set_.insert(bar.date);
            }
            bar_counts_[symbol] = bars.size();
        }
        dates_.assign(date_set_.begin(), date_set_.end());
        std::sort(dates_.begin(), dates_.end());
    }

    bool has_next() const { return current_idx_ < dates_.size(); }

    std::map<std::string, Bar> next() {
        if (!has_next()) throw std::runtime_error("MarketDataReplay: no more bars");
        const std::string& date = dates_[current_idx_];
        std::map<std::string, Bar> result;
        for (auto& [symbol, index] : bar_index_) {
            auto it = index.find(date);
            if (it != index.end()) result[symbol] = it->second;
        }
        current_idx_++;
        return result;
    }

    size_t total_bars() const { return dates_.size(); }
    size_t current_index() const { return current_idx_; }
    const std::string& current_date() const { return dates_[current_idx_]; }

    size_t bars_for_symbol(const std::string& symbol) const {
        auto it = bar_counts_.find(symbol);
        return (it != bar_counts_.end()) ? it->second : 0;
    }

    void print_summary() const {
        std::cout << "Market Data (OHLCV) Loaded:\n";
        for (auto& [sym, count] : bar_counts_)
            std::cout << "  " << sym << ": " << count << " bars\n";
        std::cout << "  Total dates: " << dates_.size() << "\n";
        if (!dates_.empty())
            std::cout << "  Range: " << dates_.front() << " to " << dates_.back() << "\n";
    }

private:
    std::map<std::string, std::map<std::string, Bar>> bar_index_;
    std::set<std::string> date_set_;
    std::vector<std::string> dates_;
    std::map<std::string, size_t> bar_counts_;
    size_t current_idx_ = 0;
};

// ============================================================================
// OptionQuote — One option contract snapshot
// ============================================================================
//
// Represents a single row from the options CSV:
//   date,symbol,expiration,strike,type,bid,ask,mark,
//   implied_volatility,delta,gamma,theta,vega,volume,open_interest
// ============================================================================

struct OptionQuote {
    std::string date;           // Observation date "YYYY-MM-DD"
    std::string symbol;         // Underlying symbol
    std::string expiration;     // Expiry date "YYYY-MM-DD"
    double strike;
    bool is_call;               // true=call, false=put

    double bid;
    double ask;
    double mark;                // Mid-market price

    double implied_vol;         // Annualized implied volatility
    double delta;
    double gamma;
    double theta;
    double vega;

    uint64_t volume;
    uint64_t open_interest;

    // Derived
    double spread() const { return ask - bid; }
    double mid() const { return (bid + ask) / 2.0; }
    double moneyness(double spot) const { return strike / spot; }

    int days_to_expiry_from(const std::string& obs_date) const {
        // Simple date diff: parse YYYY-MM-DD and compute
        // (For exact calendar math, but this is sufficient for trading)
        auto parse = [](const std::string& d) -> int {
            int y = std::stoi(d.substr(0, 4));
            int m = std::stoi(d.substr(5, 2));
            int dd = std::stoi(d.substr(8, 2));
            // Approximate days since epoch
            return y * 365 + y/4 - y/100 + y/400 + m * 30 + dd;
        };
        return parse(expiration) - parse(obs_date);
    }
};

// ============================================================================
// OptionChain — All options for one symbol on one date
// ============================================================================

struct OptionChain {
    std::string date;
    std::string symbol;
    std::vector<OptionQuote> quotes;

    // Filter helpers
    std::vector<OptionQuote> calls() const {
        std::vector<OptionQuote> result;
        for (auto& q : quotes) if (q.is_call) result.push_back(q);
        return result;
    }

    std::vector<OptionQuote> puts() const {
        std::vector<OptionQuote> result;
        for (auto& q : quotes) if (!q.is_call) result.push_back(q);
        return result;
    }

    std::vector<OptionQuote> by_expiration(const std::string& exp) const {
        std::vector<OptionQuote> result;
        for (auto& q : quotes) if (q.expiration == exp) result.push_back(q);
        return result;
    }

    // Find ATM option (closest strike to spot)
    const OptionQuote* atm_call(double spot) const {
        return closest_strike(spot, true);
    }
    const OptionQuote* atm_put(double spot) const {
        return closest_strike(spot, false);
    }

    // All unique expirations
    std::vector<std::string> expirations() const {
        std::set<std::string> exps;
        for (auto& q : quotes) exps.insert(q.expiration);
        return {exps.begin(), exps.end()};
    }

    // All unique strikes for a given expiration
    std::vector<double> strikes(const std::string& exp) const {
        std::set<double> stks;
        for (auto& q : quotes)
            if (q.expiration == exp) stks.insert(q.strike);
        return {stks.begin(), stks.end()};
    }

    size_t size() const { return quotes.size(); }
    bool empty() const { return quotes.empty(); }

private:
    const OptionQuote* closest_strike(double spot, bool call) const {
        const OptionQuote* best = nullptr;
        double best_dist = 1e18;
        for (auto& q : quotes) {
            if (q.is_call != call) continue;
            double dist = std::abs(q.strike - spot);
            if (dist < best_dist) {
                best_dist = dist;
                best = &q;
            }
        }
        return best;
    }
};

// ============================================================================
// Options CSV Parser
// ============================================================================
// Expected format (from download_options_data.py):
//   date,symbol,expiration,strike,type,bid,ask,mark,implied_volatility,
//   delta,gamma,theta,vega,volume,open_interest
//
// Column indices:
//   0=date, 1=symbol, 2=expiration, 3=strike, 4=type, 5=bid, 6=ask,
//   7=mark, 8=iv, 9=delta, 10=gamma, 11=theta, 12=vega, 13=volume, 14=oi
// ============================================================================

inline std::vector<OptionQuote> load_options_csv(const std::string& filepath) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open options CSV: " + filepath);
    }

    std::vector<OptionQuote> quotes;
    std::string line;

    // Skip header
    if (!std::getline(ifs, line)) return quotes;

    while (std::getline(ifs, line)) {
        line = trim(line);
        if (line.empty()) continue;

        auto f = split_csv_line(line);
        if (f.size() < 15) continue;

        try {
            OptionQuote q;
            q.date        = f[0];
            q.symbol      = f[1];
            q.expiration  = f[2];
            q.strike      = std::stod(f[3]);
            q.is_call     = (f[4] == "call");
            q.bid         = std::stod(f[5]);
            q.ask         = std::stod(f[6]);
            q.mark        = std::stod(f[7]);
            q.implied_vol = std::stod(f[8]);
            q.delta       = std::stod(f[9]);
            q.gamma       = std::stod(f[10]);
            q.theta       = std::stod(f[11]);
            q.vega        = std::stod(f[12]);
            q.volume      = static_cast<uint64_t>(std::stod(f[13]));
            q.open_interest = static_cast<uint64_t>(std::stod(f[14]));
            quotes.push_back(q);
        } catch (const std::exception&) {
            continue;
        }
    }

    return quotes;
}

// ============================================================================
// OptionDataReplay — Multi-symbol option chain iteration by date
// ============================================================================
// Loads option CSVs upfront, indexes by (symbol, date).
// On each date, returns a map of symbol → OptionChain for that day.
// ============================================================================

class OptionDataReplay {
public:
    // option_files: symbol → CSV path (e.g. "AAPL" → "data/options/AAPL_options.csv")
    explicit OptionDataReplay(const std::map<std::string, std::string>& option_files) {
        for (auto& [symbol, filepath] : option_files) {
            auto quotes = load_options_csv(filepath);
            for (auto& q : quotes) {
                chain_index_[symbol][q.date].push_back(q);
                date_set_.insert(q.date);
            }
            quote_counts_[symbol] = quotes.size();
        }
        dates_.assign(date_set_.begin(), date_set_.end());
        std::sort(dates_.begin(), dates_.end());
    }

    // Get option chain for a symbol on a specific date
    OptionChain get_chain(const std::string& symbol, const std::string& date) const {
        OptionChain chain;
        chain.date = date;
        chain.symbol = symbol;

        auto sym_it = chain_index_.find(symbol);
        if (sym_it == chain_index_.end()) return chain;

        auto date_it = sym_it->second.find(date);
        if (date_it == sym_it->second.end()) return chain;

        chain.quotes = date_it->second;
        return chain;
    }

    // Get all chains for a given date
    std::map<std::string, OptionChain> get_chains(const std::string& date) const {
        std::map<std::string, OptionChain> result;
        for (auto& [symbol, date_map] : chain_index_) {
            auto it = date_map.find(date);
            if (it != date_map.end()) {
                OptionChain chain;
                chain.date = date;
                chain.symbol = symbol;
                chain.quotes = it->second;
                result[symbol] = chain;
            }
        }
        return result;
    }

    bool has_data(const std::string& symbol, const std::string& date) const {
        auto sym_it = chain_index_.find(symbol);
        if (sym_it == chain_index_.end()) return false;
        return sym_it->second.count(date) > 0;
    }

    const std::vector<std::string>& dates() const { return dates_; }
    size_t total_dates() const { return dates_.size(); }

    void print_summary() const {
        std::cout << "Option Chain Data Loaded:\n";
        for (auto& [sym, count] : quote_counts_) {
            // Count unique dates for this symbol
            auto it = chain_index_.find(sym);
            size_t n_dates = (it != chain_index_.end()) ? it->second.size() : 0;
            std::cout << "  " << sym << ": " << count << " quotes across "
                      << n_dates << " dates\n";
        }
        std::cout << "  Total dates: " << dates_.size() << "\n";
        if (!dates_.empty())
            std::cout << "  Range: " << dates_.front() << " to " << dates_.back() << "\n";
    }

private:
    // symbol → (date → vector<OptionQuote>)
    std::map<std::string, std::map<std::string, std::vector<OptionQuote>>> chain_index_;
    std::set<std::string> date_set_;
    std::vector<std::string> dates_;
    std::map<std::string, size_t> quote_counts_;
};

} // namespace trading

#endif // QUANTPRICER_TRADING_MARKET_DATA_H

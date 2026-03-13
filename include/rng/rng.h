#ifndef QUANTPRICER_RNG_H
#define QUANTPRICER_RNG_H

// ============================================================================
// Random Number Generation — Book Chapter 14
// Ch 14.1: Overview of RNG needs in Monte Carlo
// Ch 14.2: RNG class hierarchy with abstract base, LCG implementation
// Ch 14.3: Statistical distribution hierarchy, Standard Normal via
//          Box-Muller and inverse CDF (rational approximation)
// Also: Ch 16.2-16.4 — Correlated path generation using Cholesky
// ============================================================================

#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

/// Abstract base class for uniform RNG — Ch 14.2
class RandomNumberGenerator {
public:
    virtual ~RandomNumberGenerator() = default;
    virtual double generate_uniform() = 0;      // U(0,1)
    virtual void seed(unsigned long s) = 0;
    virtual void reset() = 0;
};

/// Linear Congruential Generator — Ch 14.2.1, 14.2.2
/// x_{n+1} = (a * x_n + c) mod m
/// Simple but has known deficiencies — included for educational value
class LinearCongruentialGenerator : public RandomNumberGenerator {
public:
    explicit LinearCongruentialGenerator(unsigned long seed = 1)
        : seed_(seed), current_(seed),
          a_(1103515245UL), c_(12345UL), m_(2147483648UL) {}  // glibc params

    double generate_uniform() override {
        current_ = (a_ * current_ + c_) % m_;
        return static_cast<double>(current_) / static_cast<double>(m_);
    }

    void seed(unsigned long s) override { seed_ = s; current_ = s; }
    void reset() override { current_ = seed_; }

private:
    unsigned long seed_, current_;
    const unsigned long a_, c_, m_;
};

/// Mersenne Twister wrapper — production-quality RNG
/// Uses std::mt19937_64 from <random> — what you'd actually use in practice
class MersenneTwisterRNG : public RandomNumberGenerator {
public:
    explicit MersenneTwisterRNG(unsigned long seed = 42)
        : seed_(seed), engine_(seed), dist_(0.0, 1.0) {}

    double generate_uniform() override { return dist_(engine_); }
    void seed(unsigned long s) override { seed_ = s; engine_.seed(s); }
    void reset() override { engine_.seed(seed_); }

private:
    unsigned long seed_;
    std::mt19937_64 engine_;
    std::uniform_real_distribution<double> dist_;
};

// ============================================================================
// Statistical Distribution Hierarchy — Ch 14.3
// ============================================================================

/// Abstract base for statistical distributions — Ch 14.3.1
class StatisticalDistribution {
public:
    virtual ~StatisticalDistribution() = default;
    virtual double pdf(double x) const = 0;
    virtual double cdf(double x) const = 0;
    virtual double inv_cdf(double p) const = 0;  // Quantile function
    virtual double mean() const = 0;
    virtual double variance() const = 0;
};

/// Standard Normal Distribution N(0,1) — Ch 14.3.2
/// CDF via rational approximation (Abramowitz & Stegun)
/// Inverse CDF via Beasley-Springer-Moro algorithm
class StandardNormalDistribution : public StatisticalDistribution {
public:
    double pdf(double x) const override {
        return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
    }

    /// CDF: Φ(x) using the complementary error function — Ch 14.3.2
    double cdf(double x) const override {
        return 0.5 * std::erfc(-x * M_SQRT1_2);
    }

    /// Inverse CDF (Beasley-Springer-Moro approximation)
    /// Used to transform uniform samples to normal samples
    double inv_cdf(double p) const override {
        if (p <= 0.0) return -1e10;
        if (p >= 1.0) return 1e10;

        // Rational approximation for central region
        if (p > 0.5) return -inv_cdf(1.0 - p);

        double t = std::sqrt(-2.0 * std::log(p));
        // Coefficients from Abramowitz & Stegun
        double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
        double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
        return -(t - (c0 + c1 * t + c2 * t * t) /
                     (1.0 + d1 * t + d2 * t * t + d3 * t * t * t));
    }

    double mean() const override { return 0.0; }
    double variance() const override { return 1.0; }
};

// ============================================================================
// Sampling Utilities for Monte Carlo — connects Ch 14 to Ch 10, 12, 15, 16
// ============================================================================

/// Generate a vector of standard normal samples using Box-Muller — Ch 14.3.2
inline std::vector<double> generate_normals(RandomNumberGenerator& rng, size_t n) {
    std::vector<double> normals(n);
    for (size_t i = 0; i < n; i += 2) {
        double u1 = rng.generate_uniform();
        double u2 = rng.generate_uniform();
        // Avoid log(0)
        u1 = std::max(u1, 1e-15);
        double r = std::sqrt(-2.0 * std::log(u1));
        double theta = 2.0 * M_PI * u2;
        normals[i] = r * std::cos(theta);
        if (i + 1 < n) normals[i + 1] = r * std::sin(theta);
    }
    return normals;
}

/// Generate correlated normal pairs using Cholesky — Ch 16.2, 16.4
/// Given correlation rho, produces (Z1, Z2) where Corr(Z1,Z2) = rho
inline std::pair<double, double> generate_correlated_normals(
    RandomNumberGenerator& rng, double rho) {
    auto z = generate_normals(rng, 2);
    double z1 = z[0];
    double z2 = rho * z[0] + std::sqrt(1.0 - rho * rho) * z[1];  // Cholesky
    return {z1, z2};
}

/// Antithetic variates — variance reduction technique (Ch 10 extension)
inline std::vector<double> generate_antithetic_normals(
    RandomNumberGenerator& rng, size_t n) {
    auto normals = generate_normals(rng, n);
    std::vector<double> result(2 * n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = normals[i];
        result[n + i] = -normals[i];  // Antithetic pair
    }
    return result;
}

#endif // QUANTPRICER_RNG_H
